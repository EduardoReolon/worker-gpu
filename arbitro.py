"""Quem pode usar a GPU agora.

Este modulo e a razao de o worker existir como um processo so. A placa e um
recurso indivisivel: um modelo de texto de 30B ja ocupa quase toda a VRAM de
8 GB, e a difusao precisa de ~5,4 GB. Nao cabem juntos, e o que acontece
quando se tenta nao e um erro — e o processo caindo para CPU em silencio,
dezenas de vezes mais lento, sem nada no log dizendo por que.

## Um lock, um processo

O lock e um `threading.Lock` em memoria, e isso so e correto porque TODO
pedido de GPU entra por aqui: texto, imagem e conversao no mesmo processo. Se
os tres fossem servicos separados, um lock em memoria protegeria cada um de si
mesmo e nenhum dos outros — que era exatamente a situacao anterior.

## 503, e nao fila

Quem nao pega o lock recebe `503` com `Retry-After`. Nao ha fila aqui dentro,
de proposito: este processo nao tem estado duravel, e uma fila em memoria
perde trabalho no primeiro restart. Os clientes (PubliBot, CRM) ja tem fila
persistente e sabem retomar — enfileirar aqui seria uma segunda fila,
invisivel para eles e pior que a que ja tem.

O `Retry-After` e CALCULADO a partir do que esta rodando, e nao uma constante.
Mandar o cliente voltar em 60 segundos quando faltava meio e desperdicio dos
dois lados; mandar voltar em 60 quando faltava um minuto e meio e um segundo
503 garantido.

## O que ele NAO resolve

Justica. Um cliente que pede texto a cada minuto pode, na pratica, nunca
deixar espaco para um pedido de imagem que leva um minuto. Nao ha prioridade
nem envelhecimento aqui — se isso aparecer em uso, o lugar de resolver e este
arquivo, e a solucao provavelmente e uma reserva com hora marcada, nao uma
fila.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

from config import DURACAO_ESTIMADA, ESPERA_PELO_LOCK

logger = logging.getLogger("worker-gpu.arbitro")


class GpuOcupada(RuntimeError):
    """A placa esta com outro trabalho. Traz quanto falta, em segundos.

    `modelo` e o que esta carregado agora, quando se sabe. Ele viaja no 503
    porque e ali que ele decide alguma coisa: o cliente que recebeu a recusa
    esta escolhendo o que mandar em seguida, e mandar um pedido do modelo que
    ja esta na placa nao paga a troca.
    """

    def __init__(self, ocupante: str, falta: int, modelo: str | None = None):
        super().__init__(f"a GPU esta em uso por {ocupante!r}; tente em {falta}s")
        self.ocupante = ocupante
        self.falta = falta
        self.modelo = modelo


@dataclass(frozen=True)
class Ocupacao:
    """O que esta rodando agora."""

    tarefa: str
    desde: float
    # O modelo em uso, quando a tarefa tem um. Vem de quem pediu, e nao de
    # uma consulta ao Ollama: perguntar `/api/ps` a cada 503 seria uma viagem
    # a mais para saber o que este processo ja sabe.
    modelo: str | None = None

    @property
    def ha_quantos_segundos(self) -> int:
        return int(time.monotonic() - self.desde)

    @property
    def falta_estimado(self) -> int:
        """Quanto ainda deve levar. Nunca menos de 5s: um `Retry-After: 0`
        convida o cliente a voltar imediatamente e levar outro 503."""
        estimado = DURACAO_ESTIMADA.get(self.tarefa, 60)
        return max(5, estimado - self.ha_quantos_segundos)


class Arbitro:
    """O dono da GPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ocupacao: Ocupacao | None = None
        # Protege a leitura de `_ocupacao` por quem NAO tem o lock da GPU —
        # o `/health/` e quem leva o 503 precisam ler isso sem esperar.
        self._mural = threading.Lock()

    @property
    def ocupacao(self) -> Ocupacao | None:
        with self._mural:
            return self._ocupacao

    @property
    def ocupada(self) -> bool:
        return self.ocupacao is not None

    @contextmanager
    def usar(self, tarefa: str, *, modelo: str | None = None, espera: float | None = None):
        """Toma a GPU, ou levanta `GpuOcupada`.

        `tarefa` e o rotulo que aparece no 503 e no `/health/`: e o que
        transforma "ocupado" em "ocupado gerando imagem ha 20 segundos".
        `modelo` acrescenta *qual* modelo, para quem esta escolhendo o proximo
        pedido.
        """
        limite = ESPERA_PELO_LOCK if espera is None else espera

        if limite > 0:
            tomou = self._lock.acquire(timeout=limite)
        else:
            tomou = self._lock.acquire(blocking=False)

        if not tomou:
            # Le a ocupacao DEPOIS de falhar: entre a tentativa e a leitura o
            # dono pode ter mudado, e o rotulo errado num 503 manda procurar
            # no lugar errado. Errar para "outro trabalho" e honesto.
            atual = self.ocupacao
            raise GpuOcupada(
                atual.tarefa if atual else "outro trabalho",
                atual.falta_estimado if atual else 30,
                atual.modelo if atual else None,
            )

        with self._mural:
            self._ocupacao = Ocupacao(tarefa=tarefa, desde=time.monotonic(), modelo=modelo)

        comeco = time.monotonic()
        try:
            yield
        finally:
            duracao = time.monotonic() - comeco
            with self._mural:
                self._ocupacao = None
            self._lock.release()
            logger.info("%s terminou em %.1fs", tarefa, duracao)


# Uma instancia para o processo. Nao e "singleton por elegancia": duas
# instancias seriam dois locks, e dois locks sobre uma placa nao protegem nada.
ARBITRO = Arbitro()
