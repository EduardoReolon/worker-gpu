"""Quais modelos existem nesta maquina, e o que fazer quando nao existem.

Com um modelo por inquilino, lembrar de dar `ollama pull` em cada um, a cada
maquina nova, nao escala: com dez inquilinos o que acontece e esquecer um, e o
esquecimento aparece como 404 no meio de um lote.

Este modulo baixa sozinho. O desenho e o que os clientes ja sabem tratar:

    pedido de modelo ausente  ->  dispara o download EM SEGUNDO PLANO
                              ->  responde 503 `baixando_modelo` + Retry-After
    pedidos seguintes         ->  503 com o progresso, ate terminar
    terminou                  ->  o proximo pedido passa normalmente
    falhou                    ->  404, e o cliente DESISTE

## Fora do lock da GPU, sempre

Um download e rede e disco. Tomar o lock para baixar prenderia a placa por
dezenas de minutos — a imagem e a conversao ficariam em 503 por causa de um
modelo de texto que nem carregou ainda. Por isso a conferencia acontece ANTES
de `ARBITRO.usar()`, e a thread de download nao encosta no arbitro.

## O laco infinito, e as tres coisas que o impedem

O modo de este modulo dar muito errado e sempre o mesmo: o cliente pede X, o
worker acha que X nao existe, baixa, e continua achando que X nao existe. Aí
sao downloads em laco ate o disco acabar. Tres defesas:

1. **Tag implicita.** O `/api/tags` lista `qwen2.5:7b-instruct`; um pedido de
   `qwen2.5` refere-se a `qwen2.5:latest`. Comparar cru nunca casaria, e o
   worker baixaria o mesmo modelo para sempre. Por isso `_normalizar()`.
2. **Conferencia depois do download.** Terminado o pull, o catalogo e lido de
   novo. Se o modelo ainda nao aparece, isso e uma FALHA — e nao "baixou" —,
   porque repetir daria no mesmo.
3. **Memoria das falhas.** Um download que falhou e lembrado por
   `FALHA_DE_DOWNLOAD_LEMBRADA` segundos, e nesse tempo o pedido leva 404 em
   vez de disparar outro. Lembrar para sempre exigiria reiniciar o servico
   depois de uma queda de rede; nao lembrar nada e o laco.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import ollama
from config import BAIXAR_MODELO_AUTOMATICO, FALHA_DE_DOWNLOAD_LEMBRADA

logger = logging.getLogger("worker-gpu.modelos")


class Baixando(RuntimeError):
    """O modelo esta vindo. Transitorio, e com prazo: vale adiar.

    `tentar_em` sai do progresso real do download. Um `Retry-After` fixo diria
    "volte em 60s" com 5% baixados, e seria um 503 garantido.
    """

    def __init__(self, nome: str, porcento: float, tentar_em: int):
        super().__init__(f"o modelo {nome!r} esta sendo baixado ({porcento:.0f}%)")
        self.nome = nome
        self.porcento = porcento
        self.tentar_em = tentar_em


class NaoDisponivel(RuntimeError):
    """O modelo nao existe e nao da para obter. Repetir nao muda nada."""


@dataclass
class _Download:
    nome: str
    comeco: float
    porcento: float = 0.0

    @property
    def ha_segundos(self) -> int:
        return int(time.monotonic() - self.comeco)

    @property
    def falta_estimado(self) -> int:
        """Quanto ainda deve levar, pelo ritmo ate agora.

        Entre 15 e 600 segundos. O piso evita o cliente voltar a cada instante
        no comeco, quando o progresso ainda e ruido; o teto evita uma tarefa
        dormindo por horas quando a estimativa sai absurda.
        """
        if self.porcento < 1:
            return 60

        total = self.ha_segundos * 100.0 / self.porcento
        return max(15, min(600, int(total - self.ha_segundos)))


_trava = threading.Lock()
# Modelos que ja confirmamos no disco. Uma vez verdadeiro, sempre verdadeiro:
# pesos nao se desbaixam sozinhos, e um `/api/tags` por pedido seria uma
# viagem a mais em todo o lote para saber o que ja sabemos.
_no_disco: set[str] = set()
_baixando: dict[str, _Download] = {}
_falhas: dict[str, tuple[str, float]] = {}


def _normalizar(nome: str) -> str:
    """`qwen2.5` -> `qwen2.5:latest`, que e como o catalogo o chama.

    Sem isto, um pedido sem tag nunca casaria com o `/api/tags` e o worker
    baixaria o mesmo modelo indefinidamente.
    """
    nome = (nome or "").strip()
    if not nome or ":" in nome.rsplit("/", 1)[-1]:
        return nome

    return f"{nome}:latest"


def conferir(nome: str | None) -> None:
    """Deixa passar, ou levanta `Baixando` / `NaoDisponivel`.

    Chamado ANTES de tomar o lock da GPU: um download nao usa placa, e segurar
    o lock enquanto ele acontece deixaria a imagem e a conversao em 503 por
    causa de um modelo de texto.
    """
    if not nome:
        # Sem `model` no corpo, quem decide e o Ollama — e a mensagem de erro
        # dele e melhor que qualquer palpite daqui.
        return

    alvo = _normalizar(nome)

    with _trava:
        if alvo in _no_disco:
            return

        emprogresso = _baixando.get(alvo)
        if emprogresso:
            raise Baixando(nome, emprogresso.porcento, emprogresso.falta_estimado)

        mensagem = _falha_lembrada(alvo)
        if mensagem:
            raise NaoDisponivel(mensagem)

    if alvo in {_normalizar(existente) for existente in ollama.modelos_no_disco()}:
        with _trava:
            _no_disco.add(alvo)
        return

    if not BAIXAR_MODELO_AUTOMATICO:
        # Comportamento de antes: o Ollama responde 404 e a mensagem dele
        # chega ao cliente intacta.
        return

    with _trava:
        # De novo com a trava: entre a leitura do catalogo e agora, outro
        # pedido do mesmo modelo pode ter comecado o download. Dois pulls do
        # mesmo modelo nao quebram nada no Ollama, mas gastam banda em dobro e
        # poluem o `/health/`.
        if alvo in _baixando:
            emprogresso = _baixando[alvo]
            raise Baixando(nome, emprogresso.porcento, emprogresso.falta_estimado)
        if alvo in _no_disco:
            return

        _baixando[alvo] = _Download(nome=alvo, comeco=time.monotonic())

    _disparar(alvo)

    raise Baixando(nome, 0.0, 60)


def _disparar(alvo: str) -> None:
    """Sai numa thread propria. Isolado para o teste poder rodar sincrono."""
    threading.Thread(target=_baixar_ate_o_fim, args=(alvo,), daemon=True).start()


def _baixar_ate_o_fim(alvo: str) -> None:
    logger.info("Baixando %s em segundo plano; a placa segue livre.", alvo)

    def progrediu(porcento: float) -> None:
        with _trava:
            if alvo in _baixando:
                _baixando[alvo].porcento = porcento

    try:
        ollama.baixar(alvo, progrediu)
    except Exception as erro:
        logger.exception("Falhou o download de %s", alvo)
        _registrar_falha(alvo, f"nao foi possivel baixar o modelo {alvo!r}: {erro}")
        return

    # A conferencia que impede o laco. Se o Ollama disse que baixou e o modelo
    # nao aparece no catalogo, chamar isso de sucesso faria o proximo pedido
    # disparar outro download, e o seguinte tambem.
    if alvo in {_normalizar(nome) for nome in ollama.modelos_no_disco()}:
        with _trava:
            _no_disco.add(alvo)
            _baixando.pop(alvo, None)
        logger.info("%s baixado e disponivel.", alvo)
        return

    _registrar_falha(
        alvo,
        f"o download de {alvo!r} terminou sem erro, mas o modelo nao aparece no "
        f"catalogo do Ollama. Confira o nome.",
    )


def _registrar_falha(alvo: str, mensagem: str) -> None:
    with _trava:
        _baixando.pop(alvo, None)
        _falhas[alvo] = (mensagem, time.monotonic())


def _falha_lembrada(alvo: str) -> str | None:
    registro = _falhas.get(alvo)
    if not registro:
        return None

    mensagem, quando = registro
    if time.monotonic() - quando < FALHA_DE_DOWNLOAD_LEMBRADA:
        return mensagem

    # Passou o prazo: esquece, e a proxima tentativa vale. E o que permite a
    # maquina se curar de uma queda de rede sem um restart.
    _falhas.pop(alvo, None)
    return None


def estado() -> list[dict]:
    """O que o `/health/` mostra: downloads em curso.

    Sem isto, um lote inteiro levando `baixando_modelo` nao teria como ser
    distinguido, de fora, de um worker travado.
    """
    with _trava:
        return [
            {
                "modelo": item.nome,
                "porcento": round(item.porcento, 1),
                "ha_segundos": item.ha_segundos,
            }
            for item in _baixando.values()
        ]


def esquecer_tudo() -> None:
    """So para os testes: o estado e de processo, e vaza entre eles."""
    with _trava:
        _no_disco.clear()
        _baixando.clear()
        _falhas.clear()
