"""O prazo duro de um trabalho de GPU, e a morte controlada quando ele estoura.

Compartilhado pelas rotas que rodam modelo por minutos (imagem, transcricao).
O teto "macio" de cada rota so e conferido entre passos — entre passos da
difusao, entre segmentos do audio. Um processo travado na carga do modelo, ou
dentro de um passo, nunca chega a conferir, e segurava a placa (e o texto de
todos) por tempo indefinido.

Passado o prazo duro, o worker se considera QUEBRADO: responde 500 com
`error.code` e sai com codigo 1, para o systemd subir um processo limpo. Nao
ha como cancelar a thread travada — Python nao sabe fazer isso, e o torch e o
CTranslate2 menos ainda —, e um processo que continua com ela pendurada nao e
confiavel.
"""

from __future__ import annotations

import os
import threading


class WorkerTravado(RuntimeError):
    """O trabalho passou do prazo duro: o processo nao e confiavel."""


def com_prazo_duro(trabalho, segundos: float, rotulo: str, variavel: str):
    """Roda `trabalho` numa thread e desiste dela depois de `segundos`.

    `segundos <= 0` desliga: roda direto, na thread de quem chamou. Desistir
    so faz sentido junto com matar o processo, que e o que quem chama faz.
    """
    if segundos <= 0:
        return trabalho()

    resultado: dict = {}

    def alvo():
        try:
            resultado["ok"] = trabalho()
        except BaseException as exc:  # repassada inteira: o OOM de CUDA inclusive
            resultado["erro"] = exc

    thread = threading.Thread(target=alvo, name=rotulo, daemon=True)
    thread.start()
    thread.join(segundos)

    if thread.is_alive():
        raise WorkerTravado(
            f"o trabalho de {rotulo} passou de {segundos:g}s ({variavel}) sem terminar "
            f"nem falhar: travou na carga do modelo ou dentro de um passo. Causa comum: "
            f"memoria. Veja `journalctl --user -u worker-gpu` e o `memory.events` da unit."
        )
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["ok"]


def morrer_em_seguida(atraso: float = 2.0) -> None:
    """Encerra o processo com erro, depois de a resposta 500 sair.

    Codigo 1, e nao 0: e o que o systemd le como falha, e o que dispara o
    `Restart=always` e o `OnFailure=` (o aviso de queda).
    """
    temporizador = threading.Timer(atraso, os._exit, args=(1,))
    temporizador.daemon = True
    temporizador.start()
