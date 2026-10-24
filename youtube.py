"""Rota de legenda do YouTube: `POST /v1/youtube/legenda`. NAO usa a placa.

Existe por causa de ONDE esta maquina esta, e nao do que ela tem: o YouTube
costuma recusar legenda pedida de IP de nuvem (`RequestBlocked`, `IpBlocked`),
e a conexao daqui e residencial. Por isso nao entra no lock da GPU — um pedido
de legenda nao disputa nada com texto, imagem ou transcricao.

A biblioteca e a mesma do servidor do PubliBot, `youtube-transcript-api` 1.x,
para a legenda sair igual pelos dois caminhos.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import YOUTUBE_TIMEOUT
from seguranca import conferir

logger = logging.getLogger("worker-gpu.youtube")

router = APIRouter()

# O id de video do YouTube: 11 caracteres de base64 "url-safe". Conferir aqui
# poupa uma ida ao YouTube para um pedido que nao tem como dar certo.
_ID_DE_VIDEO = re.compile(r"[A-Za-z0-9_-]{11}")


class PedidoDeLegenda(BaseModel):
    video_id: str
    # Em ordem de preferencia.
    idiomas: list[str] = ["pt", "pt-BR", "en"]


def conferir_configuracao() -> None:
    import importlib.util

    if importlib.util.find_spec("youtube_transcript_api") is None:
        raise RuntimeError(
            "YOUTUBE_ATIVA esta ligado, mas o youtube-transcript-api nao esta instalado.\n"
            "  ./venv/bin/pip install -r requirements.txt\n"
            "ou YOUTUBE_ATIVA=nao no .env."
        )


def _erro(status: int, codigo: str, mensagem: str, *, retry_after: int | None = None):
    cabecalhos = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return JSONResponse(
        {"error": {"code": codigo, "message": mensagem}}, status_code=status, headers=cabecalhos
    )


def _sessao():
    """Uma `requests.Session` com teto de tempo em TODA ida.

    A biblioteca nao passa `timeout` nenhum ao `requests`: uma conexao que para
    de responder prenderia a requisicao — e a thread do servidor — para sempre.
    """
    from requests import Session

    class _SessaoComPrazo(Session):
        def request(self, *argumentos, **nomeados):
            nomeados.setdefault("timeout", YOUTUBE_TIMEOUT)
            return super().request(*argumentos, **nomeados)

    return _SessaoComPrazo()


def buscar_legenda(video_id: str, idiomas: list[str]):
    """A primeira legenda que existir, na ordem de `idiomas`.

    Dentro de cada idioma, a MANUAL antes da automatica: quem escreveu a
    legenda a mao acertou os nomes proprios que o reconhecimento de fala erra.
    Mas um idioma preferido com legenda automatica vence um idioma seguinte
    com legenda manual — a ordem de `idiomas` e do cliente.
    """
    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi

    lista = YouTubeTranscriptApi(http_client=_sessao()).list(video_id)
    for idioma in idiomas:
        for procurar in (lista.find_manually_created_transcript, lista.find_generated_transcript):
            try:
                legenda = procurar([idioma])
            except NoTranscriptFound:
                continue
            return legenda, legenda.fetch()

    raise NoTranscriptFound(video_id, idiomas, lista)


@router.post("/v1/youtube/legenda", dependencies=[Depends(conferir)])
def legenda(pedido: PedidoDeLegenda):
    """`def` e nao `async def`: a biblioteca e sincrona e vai a rede."""
    from youtube_transcript_api import (
        AgeRestricted,
        InvalidVideoId,
        NoTranscriptFound,
        PoTokenRequired,
        RequestBlocked,
        TranscriptsDisabled,
        VideoUnavailable,
        VideoUnplayable,
    )

    if not _ID_DE_VIDEO.fullmatch(pedido.video_id):
        return _erro(
            422, "video_id_invalido", f"video_id={pedido.video_id!r} nao e um id do YouTube."
        )
    idiomas = [idioma.strip() for idioma in pedido.idiomas if idioma.strip()]
    if not idiomas:
        return _erro(422, "entrada_invalida", "idiomas vazio.")

    try:
        escolhida, trechos = buscar_legenda(pedido.video_id, idiomas)
    except (NoTranscriptFound, TranscriptsDisabled, AgeRestricted, VideoUnplayable) as exc:
        # Sem legenda a ler. A restricao de idade e o video "nao reproduzivel"
        # entram aqui, e nao em `bloqueado`: tentar do servidor nao muda nada,
        # e o caminho que resta — a pessoa mandar o audio — continua valendo.
        return _erro(404, "sem_legenda", f"{type(exc).__name__}: {_primeira_linha(exc)}")
    except (VideoUnavailable, InvalidVideoId) as exc:
        # O video nao existe (ou foi removido): nem legenda, nem audio.
        return _erro(404, "video_indisponivel", f"{type(exc).__name__}: {_primeira_linha(exc)}")
    except (RequestBlocked, PoTokenRequired) as exc:
        logger.warning(
            "O YouTube recusou a legenda de %s daqui: %s", pedido.video_id, type(exc).__name__
        )
        return _erro(
            503, "bloqueado", f"{type(exc).__name__}: {_primeira_linha(exc)}", retry_after=3600
        )
    except Exception as exc:
        # Rede, tempo esgotado, resposta que a biblioteca nao entendeu. Nao e
        # o video, e nao e bloqueio declarado: e transitorio.
        logger.warning("Falha ao buscar a legenda de %s: %s", pedido.video_id, exc)
        return _erro(
            503,
            "youtube_indisponivel",
            f"{type(exc).__name__}: {_primeira_linha(exc)}",
            retry_after=300,
        )

    segmentos = [
        {"start": trecho.start, "duration": trecho.duration, "text": trecho.text}
        for trecho in trechos
    ]
    logger.info(
        "Legenda de %s: %s (%s), %s segmentos.",
        pedido.video_id,
        escolhida.language_code,
        "automatica" if escolhida.is_generated else "manual",
        len(segmentos),
    )
    return {
        "video_id": pedido.video_id,
        "idioma": escolhida.language_code,
        "gerada_automaticamente": escolhida.is_generated,
        "segmentos": segmentos,
    }


def _primeira_linha(exc: BaseException) -> str:
    """As mensagens da biblioteca sao paragrafos de ajuda; a primeira linha
    util basta para o log do cliente."""
    for linha in str(exc).splitlines():
        if linha.strip():
            return linha.strip()[:300]
    return ""
