"""Rota de voz: `POST /v1/audio/speech`, no dialeto da OpenAI, pelo Kokoro.

O PubliBot narra as laminas dos videos das redes por aqui (uma fala por
lamina), sem pagar API. O mesmo cliente fala com a OpenAI (`gpt-4o-mini-tts`)
trocando o endereco.

- **Nao usa a placa.** O Kokoro (82M) roda na CPU mais rapido que tempo real,
  entao nao passa pelo arbitro: narrar nao espera uma imagem acabar, e nao
  tira o modelo de texto da VRAM.
- **Processo a parte**, como a animacao: o Kokoro mora no ambiente dele
  (`VOZ_DIR`, instalado pelo `deploy/instalar_voz.sh`).
- **A voz**: a do pedido, se for do Kokoro (`pf_dora`, `pm_alex`...); uma da
  OpenAI (`alloy`, `nova`) ou vazia vira `VOZ_PADRAO`.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import respostas
from config import (
    VOZ_DIR,
    VOZ_PADRAO,
    VOZ_PYTHON,
    VOZ_TEMPO_MAXIMO,
    VOZ_TEXTO_MAXIMO,
    VOZ_VELOCIDADE,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.voz")

router = APIRouter()

SCRIPT = Path(__file__).resolve().parent / "falar_kokoro.py"
# Vozes do Kokoro: idioma (a, b, e, f, h, i, j, p, z) + genero (f, m) + _nome.
_VOZ_DO_KOKORO = re.compile(r"^[abefhijpz][fm]_[a-z]+$")
_ultimo: dict = {}


class PedidoDeFala(BaseModel):
    input: str
    model: str = ""
    voice: str = ""
    response_format: str = "mp3"
    speed: float | None = None


def conferir_configuracao() -> None:
    if not Path(VOZ_PYTHON).exists():
        raise RuntimeError(
            f"VOZ_ATIVA esta ligado, mas {VOZ_PYTHON} nao existe (o ambiente do Kokoro). "
            "Rode ./deploy/instalar_voz.sh, ou VOZ_ATIVA=nao."
        )


def voz_escolhida(pedida: str) -> str:
    pedida = (pedida or "").strip().lower()
    return pedida if _VOZ_DO_KOKORO.match(pedida) else VOZ_PADRAO


def _invalido(mensagem: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "pedido_invalido", "message": mensagem}}, status_code=422
    )


@router.post("/v1/audio/speech", dependencies=[Depends(conferir)])
def falar(pedido: PedidoDeFala):
    """`def` e nao `async def`: a sintese bloqueia alguns segundos."""
    texto = pedido.input.strip()
    if not texto:
        return _invalido("input vazio.")
    if len(texto) > VOZ_TEXTO_MAXIMO:
        return _invalido(f"input com mais de {VOZ_TEXTO_MAXIMO} caracteres.")
    voz = voz_escolhida(pedido.voice)
    inicio = time.monotonic()
    with tempfile.TemporaryDirectory() as pasta:
        saida = Path(pasta) / "fala"
        try:
            processo = subprocess.run(  # noqa: S603 - comando montado aqui, sem shell
                [
                    VOZ_PYTHON, str(SCRIPT), "--voz", voz, "--saida", str(saida),
                    "--velocidade", str(pedido.speed or VOZ_VELOCIDADE),
                ],
                input=texto,
                capture_output=True,
                text=True,
                timeout=VOZ_TEMPO_MAXIMO,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return respostas.indisponivel(
                "timeout", f"a voz passou de {VOZ_TEMPO_MAXIMO}s.", retry_after=None
            )
        if processo.returncode != 0 or not saida.exists():
            logger.warning("Kokoro falhou (%s): %s", processo.returncode, processo.stderr[-500:])
            return respostas.falha_do_worker(
                "falha_na_voz",
                f"o Kokoro saiu com {processo.returncode}: {processo.stderr.strip()[-400:]}",
            )
        formato = "wav" if "FORMATO wav" in processo.stdout else "mp3"
        conteudo = saida.read_bytes()
    _ultimo.update(voz=voz, segundos=round(time.monotonic() - inicio, 1), caracteres=len(texto))
    return Response(conteudo, media_type="audio/wav" if formato == "wav" else "audio/mpeg")


def estado() -> dict:
    return {
        "motor": "kokoro",
        "voz_dir": VOZ_DIR,
        "voz_padrao": VOZ_PADRAO,
        "ultimo": dict(_ultimo) or None,
    }
