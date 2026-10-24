"""Rota de voz: `POST /v1/audio/speech`, no dialeto da OpenAI.

O PubliBot narra as laminas dos videos das redes por aqui (uma fala por
lamina), sem pagar API. O mesmo cliente fala com a OpenAI (`gpt-4o-mini-tts`)
trocando o endereco.

Dois motores, escolhidos pelo NOME da voz:

- **Kokoro** (`pf_dora`, `pm_alex`; ou `kokoro:pf_dora`): 82M, na CPU, mais
  rapido que tempo real. Nao passa pelo arbitro: narrar nao espera uma imagem
  acabar, e nao tira o modelo de texto da VRAM. Soa sintetico.
- **Qwen3-TTS** (`qwen:<nome>`): bem mais natural, clonado de uma referencia
  de poucos segundos (`VOZ_VOZES_DIR/<nome>/`). Na placa, passa pelo arbitro:
  com uma animacao rodando, leva `503 gpu_ocupada` e o cliente volta depois.

O prefixo `motor:` existe para o nome nunca se confundir com o de outro
servico (o id de 20 letras da ElevenLabs, o `pt-BR-...` do Google). Uma voz que
nao e de nenhum dos dois motores leva `422 voz_desconhecida`, e nao a voz
padrao: quem tem mais de um servico de voz passa para o proximo, em vez de
narrar com a voz errada sem saber. Os nomes da OpenAI (`alloy`, `nova`) e o
vazio continuam virando `VOZ_PADRAO`.

Os dois motores moram em ambientes proprios (`voz/venv` e `voz/qwen/venv`,
instalados pelo `deploy/instalar_voz.sh`) e rodam como processo a parte, como
a animacao.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import recursos
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    VOZ_ACABAMENTO,
    VOZ_ACABAMENTO_FILTROS,
    VOZ_CONVERSAO,
    VOZ_CONVERSAO_DISPOSITIVO,
    VOZ_CONVERSAO_MAXIMO_BYTES,
    VOZ_CONVERSAO_PYTHON,
    VOZ_CONVERSAO_SEGUNDOS_MAXIMOS,
    VOZ_CONVERSAO_TEMPO_MAXIMO,
    VOZ_DIR,
    VOZ_PADRAO,
    VOZ_PYTHON,
    VOZ_QWEN,
    VOZ_QWEN_DISPOSITIVO,
    VOZ_QWEN_IDIOMA,
    VOZ_QWEN_MODELO,
    VOZ_QWEN_MODELO_DESENHO,
    VOZ_QWEN_PYTHON,
    VOZ_QWEN_TEMPO_MAXIMO,
    VOZ_TEMPO_MAXIMO,
    VOZ_TEXTO_MAXIMO,
    VOZ_VELOCIDADE,
    VOZ_VOZES_DIR,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.voz")

router = APIRouter()

SCRIPT = Path(__file__).resolve().parent / "falar_kokoro.py"
SCRIPT_QWEN = Path(__file__).resolve().parent / "falar_qwen.py"
SCRIPT_CONVERSAO = Path(__file__).resolve().parent / "converter_voz.py"
# Vozes do Kokoro: idioma (a, b, e, f, h, i, j, p, z) + genero (f, m) + _nome.
_VOZ_DO_KOKORO = re.compile(r"^[abefhijpz][fm]_[a-z]+$")
# Nome de uma voz do Qwen: a pasta dela.
_NOME_DE_VOZ = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
# As da OpenAI viram a padrao: o PubliBot manda "alloy" quando a conta nao tem voz.
_VOZES_DA_OPENAI = {
    "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer",
    "verse",
}  # fmt: skip
_KOKORO_PT = {
    "pf_dora": "feminina, portugues do Brasil",
    "pm_alex": "masculina, portugues do Brasil",
    "pm_santa": "masculina, portugues do Brasil",
}
_ultimo: dict = {}
_ultima_conversao: dict = {}


class PedidoDeFala(BaseModel):
    input: str
    model: str = ""
    voice: str = ""
    response_format: str = "mp3"
    speed: float | None = None
    # Aceitos e IGNORADOS (o `/health/` diz isso em `voz.campos`). O clone do
    # Qwen (modelo Base) nao tem controle por instrucao: so o VoiceDesign e o
    # CustomVoice tem, e nenhum dos dois mantem uma voz clonada. O tom vem da
    # gravacao de referencia (GRAVAR_VOZ.md). As falas vizinhas pediriam gerar
    # tres falas e recortar uma sem marcas de tempo: o triplo do tempo de placa
    # por um corte incerto.
    instructions: str = ""
    previous_text: str = ""
    next_text: str = ""


# O que de cada pedido cada campo opcional muda. Publicado no `/health/` para o
# cliente nao precisar adivinhar.
CAMPOS_USADOS = {"instructions": False, "previous_text": False, "next_text": False}


class VozDesconhecida(ValueError):
    pass


def conferir_configuracao() -> None:
    if not Path(VOZ_PYTHON).exists():
        raise RuntimeError(
            f"VOZ_ATIVA esta ligado, mas {VOZ_PYTHON} nao existe (o ambiente do Kokoro). "
            "Rode ./deploy/instalar_voz.sh, ou VOZ_ATIVA=nao."
        )
    if VOZ_QWEN and not Path(VOZ_QWEN_PYTHON).exists():
        raise RuntimeError(
            f"VOZ_QWEN esta ligado, mas {VOZ_QWEN_PYTHON} nao existe (o ambiente do Qwen). "
            "Rode ./deploy/instalar_voz.sh, ou VOZ_QWEN=nao."
        )
    if VOZ_CONVERSAO and not Path(VOZ_CONVERSAO_PYTHON).exists():
        raise RuntimeError(
            f"VOZ_CONVERSAO esta ligado, mas {VOZ_CONVERSAO_PYTHON} nao existe (o ambiente da "
            "conversao). Rode ./deploy/instalar_voz.sh, ou VOZ_CONVERSAO=nao."
        )
    for nome, valor in (
        ("VOZ_QWEN_DISPOSITIVO", VOZ_QWEN_DISPOSITIVO),
        ("VOZ_CONVERSAO_DISPOSITIVO", VOZ_CONVERSAO_DISPOSITIVO),
    ):
        if valor not in ("cuda", "cpu"):
            raise RuntimeError(f"{nome}={valor!r}: use cuda ou cpu.")
    try:
        voz_escolhida("")
    except VozDesconhecida as exc:
        # Nao impede subir: a voz pode ser criada depois, e as outras funcionam.
        logger.warning("VOZ_PADRAO=%s nao fala ainda: %s", VOZ_PADRAO, exc)


def _pasta_da_voz(nome: str) -> Path:
    return Path(VOZ_VOZES_DIR) / nome


def vozes_do_qwen() -> list[str]:
    pasta = Path(VOZ_VOZES_DIR)
    if not pasta.is_dir():
        return []
    return sorted(
        p.name
        for p in pasta.iterdir()
        if _NOME_DE_VOZ.match(p.name) and (p / "referencia.wav").exists()
    )


def voz_escolhida(pedida: str) -> tuple[str, str]:
    """(motor, voz) do pedido. Levanta `VozDesconhecida` para o que nao e
    destes motores."""
    pedida = (pedida or "").strip()
    if not pedida or pedida.lower() in _VOZES_DA_OPENAI:
        if VOZ_PADRAO.strip().lower() in _VOZES_DA_OPENAI:
            raise VozDesconhecida(f"VOZ_PADRAO={VOZ_PADRAO!r} nao e voz do worker.")
        pedida = VOZ_PADRAO.strip()
    motor, _, nome = pedida.rpartition(":")
    motor, nome = motor.lower(), nome.lower()
    if motor in ("", "kokoro") and _VOZ_DO_KOKORO.match(nome):
        return "kokoro", nome
    if motor == "qwen":
        if not VOZ_QWEN:
            raise VozDesconhecida(f"{pedida!r}: o Qwen nao esta ligado neste worker (VOZ_QWEN).")
        if not _NOME_DE_VOZ.match(nome) or not (_pasta_da_voz(nome) / "referencia.wav").exists():
            existentes = ", ".join(f"qwen:{v}" for v in vozes_do_qwen()) or "nenhuma"
            raise VozDesconhecida(
                f"{pedida!r} nao existe. Vozes do Qwen aqui: {existentes}. "
                f"Crie no worker: ./deploy/criar_voz.sh {nome} --gravacao <arquivo> "
                '"<o que se diz nele>" (ou uma descricao no lugar da gravacao).'
            )
        return "qwen", nome
    raise VozDesconhecida(
        f"{pedida!r} nao e voz deste worker: use uma do Kokoro (pf_dora, pm_alex) ou qwen:<nome>."
    )


def _erro(codigo: str, mensagem: str, status: int = 422) -> JSONResponse:
    return JSONResponse({"error": {"code": codigo, "message": mensagem}}, status_code=status)


def _invalido(mensagem: str) -> JSONResponse:
    return _erro("pedido_invalido", mensagem)


def _rodar(comando: list[str], *, entrada: str = "", tempo: int) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603 - comando montado aqui, sem shell
        comando, input=entrada, capture_output=True, text=True, timeout=tempo
    )


def _na_placa(
    rotulo: str,
    modelo: str,
    comando: list[str],
    *,
    entrada: str = "",
    tempo: int,
    tarefa: str = "voz",
    dispositivo: str | None = None,
):
    """Roda o processo com a placa na mao (ou sem ela, na CPU). Devolve o
    processo, ou a resposta de erro pronta (503 ocupada)."""
    if (dispositivo or VOZ_QWEN_DISPOSITIVO) == "cpu":
        return _rodar(comando, entrada=entrada, tempo=tempo)
    try:
        with ARBITRO.usar(tarefa, modelo=modelo):
            recursos.soltar_a_placa(rotulo)
            return _rodar(comando, entrada=entrada, tempo=tempo)
    except GpuOcupada as erro:
        return respostas.ocupada(erro)


def _comando_qwen(modo: str, nome: str, modelo: str) -> list[str]:
    return [
        VOZ_QWEN_PYTHON, str(SCRIPT_QWEN), modo,
        "--voz-dir", str(_pasta_da_voz(nome)),
        "--modelo", modelo,
        "--dispositivo", VOZ_QWEN_DISPOSITIVO,
        "--idioma", VOZ_QWEN_IDIOMA,
    ]  # fmt: skip


@router.post("/v1/audio/speech", dependencies=[Depends(conferir)])
def falar(pedido: PedidoDeFala):
    """`def` e nao `async def`: a sintese bloqueia de segundos a um minuto."""
    texto = pedido.input.strip()
    if not texto:
        return _invalido("input vazio.")
    if len(texto) > VOZ_TEXTO_MAXIMO:
        return _invalido(f"input com mais de {VOZ_TEXTO_MAXIMO} caracteres.")
    try:
        motor, voz = voz_escolhida(pedido.voice)
    except VozDesconhecida as exc:
        return _erro("voz_desconhecida", str(exc))
    inicio = time.monotonic()
    with tempfile.TemporaryDirectory() as pasta:
        saida = Path(pasta) / "fala"
        if motor == "kokoro":
            tempo = VOZ_TEMPO_MAXIMO
            comando = [
                VOZ_PYTHON, str(SCRIPT), "--voz", voz, "--saida", str(saida),
                "--velocidade", str(pedido.speed or VOZ_VELOCIDADE),
            ]  # fmt: skip
        else:
            tempo = VOZ_QWEN_TEMPO_MAXIMO
            comando = [*_comando_qwen("falar", voz, VOZ_QWEN_MODELO), "--saida", str(saida)]
        try:
            if motor == "kokoro":
                processo = _rodar(comando, entrada=texto, tempo=tempo)
            else:
                processo = _na_placa("falar", VOZ_QWEN_MODELO, comando, entrada=texto, tempo=tempo)
        except subprocess.TimeoutExpired:
            return respostas.indisponivel("timeout", f"a voz passou de {tempo}s.", retry_after=None)
        if isinstance(processo, JSONResponse):
            return processo
        if processo.returncode != 0 or not saida.exists():
            logger.warning("%s falhou (%s): %s", motor, processo.returncode, processo.stderr[-500:])
            return respostas.falha_do_worker(
                "falha_na_voz",
                f"o {motor} saiu com {processo.returncode}: {processo.stderr.strip()[-400:]}",
            )
        formato = "wav" if "FORMATO wav" in processo.stdout else "mp3"
        conteudo, formato = acabamento(saida.read_bytes(), formato)
    _ultimo.update(
        motor=motor,
        voz=voz if motor == "kokoro" else f"qwen:{voz}",
        segundos=round(time.monotonic() - inicio, 1),
        caracteres=len(texto),
    )
    return Response(conteudo, media_type="audio/wav" if formato == "wav" else "audio/mpeg")


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


_avisou_sem_ffmpeg = False


def acabamento(conteudo: bytes, formato: str) -> tuple[bytes, str]:
    """A fala com o acabamento de locucao (`VOZ_ACABAMENTO_FILTROS`), em MP3.

    Nunca derruba a fala: sem ffmpeg, ou se ele falhar, devolve o audio como
    veio. O acabamento e melhoria, nao condicao."""
    global _avisou_sem_ffmpeg
    if not VOZ_ACABAMENTO:
        return conteudo, formato
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        if not _avisou_sem_ffmpeg:
            logger.warning("VOZ_ACABAMENTO ligado, mas sem ffmpeg: a voz sai sem acabamento.")
            _avisou_sem_ffmpeg = True
        return conteudo, formato
    try:
        processo = subprocess.run(  # noqa: S603 - comando montado aqui, sem shell
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                "-af", VOZ_ACABAMENTO_FILTROS,
                # O loudnorm trabalha em 192 kHz; volta para uma taxa de fala.
                "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k",
                "-f", "mp3", "pipe:1",
            ],
            input=conteudo,
            capture_output=True,
            timeout=60,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        logger.warning("Acabamento passou de 60s; a voz sai sem ele.")
        return conteudo, formato
    if processo.returncode != 0 or not processo.stdout:
        logger.warning(
            "Acabamento falhou (%s): %s",
            processo.returncode,
            processo.stderr[-300:].decode(errors="replace"),
        )
        return conteudo, formato
    return processo.stdout, "mp3"


def _descricao_da_voz(nome: str) -> dict:
    pasta = _pasta_da_voz(nome)
    descricao = pasta / "descricao.txt"
    return {
        "id": f"qwen:{nome}",
        "motor": "qwen",
        "descricao": descricao.read_text(encoding="utf-8").strip() if descricao.exists() else None,
        # Sem a transcricao, o clone usa so a impressao da voz e imita menos.
        "transcrita": (pasta / "referencia.txt").exists(),
    }


@router.get("/v1/audio/voices", dependencies=[Depends(conferir)])
def listar_vozes():
    vozes = [{"id": v, "motor": "kokoro", "descricao": d} for v, d in _KOKORO_PT.items()]
    if VOZ_QWEN:
        vozes += [_descricao_da_voz(v) for v in vozes_do_qwen()]
    return {"object": "list", "padrao": VOZ_PADRAO, "data": vozes}


@router.post("/v1/audio/voices", dependencies=[Depends(conferir)])
def criar_voz(
    nome: str = Form(...),
    descricao: str = Form(""),
    texto: str = Form(""),
    substituir: bool = Form(False),
    gravacao: UploadFile | None = None,
):
    """Cria uma voz do Qwen: de uma DESCRICAO (o modelo VoiceDesign inventa a
    voz e grava a referencia) ou de uma GRAVACAO (de 5 a 15 s de fala, com o
    `texto` do que se diz nela, de preferencia)."""
    if not VOZ_QWEN:
        return _erro("voz_desligada", "o Qwen nao esta ligado neste worker (VOZ_QWEN).", 404)
    nome = nome.strip().lower()
    if not _NOME_DE_VOZ.match(nome):
        return _invalido("nome: letras minusculas, numeros, - e _, ate 40.")
    if bool(descricao.strip()) == (gravacao is not None):
        return _invalido("mande descricao OU gravacao.")
    if (_pasta_da_voz(nome) / "referencia.wav").exists() and not substituir:
        return _erro("voz_existente", f"qwen:{nome} ja existe (substituir=true troca).", 409)
    try:
        if gravacao is not None:
            with tempfile.TemporaryDirectory() as pasta:
                arquivo = Path(pasta) / "gravacao"
                arquivo.write_bytes(gravacao.file.read())
                comando = [
                    *_comando_qwen("importar", nome, VOZ_QWEN_MODELO),
                    "--gravacao", str(arquivo), "--texto", texto,
                ]  # fmt: skip
                processo = _rodar(comando, tempo=VOZ_QWEN_TEMPO_MAXIMO)
        else:
            comando = [
                *_comando_qwen("desenhar", nome, VOZ_QWEN_MODELO_DESENHO),
                "--descricao", descricao.strip(),
            ]  # fmt: skip
            processo = _na_placa(
                "criar a voz", VOZ_QWEN_MODELO_DESENHO, comando, tempo=VOZ_QWEN_TEMPO_MAXIMO
            )
    except subprocess.TimeoutExpired:
        return respostas.indisponivel(
            "timeout", f"criar a voz passou de {VOZ_QWEN_TEMPO_MAXIMO}s.", retry_after=None
        )
    if isinstance(processo, JSONResponse):
        return processo
    if processo.returncode == 2:
        return _invalido(processo.stderr.strip()[-400:])
    if processo.returncode != 0:
        logger.warning("criar voz falhou (%s): %s", processo.returncode, processo.stderr[-500:])
        return respostas.falha_do_worker(
            "falha_na_voz",
            f"o qwen saiu com {processo.returncode}: {processo.stderr.strip()[-400:]}",
        )
    return JSONResponse(_descricao_da_voz(nome), status_code=201)


def referencia_para_converter(pedida: str) -> Path:
    """A gravacao de referencia da voz alvo. Levanta `VozDesconhecida`.

    So as vozes `qwen:<nome>` tem uma; nao precisa do Qwen ligado, so da pasta."""
    pedida = (pedida or "").strip() or VOZ_PADRAO.strip()
    motor, _, nome = pedida.rpartition(":")
    motor, nome = motor.lower(), nome.lower()
    if motor in ("", "kokoro") and _VOZ_DO_KOKORO.match(nome):
        raise VozDesconhecida(
            f"{pedida!r} e do Kokoro, que nao tem gravacao de referencia: converta para "
            "uma qwen:<nome>."
        )
    if motor != "qwen":
        raise VozDesconhecida(f"{pedida!r} nao e voz deste worker: use qwen:<nome>.")
    referencia = _pasta_da_voz(nome) / "referencia.wav"
    if not _NOME_DE_VOZ.match(nome) or not referencia.exists():
        existentes = ", ".join(f"qwen:{v}" for v in vozes_do_qwen()) or "nenhuma"
        raise VozDesconhecida(f"{pedida!r} nao existe. Vozes do Qwen aqui: {existentes}.")
    return referencia


@router.post("/v1/audio/convert", dependencies=[Depends(conferir)])
def converter(audio: UploadFile, voice: str = Form("")):
    """A gravacao no timbre de uma voz do worker, com a entonacao de quem gravou."""
    if not VOZ_CONVERSAO:
        return _erro(
            "conversao_desligada", "a conversao de voz nao esta ligada (VOZ_CONVERSAO).", 404
        )
    try:
        alvo = referencia_para_converter(voice)
    except VozDesconhecida as exc:
        return _erro("voz_desconhecida", str(exc))
    conteudo = audio.file.read(VOZ_CONVERSAO_MAXIMO_BYTES + 1)
    if not conteudo:
        return _invalido("audio vazio.")
    if len(conteudo) > VOZ_CONVERSAO_MAXIMO_BYTES:
        return _erro(
            "arquivo_grande", f"audio com mais de {VOZ_CONVERSAO_MAXIMO_BYTES} bytes.", 413
        )
    inicio = time.monotonic()
    with tempfile.TemporaryDirectory() as pasta:
        # A extensao ajuda o leitor de audio a reconhecer o formato.
        sufixo = Path(audio.filename or "").suffix.lower()[:5] or ".mp3"
        origem = Path(pasta) / f"origem{sufixo}"
        origem.write_bytes(conteudo)
        saida = Path(pasta) / "convertida"
        comando = [
            VOZ_CONVERSAO_PYTHON, str(SCRIPT_CONVERSAO),
            "--origem", str(origem), "--alvo", str(alvo), "--saida", str(saida),
            "--dispositivo", VOZ_CONVERSAO_DISPOSITIVO,
            "--segundos-maximos", str(VOZ_CONVERSAO_SEGUNDOS_MAXIMOS),
        ]  # fmt: skip
        try:
            processo = _na_placa(
                "converter a voz", "chatterbox-vc", comando,
                tempo=VOZ_CONVERSAO_TEMPO_MAXIMO, tarefa="conversao_de_voz",
                dispositivo=VOZ_CONVERSAO_DISPOSITIVO,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return respostas.indisponivel(
                "timeout", f"a conversao passou de {VOZ_CONVERSAO_TEMPO_MAXIMO}s.", retry_after=None
            )
        if isinstance(processo, JSONResponse):
            return processo
        if processo.returncode == 2:
            return _invalido(processo.stderr.strip()[-400:])
        if processo.returncode != 0 or not saida.exists():
            logger.warning("conversao falhou (%s): %s", processo.returncode, processo.stderr[-500:])
            return respostas.falha_do_worker(
                "falha_na_voz",
                f"a conversao saiu com {processo.returncode}: {processo.stderr.strip()[-400:]}",
            )
        formato = "wav" if "FORMATO wav" in processo.stdout else "mp3"
        convertida, formato = acabamento(saida.read_bytes(), formato)
    _ultima_conversao.update(
        voz=f"qwen:{alvo.parent.name}",
        segundos=round(time.monotonic() - inicio, 1),
        bytes=len(conteudo),
    )
    return Response(convertida, media_type="audio/wav" if formato == "wav" else "audio/mpeg")


def estado() -> dict:
    corpo = {
        # Da 2.11, quando so havia o Kokoro; `motores` diz o que esta ligado.
        "motor": "kokoro",
        "motores": {"kokoro": True, "qwen": VOZ_QWEN},
        "voz_dir": VOZ_DIR,
        "voz_padrao": VOZ_PADRAO,
        "campos": dict(CAMPOS_USADOS),
        "acabamento": {"ligado": VOZ_ACABAMENTO, "ffmpeg": _ffmpeg() is not None},
        "ultimo": dict(_ultimo) or None,
    }
    if VOZ_CONVERSAO:
        corpo["conversao"] = {
            "modelo": "chatterbox-vc",
            "dispositivo": VOZ_CONVERSAO_DISPOSITIVO,
            "segundos_maximos": VOZ_CONVERSAO_SEGUNDOS_MAXIMOS,
            "ultima": dict(_ultima_conversao) or None,
        }
    if VOZ_QWEN:
        corpo["qwen"] = {
            "modelo": VOZ_QWEN_MODELO,
            "dispositivo": VOZ_QWEN_DISPOSITIVO,
            "idioma": VOZ_QWEN_IDIOMA,
            "vozes": [f"qwen:{v}" for v in vozes_do_qwen()],
        }
    return corpo
