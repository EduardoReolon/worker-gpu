"""Rota de transcricao: `POST /v1/audio/transcriptions`, arbitrada.

Recebe um arquivo de audio (ou video: o que importa e a trilha de som) e
devolve o texto com os segmentos cronometrados, no dialeto da OpenAI — como as
rotas de texto e imagem. Quem usa e o PubliBot, para video do YouTube sem
legenda: os `segments` viram secoes por janela de tempo, e sao elas que deixam
citar o minuto certo.

O motor e o `faster-whisper` (CTranslate2). Ele disputa a VRAM com o modelo de
texto como a difusao disputa, entao entra no MESMO lock e descarrega o Ollama
antes de carregar.

Os erros seguem o contrato das outras rotas: 503 com `error.code` para o que e
transitorio, 500 com `error.code` para o worker quebrado, e 422 para o arquivo
que nao e audio.
"""

from __future__ import annotations

import ctypes
import gc
import importlib.util
import logging
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import ollama
import prazo
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    MAX_AUDIO_BYTES,
    OLLAMA_DESCARREGAR_PARA_TRANSCRICAO,
    TRANSCRICAO_BEAM,
    TRANSCRICAO_COMPUTACAO,
    TRANSCRICAO_DEVICE,
    TRANSCRICAO_MODELO,
    TRANSCRICAO_OCIOSO_SEGUNDOS,
    TRANSCRICAO_TEMPO_MAXIMO,
    TRANSCRICAO_TEMPO_TRAVADO,
    TRANSCRICAO_VAD,
)
from prazo import WorkerTravado
from seguranca import conferir

logger = logging.getLogger("worker-gpu.transcricao")

router = APIRouter()

FORMATOS_DE_RESPOSTA = ("json", "verbose_json", "text")

_modelo = None
_dispositivo_do_modelo = ""
_ultimo_dispositivo = ""
_trava = threading.Lock()
_temporizador: threading.Timer | None = None


class TempoEsgotado(RuntimeError):
    """A transcricao passou de `TRANSCRICAO_TEMPO_MAXIMO`."""


class AudioInvalido(ValueError):
    """O arquivo nao e audio que o ffmpeg leia, ou o pedido nao faz sentido."""


class ArquivoGrande(ValueError):
    """O upload passou de `MAX_AUDIO_BYTES`."""


def conferir_configuracao() -> None:
    """Recusa subir mal configurado, em vez de falhar no primeiro audio.

    O modelo carrega preguicosamente: sem esta conferencia, um `faster-whisper`
    ausente so apareceria como 500 na primeira transcricao — dias depois,
    quando chegar o primeiro video sem legenda.
    """
    if TRANSCRICAO_DEVICE not in ("auto", "cpu", "cuda"):
        raise RuntimeError(f"TRANSCRICAO_DEVICE={TRANSCRICAO_DEVICE!r}. Use auto, cpu ou cuda.")
    if 0 < TRANSCRICAO_TEMPO_TRAVADO <= TRANSCRICAO_TEMPO_MAXIMO:
        raise RuntimeError(
            f"TRANSCRICAO_TEMPO_TRAVADO={TRANSCRICAO_TEMPO_TRAVADO} precisa ser maior que "
            f"TRANSCRICAO_TEMPO_MAXIMO={TRANSCRICAO_TEMPO_MAXIMO}: um audio longo mas "
            f"saudavel tem que levar 503 `timeout`, e nao derrubar o processo."
        )
    if importlib.util.find_spec("faster_whisper") is None:
        raise RuntimeError(
            "TRANSCRICAO_ATIVA esta ligado, mas o faster-whisper nao esta instalado.\n"
            "  ./venv/bin/pip install -r requirements.txt\n"
            "ou TRANSCRICAO_ATIVA=nao no .env."
        )


# ---------------------------------------------------------------------------
# Dispositivo e modelo
# ---------------------------------------------------------------------------
def resolver_dispositivo() -> str:
    """cpu ou cuda, ja resolvido — nunca "auto" para dentro do codigo."""
    if TRANSCRICAO_DEVICE == "cpu":
        return "cpu"

    import ctranslate2

    tem_placa = ctranslate2.get_cuda_device_count() > 0
    if TRANSCRICAO_DEVICE == "cuda" and not tem_placa:
        logger.warning(
            "TRANSCRICAO_DEVICE=cuda, mas o CTranslate2 nao ve nenhuma GPU. Vou "
            "transcrever em CPU, o que leva da ordem da duracao do audio."
        )
        return "cpu"

    return "cuda" if tem_placa else "cpu"


def _preparar_bibliotecas_cuda() -> None:
    """Deixa o cuBLAS e o cuDNN do CUDA 12 carregados antes do CTranslate2.

    O CTranslate2 abre `libcublas.so.12` e `libcudnn*.so.9` pelo NOME, e so os
    acha no caminho de bibliotecas do sistema. Numa maquina sem o CUDA
    instalado a parte, eles existem apenas dentro do venv, nos pacotes
    `nvidia-*` que o torch traz — e a primeira transcricao falharia com
    "Library libcublas.so.12 is not found".

    Carregar com RTLD_GLOBAL resolve: um `dlopen` pelo nome encontra a
    biblioteca ja carregada no processo. Nao achar nada nao e erro — o sistema
    pode ter o CUDA instalado, e se nao tiver, a mensagem de `_carregar` diz
    o que fazer.
    """
    for pacote in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            especificacao = importlib.util.find_spec(pacote)
        except (ImportError, ValueError):
            continue
        if especificacao is None or not especificacao.submodule_search_locations:
            continue

        for raiz in especificacao.submodule_search_locations:
            pasta = Path(raiz) / "lib"
            if not pasta.is_dir():
                continue
            # A ordem importa pouco com RTLD_GLOBAL, mas o `Lt` antes do
            # cuBLAS e o nucleo do cuDNN antes dos modulos evita um aviso a
            # mais no log de quem estiver olhando.
            for biblioteca in sorted(pasta.glob("lib*.so*")):
                try:
                    ctypes.CDLL(str(biblioteca), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _carregar(dispositivo: str):
    from faster_whisper import WhisperModel

    computacao = TRANSCRICAO_COMPUTACAO if dispositivo == "cuda" else "int8"
    if dispositivo == "cuda":
        _preparar_bibliotecas_cuda()

    logger.info(
        "Carregando o Whisper %s em %s (%s)...", TRANSCRICAO_MODELO, dispositivo, computacao
    )
    inicio = time.perf_counter()
    try:
        modelo = WhisperModel(TRANSCRICAO_MODELO, device=dispositivo, compute_type=computacao)
    except Exception as exc:
        texto = str(exc)
        if "libcublas" in texto or "libcudnn" in texto:
            raise RuntimeError(
                f"{texto}\nO CTranslate2 precisa do cuBLAS do CUDA 12 e do cuDNN 9. "
                f"Instale no venv do worker:\n"
                f'  ./venv/bin/pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"\n'
                f"ou use TRANSCRICAO_DEVICE=cpu."
            ) from exc
        raise

    logger.info("Whisper pronto em %.1fs.", time.perf_counter() - inicio)
    return modelo


def obter_modelo(dispositivo: str):
    global _modelo, _dispositivo_do_modelo

    with _trava:
        if _modelo is not None and _dispositivo_do_modelo != dispositivo:
            _descarregar_sem_trava()
        if _modelo is None:
            _modelo = _carregar(dispositivo)
            _dispositivo_do_modelo = dispositivo
        return _modelo


def _descarregar_sem_trava() -> None:
    global _modelo, _dispositivo_do_modelo

    if _modelo is None:
        return

    logger.info("Descarregando o Whisper e devolvendo a memoria dele.")
    _modelo = None
    _dispositivo_do_modelo = ""
    # O CTranslate2 solta a VRAM quando o modelo e coletado.
    gc.collect()


def descarregar() -> None:
    with _trava:
        _descarregar_sem_trava()


def _agendar_descarga() -> None:
    global _temporizador

    if TRANSCRICAO_OCIOSO_SEGUNDOS <= 0:
        return
    if _temporizador is not None:
        _temporizador.cancel()

    _temporizador = threading.Timer(TRANSCRICAO_OCIOSO_SEGUNDOS, descarregar)
    # Daemon: um temporizador pendente nao pode segurar o desligamento.
    _temporizador.daemon = True
    _temporizador.start()


# ---------------------------------------------------------------------------
# Transcricao
# ---------------------------------------------------------------------------
def transcrever(dispositivo: str, caminho: str, idioma: str | None, contexto: str | None) -> dict:
    """O resultado no formato `verbose_json` da OpenAI.

    `transcribe` devolve um GERADOR: o trabalho acontece enquanto se itera, e
    e entre um segmento e outro que o teto macio pode ser conferido.
    """
    modelo = obter_modelo(dispositivo)

    try:
        segmentos, info = modelo.transcribe(
            caminho,
            language=idioma,
            initial_prompt=contexto,
            beam_size=TRANSCRICAO_BEAM,
            vad_filter=TRANSCRICAO_VAD,
        )
    except Exception as exc:
        # O PyAV (o ffmpeg por dentro do faster-whisper) recusa o que nao e
        # audio com uma excecao do modulo `av`; um idioma inexistente vira
        # ValueError do tokenizador. Os dois sao o PEDIDO, nao o worker.
        if type(exc).__module__.startswith("av") or isinstance(exc, ValueError):
            raise AudioInvalido(str(exc)) from exc
        raise

    limite = (
        time.perf_counter() + TRANSCRICAO_TEMPO_MAXIMO if TRANSCRICAO_TEMPO_MAXIMO > 0 else None
    )
    lista = []
    for segmento in segmentos:
        lista.append(
            {
                "id": len(lista),
                "start": round(segmento.start, 2),
                "end": round(segmento.end, 2),
                "text": segmento.text.strip(),
            }
        )
        if limite is not None and time.perf_counter() > limite:
            raise TempoEsgotado(
                f"a transcricao passou de {TRANSCRICAO_TEMPO_MAXIMO}s no minuto "
                f"{segmento.end / 60:.0f} de {info.duration / 60:.0f}. Divida o audio, use "
                f"um modelo menor (TRANSCRICAO_MODELO) ou aumente TRANSCRICAO_TEMPO_MAXIMO."
            )

    return {
        "task": "transcribe",
        "language": info.language,
        "duration": round(info.duration, 2),
        "text": " ".join(item["text"] for item in lista if item["text"]),
        "segments": lista,
    }


def _e_falta_de_vram(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _invalido(mensagem: str, *, codigo: str = "arquivo_invalido", status: int = 422):
    """4xx com `error.message`, que e o que o cliente le, e `detail`, que e o
    que o FastAPI poe nos outros 4xx deste servico."""
    return JSONResponse(
        {"detail": mensagem, "error": {"code": codigo, "message": mensagem}},
        status_code=status,
    )


def _guardar(arquivo: UploadFile, destino: Path) -> int:
    """Copia o upload para o disco em blocos, recusando o que passa do teto.

    Em blocos, e nao `read()` inteiro: um audio de uma hora em WAV tem 600 MB,
    e ler tudo para a memoria so para gravar de novo e dobrar o pico por nada.
    """
    total = 0
    with destino.open("wb") as saida:
        while bloco := arquivo.file.read(1024 * 1024):
            total += len(bloco)
            if total > MAX_AUDIO_BYTES:
                raise ArquivoGrande(f"arquivo excede {MAX_AUDIO_BYTES} bytes (MAX_AUDIO_BYTES).")
            saida.write(bloco)
    return total


@router.post("/v1/audio/transcriptions", dependencies=[Depends(conferir)])
def transcrever_audio(
    file: UploadFile,
    model: str | None = Form(default=None),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
):
    """Transcreve um audio.

    `model` e aceito e ignorado: o modelo e o do `.env` (`TRANSCRICAO_MODELO`),
    como o de imagem. `language` vazio faz o Whisper detectar. `prompt` vira o
    `initial_prompt` — util para nomes proprios e siglas ("BDI", "SINAPI").

    `def` e nao `async def`: a transcricao bloqueia por minutos, e no event
    loop ela congelaria o processo inteiro.
    """
    global _ultimo_dispositivo

    if response_format not in FORMATOS_DE_RESPOSTA:
        return _invalido(
            f"response_format={response_format!r}. Use um de: {', '.join(FORMATOS_DE_RESPOSTA)}."
        )
    idioma = (language or "").strip().lower() or None

    sufixo = Path(file.filename or "").suffix.lower()[:10]
    with tempfile.TemporaryDirectory() as pasta:
        caminho = Path(pasta) / f"audio{sufixo}"
        try:
            tamanho = _guardar(file, caminho)
        except ArquivoGrande as exc:
            return _invalido(str(exc), codigo="arquivo_grande", status=413)
        if tamanho == 0:
            return _invalido("arquivo vazio.")

        inicio = time.perf_counter()
        try:
            with ARBITRO.usar("transcricao", modelo=TRANSCRICAO_MODELO):
                if OLLAMA_DESCARREGAR_PARA_TRANSCRICAO:
                    ollama.descarregar_tudo()

                dispositivo = resolver_dispositivo()
                try:
                    resultado = prazo.com_prazo_duro(
                        lambda: transcrever(dispositivo, str(caminho), idioma, prompt),
                        TRANSCRICAO_TEMPO_TRAVADO,
                        "transcricao",
                        "TRANSCRICAO_TEMPO_TRAVADO",
                    )
                except WorkerTravado as exc:
                    logger.critical("%s Encerrando o processo para o systemd subir outro.", exc)
                    prazo.morrer_em_seguida()
                    return respostas.falha_do_worker("worker_travado", str(exc))
                except TempoEsgotado as exc:
                    logger.warning("%s", exc)
                    return respostas.indisponivel("timeout", str(exc), retry_after=None)
                except AudioInvalido as exc:
                    logger.warning("Audio recusado (%s): %s", file.filename, exc)
                    return _invalido(f"o arquivo nao e audio legivel: {exc}")
                except Exception as exc:
                    if not _e_falta_de_vram(exc):
                        logger.exception("Falha ao transcrever %s", file.filename)
                        return respostas.falha_do_worker(
                            "falha_na_transcricao", f"{type(exc).__name__}: {exc}"
                        )
                    logger.warning("VRAM insuficiente para o Whisper (%s).", exc)
                    descarregar()
                    return respostas.indisponivel(
                        "sem_vram",
                        "sem VRAM para transcrever agora. O trabalho volta para a fila. "
                        "Se persistir, use TRANSCRICAO_MODELO=medium ou "
                        "TRANSCRICAO_COMPUTACAO=int8.",
                    )

                _ultimo_dispositivo = dispositivo
        except GpuOcupada as erro:
            return respostas.ocupada(erro)
        finally:
            _agendar_descarga()

    logger.info(
        "Transcrito %s: %.0fs de audio em %.0fs (%s, %s, %s segmentos).",
        file.filename,
        resultado["duration"],
        time.perf_counter() - inicio,
        resultado["language"],
        dispositivo,
        len(resultado["segments"]),
    )

    if response_format == "text":
        return PlainTextResponse(resultado["text"])
    if response_format == "json":
        return {"text": resultado["text"]}
    return resultado


def estado() -> dict:
    """O que o `/health/` do worker mostra sobre esta rota."""
    return {
        "modelo": TRANSCRICAO_MODELO,
        "dispositivo": TRANSCRICAO_DEVICE,
        "ultimo_dispositivo": _ultimo_dispositivo or "(nada transcrito ainda)",
        "computacao": TRANSCRICAO_COMPUTACAO,
        "vad": TRANSCRICAO_VAD,
        "carregado": _modelo is not None,
        "ocioso_segundos": TRANSCRICAO_OCIOSO_SEGUNDOS,
        "tempo_maximo": TRANSCRICAO_TEMPO_MAXIMO,
        "tempo_travado": TRANSCRICAO_TEMPO_TRAVADO,
        "max_bytes": MAX_AUDIO_BYTES,
    }
