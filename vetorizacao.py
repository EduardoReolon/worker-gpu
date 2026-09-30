"""Rota de vetorizacao: `POST /v1/embeddings`, arbitrada.

Quem usa e o PubliBot, para INDEXAR documentos: um vetor por paragrafo. O
servidor dele e um ARM de 1 CPU, onde isso leva minutos; na placa, segundos. A
CONSULTA continua no servidor — buscar nao pode depender desta maquina estar
ligada —, e por isso o vetor daqui precisa sair IGUAL ao de la: e o mesmo
indice.

## Por que torch, e nao o fastembed do servidor

O servidor usa `fastembed==0.8.0` (ONNX). A versao de placa dele, o
`fastembed-gpu`, traz o `onnxruntime-gpu` — que briga com o `onnxruntime` que o
faster-whisper usa (os dois instalam o mesmo modulo) e depende do cuDNN do
CUDA 12, que grava os mesmos arquivos que o cuDNN do torch.

O calculo do fastembed para ESTE modelo e simples e esta no codigo dele: mean
pooling sobre a ultima camada, SEM normalizar, truncando em 512 tokens. Aqui e
o mesmo calculo, sobre os mesmos pesos, em float32. O teste de conformidade
(`conferir_vetorizacao.py`) e o que prova que deu certo.

O worker nao acrescenta nada ao texto: o prefixo `passage: ` que o e5 exige ja
vem do cliente.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import ollama
import prazo
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    FALHA_DE_DOWNLOAD_LEMBRADA,
    OLLAMA_DESCARREGAR_PARA_VETORIZACAO,
    VETORIZACAO_DEVICE,
    VETORIZACAO_DTYPE,
    VETORIZACAO_LOTE,
    VETORIZACAO_MAXIMO_TEXTOS,
    VETORIZACAO_MODELO,
    VETORIZACAO_OCIOSO_SEGUNDOS,
    VETORIZACAO_TEMPO_TRAVADO,
)
from prazo import WorkerTravado
from seguranca import conferir

logger = logging.getLogger("worker-gpu.vetorizacao")

router = APIRouter()

# O que baixar do repositorio. O do e5 publica os pesos em TRES formatos
# (safetensors, pytorch_model.bin e ONNX): baixar tudo seriam ~7 GB para usar 2.
ARQUIVOS_DO_MODELO = ["*.json", "*.safetensors", "sentencepiece.bpe.model"]

# O e5 foi treinado com 512 tokens, e e onde o fastembed corta. Cortar em
# outro ponto daria vetor diferente para paragrafo longo.
MAXIMO_DE_TOKENS = 512

_modelo = None
_tokenizador = None
_dispositivo_do_modelo = ""
_ultimo_dispositivo = ""
_baixado = False
_trava = threading.Lock()
_temporizador: threading.Timer | None = None

# O download em segundo plano. Uma thread so, por processo: varios documentos
# na fila recebem o mesmo 503, e ninguem baixa o modelo duas vezes.
_download_trava = threading.Lock()
_download_thread: threading.Thread | None = None
_download_erro = ""
_download_falhou_em = 0.0


class PedidoDeVetores(BaseModel):
    model: str = ""
    input: list[str] | str
    # Aceitos e ignorados, do dialeto da OpenAI.
    encoding_format: str = "float"
    user: str | None = None


def conferir_configuracao() -> None:
    if VETORIZACAO_DEVICE not in ("auto", "cpu", "cuda"):
        raise RuntimeError(f"VETORIZACAO_DEVICE={VETORIZACAO_DEVICE!r}. Use auto, cpu ou cuda.")
    if VETORIZACAO_DTYPE not in ("float32", "float16", "bfloat16"):
        raise RuntimeError(
            f"VETORIZACAO_DTYPE={VETORIZACAO_DTYPE!r}. Use float32, float16 ou bfloat16."
        )
    if OLLAMA_DESCARREGAR_PARA_VETORIZACAO not in ("sim", "nao", "se_faltar"):
        raise RuntimeError(
            f"OLLAMA_DESCARREGAR_PARA_VETORIZACAO={OLLAMA_DESCARREGAR_PARA_VETORIZACAO!r}. "
            f"Use sim, nao ou se_faltar."
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def modelo_esta_no_disco() -> bool:
    """Se os pesos e o tokenizador ja estao no cache local.

    Uma vez verdadeiro, sempre verdadeiro: pesos nao se desbaixam, e olhar o
    cache a cada pedido poria trabalho de disco num caminho rapido.
    """
    global _baixado

    if _baixado:
        return True
    if Path(VETORIZACAO_MODELO).is_dir():
        _baixado = True
        return True

    try:
        from huggingface_hub import try_to_load_from_cache

        pesos = try_to_load_from_cache(VETORIZACAO_MODELO, "model.safetensors")
        # Um dos dois basta: o tokenizador "rapido" ou o sentencepiece de onde
        # ele e montado. Exigir um so, e o repositorio nao ter aquele, daria
        # 503 `modelo_carregando` para sempre.
        tokenizadores = [
            try_to_load_from_cache(VETORIZACAO_MODELO, nome)
            for nome in ("tokenizer.json", "sentencepiece.bpe.model")
        ]
    except Exception:
        return False

    _baixado = isinstance(pesos, str) and any(isinstance(t, str) for t in tokenizadores)
    return _baixado


def baixar() -> str:
    """Baixa o modelo para o cache e devolve a pasta. Bloqueia."""
    from huggingface_hub import snapshot_download

    return snapshot_download(VETORIZACAO_MODELO, allow_patterns=ARQUIVOS_DO_MODELO)


def _baixar_em_segundo_plano() -> None:
    global _download_erro, _download_falhou_em, _baixado

    logger.info("Baixando %s em segundo plano (~2 GB)...", VETORIZACAO_MODELO)
    inicio = time.perf_counter()
    try:
        baixar()
    except Exception as exc:
        logger.exception("Falha ao baixar %s", VETORIZACAO_MODELO)
        with _download_trava:
            _download_erro = f"{type(exc).__name__}: {exc}"
            _download_falhou_em = time.monotonic()
        return

    with _download_trava:
        _download_erro = ""
    _baixado = True
    logger.info("%s baixado em %.0fs.", VETORIZACAO_MODELO, time.perf_counter() - inicio)


def iniciar_download() -> tuple[str, int]:
    """Dispara o download (uma vez) e devolve a mensagem e o `Retry-After`.

    Uma falha recente e lembrada por `FALHA_DE_DOWNLOAD_LEMBRADA`: sem memoria,
    cada retentativa do cliente dispararia um download novo de algo que nao
    vai vir; com memoria eterna, uma queda de rede exigiria reiniciar o
    servico.
    """
    global _download_thread

    with _download_trava:
        if _download_thread is not None and _download_thread.is_alive():
            return f"o modelo {VETORIZACAO_MODELO} esta sendo baixado (~2 GB).", 60

        if _download_erro:
            passou = time.monotonic() - _download_falhou_em
            if passou < FALHA_DE_DOWNLOAD_LEMBRADA:
                falta = int(FALHA_DE_DOWNLOAD_LEMBRADA - passou) + 1
                return (
                    f"o download de {VETORIZACAO_MODELO} falhou ({_download_erro}). "
                    f"Nova tentativa em {falta}s.",
                    falta,
                )

        _download_thread = threading.Thread(
            target=_baixar_em_segundo_plano, name="baixar-vetorizacao", daemon=True
        )
        _download_thread.start()
        return f"o modelo {VETORIZACAO_MODELO} comecou a ser baixado (~2 GB).", 60


def _baixando() -> bool:
    with _download_trava:
        return _download_thread is not None and _download_thread.is_alive()


# ---------------------------------------------------------------------------
# Dispositivo e modelo
# ---------------------------------------------------------------------------
def resolver_dispositivo() -> str:
    if VETORIZACAO_DEVICE == "cpu":
        return "cpu"

    import torch

    tem_placa = torch.cuda.is_available()
    if VETORIZACAO_DEVICE == "cuda" and not tem_placa:
        logger.warning("VETORIZACAO_DEVICE=cuda, mas o torch nao ve GPU. Vou vetorizar em CPU.")
        return "cpu"
    return "cuda" if tem_placa else "cpu"


def _carregar(dispositivo: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    pasta = VETORIZACAO_MODELO if Path(VETORIZACAO_MODELO).is_dir() else baixar()
    precisao = getattr(torch, VETORIZACAO_DTYPE) if dispositivo == "cuda" else torch.float32

    logger.info("Carregando %s em %s (%s)...", VETORIZACAO_MODELO, dispositivo, precisao)
    inicio = time.perf_counter()
    tokenizador = AutoTokenizer.from_pretrained(pasta)
    modelo = AutoModel.from_pretrained(pasta, torch_dtype=precisao).to(dispositivo).eval()
    logger.info("Vetorizador pronto em %.1fs.", time.perf_counter() - inicio)
    return modelo, tokenizador


def obter_modelo(dispositivo: str):
    global _modelo, _tokenizador, _dispositivo_do_modelo

    with _trava:
        if _modelo is not None and _dispositivo_do_modelo != dispositivo:
            _descarregar_sem_trava()
        if _modelo is None:
            _modelo, _tokenizador = _carregar(dispositivo)
            _dispositivo_do_modelo = dispositivo
        return _modelo, _tokenizador


def _liberar_placa() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _descarregar_sem_trava() -> None:
    global _modelo, _tokenizador, _dispositivo_do_modelo

    if _modelo is None:
        return

    logger.info("Descarregando o vetorizador e devolvendo a memoria dele.")
    _modelo = None
    _tokenizador = None
    _dispositivo_do_modelo = ""
    gc.collect()
    _liberar_placa()


def descarregar() -> None:
    with _trava:
        _descarregar_sem_trava()


def _agendar_descarga() -> None:
    global _temporizador

    if VETORIZACAO_OCIOSO_SEGUNDOS <= 0:
        return
    if _temporizador is not None:
        _temporizador.cancel()

    _temporizador = threading.Timer(VETORIZACAO_OCIOSO_SEGUNDOS, descarregar)
    _temporizador.daemon = True
    _temporizador.start()


# ---------------------------------------------------------------------------
# Vetorizacao
# ---------------------------------------------------------------------------
def vetorizar(dispositivo: str, textos: list[str]) -> tuple[list[list[float]], int]:
    """Os vetores, na ordem dos textos, e o total de tokens.

    Mean pooling sobre a ultima camada, sem normalizar — o calculo do
    fastembed 0.8 para este modelo. A mascara de atencao entra na media: o
    preenchimento de um lote nao pode puxar o vetor de um paragrafo curto.
    """
    import torch

    modelo, tokenizador = obter_modelo(dispositivo)
    limite = min(MAXIMO_DE_TOKENS, getattr(tokenizador, "model_max_length", MAXIMO_DE_TOKENS))

    vetores: list[list[float]] = []
    tokens = 0
    for inicio in range(0, len(textos), max(1, VETORIZACAO_LOTE)):
        lote = textos[inicio : inicio + VETORIZACAO_LOTE]
        entrada = tokenizador(
            lote, padding=True, truncation=True, max_length=limite, return_tensors="pt"
        ).to(dispositivo)

        with torch.inference_mode():
            saida = modelo(**entrada).last_hidden_state

        mascara = entrada["attention_mask"].unsqueeze(-1).to(saida.dtype)
        soma = (saida * mascara).sum(dim=1)
        media = soma / mascara.sum(dim=1).clamp(min=1e-9)

        vetores.extend(media.float().cpu().tolist())
        tokens += int(entrada["attention_mask"].sum())

    return vetores, tokens


def _e_falta_de_vram(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _invalido(mensagem: str, codigo: str = "entrada_invalida") -> JSONResponse:
    return JSONResponse(
        {"detail": mensagem, "error": {"code": codigo, "message": mensagem}}, status_code=422
    )


def _com_prazo(dispositivo: str, textos: list[str]):
    return prazo.com_prazo_duro(
        lambda: vetorizar(dispositivo, textos),
        VETORIZACAO_TEMPO_TRAVADO,
        "vetorizacao",
        "VETORIZACAO_TEMPO_TRAVADO",
    )


@router.post("/v1/embeddings", dependencies=[Depends(conferir)])
def vetores(pedido: PedidoDeVetores):
    """`def` e nao `async def`: a inferencia bloqueia, e no event loop ela
    congelaria o processo inteiro."""
    global _ultimo_dispositivo

    if pedido.model != VETORIZACAO_MODELO:
        # Recusar, e nao usar o daqui em silencio: o cliente guarda o vetor
        # num indice do modelo que ele PEDIU.
        return _invalido(
            f"model={pedido.model!r}, mas este worker vetoriza com {VETORIZACAO_MODELO!r}. "
            f"Vetor de outro modelo no mesmo indice estraga a busca.",
            codigo="modelo_errado",
        )

    textos = [pedido.input] if isinstance(pedido.input, str) else list(pedido.input)
    if not textos:
        return _invalido("input vazio.")
    if len(textos) > VETORIZACAO_MAXIMO_TEXTOS:
        return _invalido(
            f"{len(textos)} textos num pedido; o teto e {VETORIZACAO_MAXIMO_TEXTOS} "
            f"(VETORIZACAO_MAXIMO_TEXTOS). Divida o documento."
        )

    if not modelo_esta_no_disco():
        # Nao bloquear o pedido enquanto ~2 GB descem: o cliente reagenda.
        mensagem, espera = iniciar_download()
        return respostas.indisponivel("modelo_carregando", mensagem, retry_after=espera)

    inicio = time.perf_counter()
    try:
        with ARBITRO.usar("vetorizacao", modelo=VETORIZACAO_MODELO):
            dispositivo = resolver_dispositivo()
            if dispositivo == "cuda" and OLLAMA_DESCARREGAR_PARA_VETORIZACAO == "sim":
                ollama.descarregar_tudo()

            try:
                try:
                    lista, tokens = _com_prazo(dispositivo, textos)
                except Exception as exc:
                    if not (
                        _e_falta_de_vram(exc)
                        and dispositivo == "cuda"
                        and OLLAMA_DESCARREGAR_PARA_VETORIZACAO == "se_faltar"
                    ):
                        raise
                    # Nao coube ao lado do modelo de texto: solta o Ollama e
                    # tenta de novo, uma vez.
                    logger.info("Sem VRAM ao lado do Ollama; descarregando-o e repetindo.")
                    _liberar_placa()
                    ollama.descarregar_tudo()
                    lista, tokens = _com_prazo(dispositivo, textos)
            except WorkerTravado as exc:
                logger.critical("%s Encerrando o processo para o systemd subir outro.", exc)
                prazo.morrer_em_seguida()
                return respostas.falha_do_worker("worker_travado", str(exc))
            except Exception as exc:
                if not _e_falta_de_vram(exc):
                    logger.exception("Falha ao vetorizar")
                    return respostas.falha_do_worker(
                        "falha_na_vetorizacao", f"{type(exc).__name__}: {exc}"
                    )
                logger.warning("VRAM insuficiente para vetorizar (%s).", exc)
                descarregar()
                return respostas.indisponivel(
                    "sem_vram", "sem VRAM para vetorizar agora. O trabalho volta para a fila."
                )

            _ultimo_dispositivo = dispositivo
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    finally:
        _agendar_descarga()

    logger.info(
        "Vetorizados %s textos (%s tokens) em %.1fs (%s).",
        len(textos),
        tokens,
        time.perf_counter() - inicio,
        dispositivo,
    )

    return {
        "object": "list",
        "model": VETORIZACAO_MODELO,
        "data": [
            {"object": "embedding", "index": indice, "embedding": vetor}
            for indice, vetor in enumerate(lista)
        ],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


def estado() -> dict:
    """O que o `/health/` do worker mostra sobre esta rota."""
    with _download_trava:
        erro = _download_erro

    return {
        "modelo": VETORIZACAO_MODELO,
        "dispositivo": VETORIZACAO_DEVICE,
        "ultimo_dispositivo": _ultimo_dispositivo or "(nada vetorizado ainda)",
        "precisao": VETORIZACAO_DTYPE,
        "baixado": modelo_esta_no_disco(),
        "baixando": _baixando(),
        "erro_download": erro or None,
        "carregado": _modelo is not None,
        "lote": VETORIZACAO_LOTE,
        "ocioso_segundos": VETORIZACAO_OCIOSO_SEGUNDOS,
        "tempo_travado": VETORIZACAO_TEMPO_TRAVADO,
        "descarregar_ollama": OLLAMA_DESCARREGAR_PARA_VETORIZACAO,
    }
