"""Rota de imagem: `POST /v1/images/generations`, arbitrada.

Fala o dialeto de imagem da OpenAI (`b64_json`) porque e o unico que os
clientes conhecem — e porque isso torna o worker substituivel por um provedor
pago sem ninguem reescrever nada.

## A placa, em camadas

1. **O arbitro** (`arbitro.py`) garante um trabalho de GPU por vez no
   processo inteiro: texto, imagem e conversao disputam o mesmo lock.
2. **O Ollama e descarregado** antes de gerar, com o lock na mao. E seguro
   justamente por isso: detendo o lock, ninguem esta gerando texto.
3. **Os pesos ficam na RAM** (`enable_model_cpu_offload`) e sobem para a placa
   por submodulo. O pico cai de ~7 GB para ~5,4 GB.
4. **A placa e devolvida** depois de `IMAGEM_OCIOSO_SEGUNDOS` sem pedido.
5. **Faltando VRAM, recusa** — nao cai para CPU. Medido em uso: um lote em CPU
   consumiu horas de processador e 17 GB de RAM (float32), com a maquina
   inutilizavel. 503 devolve o trabalho para a fila do cliente, que espera.

## Tamanho

Medido nesta placa, o tamanho quase nao muda o tempo: 512x288 levou 14,5s e
1024x576 levou 17,8s — quatro vezes mais pixels por 23% mais tempo. O custo
dominante e mover pesos entre RAM e VRAM, que e fixo por geracao. Reduzir o
tamanho para "economizar" nao economiza; o que muda o tempo e o modelo.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import threading
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import ollama
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    IMAGEM_DEVICE,
    IMAGEM_GUIDANCE,
    IMAGEM_LADO_MAXIMO,
    IMAGEM_MAXIMO,
    IMAGEM_MODELO,
    IMAGEM_NEGATIVO,
    IMAGEM_OCIOSO_SEGUNDOS,
    IMAGEM_PASSOS,
    IMAGEM_PERMITIR_CPU,
    IMAGEM_TEMPO_MAXIMO,
    OLLAMA_DESCARREGAR_PARA_IMAGEM,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.imagem")

router = APIRouter()


class TempoEsgotado(RuntimeError):
    """A geracao passou de `IMAGEM_TEMPO_MAXIMO`."""


_pipeline = None
_dispositivo_do_pipeline = ""
_ultimo_dispositivo = ""
_baixado = False
_trava = threading.Lock()
_temporizador: threading.Timer | None = None


class PedidoDeImagem(BaseModel):
    prompt: str
    model: str = ""
    n: int = 1
    size: str = "1024x576"
    # Aceito e ignorado: este servico so devolve base64. Um link temporario
    # expiraria antes da publicacao e viraria imagem quebrada no site.
    response_format: str = "b64_json"
    seed: int | None = Field(default=None)


# ---------------------------------------------------------------------------
# Dispositivo e pipeline
# ---------------------------------------------------------------------------
def resolver_dispositivo() -> str:
    """cpu ou cuda, ja resolvido — nunca "auto" para dentro do codigo."""
    if IMAGEM_DEVICE == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    tem_placa = torch.cuda.is_available()
    if IMAGEM_DEVICE == "cuda" and not tem_placa:
        logger.warning(
            "IMAGEM_DEVICE=cuda, mas o torch nao ve nenhuma GPU. Vou gerar em "
            "CPU, o que leva minutos por imagem. Confira o driver e se o torch "
            "foi instalado com suporte a CUDA."
        )
        return "cpu"

    return "cuda" if tem_placa else "cpu"


def modelo_esta_no_disco() -> bool:
    """Se os pesos ja estao no cache local.

    Baixado != carregado. O primeiro custa minutos e acontece uma vez; sem
    esta pergunta, a primeira geracao de todas baixa alguns GB DENTRO da
    requisicao e o cliente ve um tempo esgotado sem explicacao.

    Uma vez verdadeiro, sempre verdadeiro: pesos nao se desbaixam, e varrer o
    cache a cada `/health/` poria trabalho de disco num endpoint rapido.
    """
    global _baixado

    if _baixado:
        return True
    if os.path.isdir(IMAGEM_MODELO):
        _baixado = True
        return True

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(IMAGEM_MODELO, local_files_only=True)
    except Exception:
        return False

    _baixado = True
    return True


def _montar_pipeline(dispositivo: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    meia = dispositivo == "cuda"
    argumentos = {
        "torch_dtype": torch.float16 if meia else torch.float32,
        "use_safetensors": True,
    }
    if meia:
        argumentos["variant"] = "fp16"

    logger.info("Carregando %s em %s...", IMAGEM_MODELO, dispositivo)
    inicio = time.perf_counter()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(IMAGEM_MODELO, **argumentos)
    except Exception:
        if not meia:
            raise
        # Nem todo repositorio publica a variante fp16. Sem esta segunda
        # tentativa, trocar de modelo no `.env` falharia com um erro sobre
        # arquivo ausente que nao menciona `variant`.
        logger.warning("%s nao tem variante fp16; carregando os pesos completos.", IMAGEM_MODELO)
        argumentos.pop("variant")
        pipe = AutoPipelineForText2Image.from_pretrained(IMAGEM_MODELO, **argumentos)

    if dispositivo == "cuda":
        # `enable_model_cpu_offload`, e NAO `.to("cuda")`. Os pesos ficam na
        # RAM e cada submodulo sobe para a placa na hora de rodar; o pico cai
        # para cerca de metade. Um `.to("cuda")` desfaz o arranjo em silencio:
        # continua funcionando e passa a ocupar o dobro.
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
    else:
        pipe.to("cpu")

    pipe.set_progress_bar_config(disable=True)
    logger.info("Modelo pronto em %.1fs.", time.perf_counter() - inicio)
    return pipe


def _obter_pipeline(dispositivo: str):
    global _pipeline, _dispositivo_do_pipeline

    with _trava:
        if _pipeline is not None and _dispositivo_do_pipeline != dispositivo:
            _descarregar_sem_trava()
        if _pipeline is None:
            _pipeline = _montar_pipeline(dispositivo)
            _dispositivo_do_pipeline = dispositivo
        return _pipeline


def _descarregar_sem_trava() -> None:
    global _pipeline, _dispositivo_do_pipeline

    if _pipeline is None:
        return

    logger.info("Descarregando o modelo de imagem e devolvendo a placa.")
    _pipeline = None
    _dispositivo_do_pipeline = ""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def descarregar() -> None:
    with _trava:
        _descarregar_sem_trava()


def _agendar_descarga() -> None:
    """Devolve a placa depois de um tempo sem pedido.

    Sem isto o modelo fica residente para sempre e o Ollama disputa o que
    sobra. Quem paga e o proximo pedido depois da pausa; num volume baixo,
    compensa.
    """
    global _temporizador

    if IMAGEM_OCIOSO_SEGUNDOS <= 0:
        return
    if _temporizador is not None:
        _temporizador.cancel()

    _temporizador = threading.Timer(IMAGEM_OCIOSO_SEGUNDOS, descarregar)
    # Daemon: um temporizador pendente nao pode segurar o desligamento.
    _temporizador.daemon = True
    _temporizador.start()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _medidas(tamanho: str) -> tuple[int, int]:
    """ "1024x576" -> (1024, 576), recusando o que o modelo nao aceita."""
    try:
        largura, _, altura = tamanho.lower().partition("x")
        largura, altura = int(largura), int(altura)
    except ValueError as exc:
        raise HTTPException(422, f"size invalido: {tamanho!r}. Use algo como 1024x576.") from exc

    for medida in (largura, altura):
        if medida <= 0 or medida % 8:
            # Os modelos de difusao trabalham num espaco latente 8x menor. Um
            # lado que nao e multiplo de 8 nao falha: ele e arredondado por
            # dentro, e a imagem volta com tamanho diferente do pedido.
            raise HTTPException(422, f"size {tamanho!r}: cada lado precisa ser multiplo de 8.")
        if medida > IMAGEM_LADO_MAXIMO:
            raise HTTPException(
                422,
                f"size {tamanho!r} passa de {IMAGEM_LADO_MAXIMO}px por lado, que e o "
                f"que esta placa comporta. Ajuste IMAGEM_LADO_MAXIMO se ela comportar mais.",
            )

    return largura, altura


@router.post("/v1/images/generations", dependencies=[Depends(conferir)])
def gerar(pedido: PedidoDeImagem):
    """`def` e nao `async def`: rodar difusao no event loop congela o
    processo, e ate o `/health/` para de responder."""
    global _ultimo_dispositivo

    if not pedido.prompt.strip():
        raise HTTPException(422, "prompt vazio.")

    quantas = max(1, min(pedido.n, IMAGEM_MAXIMO))
    largura, altura = _medidas(pedido.size)

    try:
        with ARBITRO.usar("imagem", modelo=IMAGEM_MODELO):
            # Com o lock na mao: ninguem esta gerando texto, entao mandar o
            # Ollama soltar a VRAM e seguro. E a razao de o arbitro existir.
            if OLLAMA_DESCARREGAR_PARA_IMAGEM:
                ollama.descarregar_tudo()

            dispositivo = resolver_dispositivo()
            try:
                imagens = gerar_imagens(dispositivo, pedido, quantas, largura, altura)
            except TempoEsgotado as exc:
                logger.warning("%s", exc)
                # Sem `Retry-After`: para o codigo `timeout` a documentacao
                # manda reduzir o pedido, e nao voltar igual mais tarde. Mandar
                # o cabecalho junto seria o contrato se contradizendo dentro da
                # mesma resposta.
                return respostas.indisponivel("timeout", str(exc), retry_after=None)
            except Exception as exc:
                if not _e_falta_de_vram(exc) or dispositivo == "cpu":
                    raise

                if not IMAGEM_PERMITIR_CPU:
                    logger.warning(
                        "VRAM insuficiente (%s) e IMAGEM_PERMITIR_CPU desligado. "
                        "Recusando em vez de gastar horas de CPU.",
                        exc,
                    )
                    descarregar()
                    return respostas.indisponivel(
                        "sem_vram",
                        "sem VRAM para gerar agora. O trabalho volta para a fila. "
                        "Para gerar em CPU assim mesmo, IMAGEM_PERMITIR_CPU=true "
                        "(meca antes: pode levar horas).",
                        retry_after=600,
                    )

                logger.warning(
                    "VRAM insuficiente (%s). Refazendo em CPU, por "
                    "IMAGEM_PERMITIR_CPU — vai levar MUITO mais tempo.",
                    exc,
                )
                descarregar()
                dispositivo = "cpu"
                imagens = gerar_imagens(dispositivo, pedido, quantas, largura, altura)

            _ultimo_dispositivo = dispositivo
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    finally:
        _agendar_descarga()

    logger.info("%s imagem(ns) %sx%s (%s).", len(imagens), largura, altura, dispositivo)
    return {
        "created": int(time.time()),
        "data": [
            {"b64_json": base64.b64encode(png).decode("ascii"), "revised_prompt": pedido.prompt}
            for png in imagens
        ],
    }


def gerar_imagens(
    dispositivo: str, pedido: PedidoDeImagem, quantas: int, largura: int, altura: int
) -> list[bytes]:
    """As imagens em PNG. Quem chamou converte para o formato que quiser."""
    import torch

    pipe = _obter_pipeline(dispositivo)

    gerador = None
    if pedido.seed is not None:
        # Semente fixa reproduz uma imagem exata — util ao ajustar o prompt.
        # Sem ela cada opcao do lote e diferente, que e o ponto de um lote.
        gerador = torch.Generator(device="cpu").manual_seed(pedido.seed)

    saida = pipe(
        prompt=pedido.prompt,
        negative_prompt=IMAGEM_NEGATIVO or None,
        num_images_per_prompt=quantas,
        num_inference_steps=IMAGEM_PASSOS,
        guidance_scale=IMAGEM_GUIDANCE,
        width=largura,
        height=altura,
        generator=gerador,
        callback_on_step_end=vigia_do_relogio(dispositivo, largura, altura),
    )

    bytes_das_imagens = []
    for imagem in saida.images:
        memoria = io.BytesIO()
        imagem.save(memoria, format="PNG")
        bytes_das_imagens.append(memoria.getvalue())
    return bytes_das_imagens


def vigia_do_relogio(dispositivo: str, largura: int, altura: int):
    """Interrompe a difusao quando ela passa do orcamento.

    O `callback_on_step_end` e o unico ponto em que da para desistir: o laco
    nao olha para sinal nem para timeout, e por isso um `systemctl restart`
    fica preso ate o `TimeoutStopSec` — foi o que aconteceu numa geracao que
    levou horas.
    """
    if IMAGEM_TEMPO_MAXIMO <= 0:
        return None

    limite = time.perf_counter() + IMAGEM_TEMPO_MAXIMO

    def conferir_relogio(pipe, passo, timestep, argumentos):
        if time.perf_counter() > limite:
            raise TempoEsgotado(
                f"a geracao passou de {IMAGEM_TEMPO_MAXIMO}s em {dispositivo} no "
                f"passo {passo} de {IMAGEM_PASSOS} ({largura}x{altura}). Reduza os "
                f"passos, use um modelo menor, ou aumente IMAGEM_TEMPO_MAXIMO."
            )
        return argumentos

    return conferir_relogio


def _e_falta_de_vram(exc: Exception) -> bool:
    """Distingue "faltou memoria na placa" de um defeito de verdade.

    Pelo texto, e nao so pelo tipo: a mesma falta chega tambem como
    `RuntimeError` vinda do cuBLAS ou do cuDNN, com a mensagem dentro.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass

    texto = str(exc).lower()
    return "out of memory" in texto or "cuda error" in texto


def estado() -> dict:
    """O que o `/health/` do worker mostra sobre esta rota."""
    return {
        "modelo": IMAGEM_MODELO,
        "dispositivo": IMAGEM_DEVICE,
        "ultimo_dispositivo": _ultimo_dispositivo or "(nada gerado ainda)",
        "baixado": modelo_esta_no_disco(),
        "carregado": _pipeline is not None,
        "passos": IMAGEM_PASSOS,
        "permite_cpu": IMAGEM_PERMITIR_CPU,
        "tempo_maximo": IMAGEM_TEMPO_MAXIMO,
    }
