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
import importlib.util
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
    IMAGEM_AREA_MAXIMA_MP,
    IMAGEM_DEVICE,
    IMAGEM_DTYPE,
    IMAGEM_GUIDANCE,
    IMAGEM_LADO_MAXIMO,
    IMAGEM_MAXIMO,
    IMAGEM_MODELO,
    IMAGEM_OCIOSO_SEGUNDOS,
    IMAGEM_PASSOS,
    IMAGEM_PERMITIR_CPU,
    IMAGEM_QUANTIZAR,
    IMAGEM_QUANTIZAR_COMPONENTES,
    IMAGEM_SCHEDULER,
    IMAGEM_TEMPO_MAXIMO,
    IMAGEM_TEMPO_TRAVADO,
    IMAGEM_VAE,
    OLLAMA_DESCARREGAR_PARA_IMAGEM,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.imagem")

router = APIRouter()


class TempoEsgotado(RuntimeError):
    """A geracao passou de `IMAGEM_TEMPO_MAXIMO`."""


class WorkerTravado(RuntimeError):
    """O trabalho passou do `IMAGEM_TEMPO_TRAVADO`: o processo nao e confiavel."""


_pipeline = None
_dispositivo_do_pipeline = ""
# "sdxl", "outra" ou "" (ainda nao carregou). A grade de proporcoes e do SDXL:
# avisar sobre ela num modelo de outra familia mandaria mudar o que esta certo.
_familia_do_pipeline = ""
_ultimo_dispositivo = ""
_baixado = False
_trava = threading.Lock()
_temporizador: threading.Timer | None = None


# As proporcoes em que o SDXL foi treinado. Metade da grade; a outra metade sao
# as mesmas deitadas, e `GRADE_DO_SDXL` junta as duas.
#
# Todas ficam em torno de 1,05 megapixel — a grade e de PROPORCOES a area
# constante, e nao de tamanhos livres. Gerar fora dela nao da erro: da assunto
# duplicado, geometria torta e composicao incoerente, porque o modelo nunca viu
# um enquadramento daquele formato.
_MEIA_GRADE = (
    (512, 2048),
    (512, 1984),
    (512, 1920),
    (512, 1856),
    (576, 1792),
    (576, 1728),
    (576, 1664),
    (640, 1600),
    (640, 1536),
    (704, 1472),
    (704, 1408),
    (704, 1344),
    (768, 1344),
    (768, 1280),
    (832, 1216),
    (832, 1152),
    (896, 1152),
    (896, 1088),
    (960, 1088),
    (960, 1024),
    (1024, 1024),
)
GRADE_DO_SDXL = tuple(
    sorted(
        {(largura, altura) for largura, altura in _MEIA_GRADE}
        | {(altura, largura) for largura, altura in _MEIA_GRADE}
    )
)

# Amostradores disponiveis, por nome curto. O amostrador decide como os passos
# caminham do ruido para a imagem, e com pouco passo a escolha aparece.
#
# Publicado (e nao privado) porque a `bancada.py` compara amostradores usando
# ESTE mapa: uma bancada com a sua propria lista mediria uma configuracao que o
# servico nao usa, que e o jeito mais facil de escolher errado.
SCHEDULERS = {
    "dpm++2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++2m": ("DPMSolverMultistepScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "ddim": ("DDIMScheduler", {}),
}

# Palavras que so aparecem em texto portugues, para o aviso de prompt.
#
# Nao e deteccao de idioma: e uma peneira grosseira de proposito, e errar para
# "nao avisa" e o certo. Um aviso a mais num prompt em ingles seria ruido no
# journal; o aviso que falta e so um aviso que falta.
_PISTAS_DE_PORTUGUES = (
    " de ",
    " da ",
    " do ",
    " com ",
    " uma ",
    " um ",
    " que ",
    " para ",
    " sem ",
    "cao ",
    "ção",
)


class PedidoDeImagem(BaseModel):
    prompt: str
    model: str = ""
    n: int = 1
    # 1344x768 e nao 1024x576: e a proporcao 16:9 que o SDXL conhece do
    # treino. Veja `IMAGEM_LADO_MAXIMO` no `config.py`.
    size: str = "1344x768"
    # Aceito e ignorado: este servico so devolve base64. Um link temporario
    # expiraria antes da publicacao e viraria imagem quebrada no site.
    response_format: str = "b64_json"
    seed: int | None = Field(default=None)
    # Repassado como veio, e so quando veio. O worker nao tem negativo proprio:
    # um padrao aplicado a todo pedido brigava com quem pedia texto na imagem.
    # Modelos destilados (Z-Image-Turbo, FLUX schnell) rodam com guidance 0 e
    # ignoram o negativo.
    negative_prompt: str | None = None


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


def conferir_configuracao() -> None:
    """Recusa subir com precisao ou quantizacao invalidas.

    Na subida, e nao no primeiro pedido: o modelo carrega preguicosamente, e um
    `bitsandbytes` ausente so apareceria como 500 na primeira imagem.
    """
    if IMAGEM_DTYPE not in ("float16", "bfloat16"):
        raise RuntimeError(f"IMAGEM_DTYPE={IMAGEM_DTYPE!r}. Use float16 ou bfloat16.")
    if IMAGEM_QUANTIZAR not in ("nao", "4bit", "8bit"):
        raise RuntimeError(f"IMAGEM_QUANTIZAR={IMAGEM_QUANTIZAR!r}. Use nao, 4bit ou 8bit.")
    if 0 < IMAGEM_TEMPO_TRAVADO <= IMAGEM_TEMPO_MAXIMO:
        raise RuntimeError(
            f"IMAGEM_TEMPO_TRAVADO={IMAGEM_TEMPO_TRAVADO} precisa ser maior que "
            f"IMAGEM_TEMPO_MAXIMO={IMAGEM_TEMPO_MAXIMO}: uma geracao lenta mas saudavel "
            f"tem que levar 503 `timeout`, e nao derrubar o processo."
        )
    if IMAGEM_QUANTIZAR != "nao" and importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            f"IMAGEM_QUANTIZAR={IMAGEM_QUANTIZAR}, mas o bitsandbytes nao esta instalado.\n"
            f"  ./venv/bin/pip install bitsandbytes"
        )


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


def aplicar_scheduler(pipe, nome: str):
    """Troca o amostrador do pipeline. `nome` vazio mantem o do modelo.

    `from_config` e nao um construtor novo: o amostrador precisa herdar a
    configuracao de ruido DO MODELO (betas, passos de treino). Construido do
    zero com os padroes da classe, ele gera — e gera errado, sem erro nenhum.
    """
    if not nome:
        return pipe

    if nome not in SCHEDULERS:
        raise RuntimeError(
            f"IMAGEM_SCHEDULER={nome!r} nao existe. Use um de: {', '.join(sorted(SCHEDULERS))}"
        )

    atual = type(pipe.scheduler).__name__
    if "FlowMatch" in atual:
        # Os amostradores da lista sao de difusao classica (SDXL, SD 1.5). Num
        # modelo de "flow matching" (Z-Image, FLUX, SD 3.5) o `from_config`
        # nao reclama — ele gera ruido. Recusar e o unico jeito de nao sair
        # imagem estragada sem erro.
        raise RuntimeError(
            f"IMAGEM_SCHEDULER={nome!r} nao serve a este modelo, que usa {atual}. "
            f"Deixe IMAGEM_SCHEDULER vazio no .env."
        )

    import diffusers

    classe, extras = SCHEDULERS[nome]
    pipe.scheduler = getattr(diffusers, classe).from_config(pipe.scheduler.config, **extras)
    logger.info("Amostrador: %s (%s)", nome, classe)

    return pipe


# O `scaling_factor` do VAE do SDXL. E ele que identifica a familia, e nao o
# nome do repositorio: `RealVisXL`, `Juggernaut-XL` e `animagine-xl` sao todos
# SDXL e nao tem substring em comum que de para procurar sem errar.
_ESCALA_DO_VAE_DO_SDXL = 0.13025


def _avisar_do_vae(pipe, vae: str, meia: bool) -> None:
    """O VAE do SDXL estoura em float16, e o defeito e visivel.

    Manchas, faixas de cor e, as vezes, imagem preta — o tipo de defeito que
    quem olha chama de "cara de IA" sem saber apontar o que e.

    A familia e descoberta no OBJETO CARREGADO, e nao no nome do modelo. Pelo
    nome nao da: `RealVisXL_V5.0` e SDXL e nao contem "sdxl"; procurar "xl"
    solto acertaria ele e erraria em qualquer repositorio com "xl" no meio de
    outra palavra. O `scaling_factor` do VAE, ao contrario, e um fato do modelo
    — e os VAE de 16 canais (SD 3.5, FLUX) nao tem esse problema e nao caem
    aqui.

    O worker nao corrige sozinho: trocar o VAE por conta propria mudaria a
    saida de quem nao pediu. Mas ficar calado seria deixar a imagem sair pior
    sem nada denunciando, que e o unico defeito que este servico nao aceita.
    """
    global _familia_do_pipeline

    try:
        config = pipe.vae.config
        canais = int(config.latent_channels)
        escala = float(config.scaling_factor)
    except Exception:
        _familia_do_pipeline = "outra"
        return

    e_sdxl = canais == 4 and abs(escala - _ESCALA_DO_VAE_DO_SDXL) <= 1e-4
    _familia_do_pipeline = "sdxl" if e_sdxl else "outra"

    if vae or not meia or not e_sdxl:
        return

    logger.warning(
        "Este modelo usa o VAE do SDXL (scaling_factor=%s) em float16, e "
        "IMAGEM_VAE esta vazio. Esse VAE estoura em float16 e produz manchas e "
        "faixas de cor. Ponha IMAGEM_VAE=madebyollin/sdxl-vae-fp16-fix no .env "
        "e reinicie.",
        escala,
    )


def _montar_pipeline(dispositivo: str, modelo: str = "", vae: str = "", scheduler: str = ""):
    """O pipeline como o SERVICO o monta.

    Os parametros existem para a `bancada.py` comparar variantes por aqui, e
    nao por um caminho proprio: uma bancada que monta o pipeline de outro jeito
    mede uma configuracao que o servico nao usa.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    modelo = modelo or IMAGEM_MODELO
    vae = vae if vae != "" else IMAGEM_VAE
    scheduler = scheduler if scheduler != "" else IMAGEM_SCHEDULER

    meia = dispositivo == "cuda"
    precisao = getattr(torch, IMAGEM_DTYPE) if meia else torch.float32
    argumentos = {
        "torch_dtype": precisao,
        "use_safetensors": True,
    }
    if meia and IMAGEM_DTYPE == "float16":
        # So em float16: e a unica variante que os repositorios SDXL publicam
        # a parte. Um modelo em bfloat16 ja vem assim no ramo principal.
        argumentos["variant"] = "fp16"

    if meia and IMAGEM_QUANTIZAR != "nao":
        from diffusers.quantizers import PipelineQuantizationConfig

        if IMAGEM_QUANTIZAR == "4bit":
            backend = "bitsandbytes_4bit"
            extras = {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": precisao,
            }
        else:
            backend, extras = "bitsandbytes_8bit", {"load_in_8bit": True}

        logger.info(
            "Quantizando %s em %s.", ", ".join(IMAGEM_QUANTIZAR_COMPONENTES), IMAGEM_QUANTIZAR
        )
        argumentos["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=backend,
            quant_kwargs=extras,
            components_to_quantize=IMAGEM_QUANTIZAR_COMPONENTES,
        )

    if vae:
        from diffusers import AutoencoderKL

        logger.info("VAE alternativo: %s", vae)
        argumentos["vae"] = AutoencoderKL.from_pretrained(vae, torch_dtype=precisao)

    logger.info("Carregando %s em %s...", modelo, dispositivo)
    inicio = time.perf_counter()
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(modelo, **argumentos)
    except Exception:
        if not meia:
            raise
        # Nem todo repositorio publica a variante fp16. Sem esta segunda
        # tentativa, trocar de modelo no `.env` falharia com um erro sobre
        # arquivo ausente que nao menciona `variant`.
        logger.warning("%s nao tem variante fp16; carregando os pesos completos.", modelo)
        argumentos.pop("variant")
        pipe = AutoPipelineForText2Image.from_pretrained(modelo, **argumentos)

    aplicar_scheduler(pipe, scheduler)
    # Depois de carregar, porque a familia do VAE se descobre no objeto.
    _avisar_do_vae(pipe, vae, meia)

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

    megapixels = largura * altura / 1e6
    if megapixels > IMAGEM_AREA_MAXIMA_MP:
        # Teto de AREA, alem do teto por lado. Os dois precisam existir: um
        # teto de lado em 1536 sozinho deixaria passar 1536x1536, que e o
        # dobro da area de treino do modelo e VRAM que esta placa nao tem.
        raise HTTPException(
            422,
            f"size {tamanho!r} tem {megapixels:.2f} megapixels, acima do teto de "
            f"{IMAGEM_AREA_MAXIMA_MP} MP. O SDXL foi treinado em torno de 1,05 MP e "
            f"gera pior acima disso. Proporcao mais proxima na grade de treino: "
            f"{_perto_na_grade(largura, altura)}. Ajuste IMAGEM_AREA_MAXIMA_MP se "
            f"esta placa comportar mais.",
        )

    for medida in (largura, altura):
        if medida <= 0 or medida % 16:
            # 16, e nao 8: o espaco latente e 8x menor, e os modelos de
            # transformer (Z-Image, FLUX, SD 3.5) ainda o agrupam em blocos de
            # 2x2. Com 8, um lado como 1352 passava aqui e o pipeline recusava
            # la dentro, com um 500. Toda a grade do SDXL e multipla de 64.
            raise HTTPException(422, f"size {tamanho!r}: cada lado precisa ser multiplo de 16.")
        if medida > IMAGEM_LADO_MAXIMO:
            raise HTTPException(
                422,
                f"size {tamanho!r} passa de {IMAGEM_LADO_MAXIMO}px por lado, que e o "
                f"que esta placa comporta. Ajuste IMAGEM_LADO_MAXIMO se ela comportar mais.",
            )

    return largura, altura


def _perto_na_grade(largura: int, altura: int) -> str:
    """A proporcao treinada mais parecida com a pedida.

    Comparada pela PROPORCAO e nao pela area, porque a area de toda a grade e
    praticamente a mesma (0,95 a 1,05 MP): comparar por area mandaria todo
    mundo para o 1024x1024. Quem pede 1200x630 — o tamanho de `og:image` — quer
    aquele FORMATO, e o que serve a ele e o 1344x704.
    """
    if not altura:
        return "1024x1024"

    pedida = largura / altura
    melhor = min(GRADE_DO_SDXL, key=lambda medida: abs(medida[0] / medida[1] - pedida))

    return f"{melhor[0]}x{melhor[1]}"


def _avisar_de_tamanho_fora_da_grade(largura: int, altura: int) -> None:
    """Avisa, e nao recusa.

    Recusar quebraria quem tem motivo para pedir outro formato — 1200x630 e o
    tamanho que as redes sociais pedem para `og:image`, e ele nao esta na
    grade. Quem pede assim aceita o custo; o que nao se aceita e pagar o custo
    sem saber que existe.
    """
    if (largura, altura) in GRADE_DO_SDXL:
        return
    if _familia_do_pipeline != "sdxl":
        return

    logger.warning(
        "size %dx%d esta fora da grade de proporcoes em que o SDXL foi treinado. "
        "Nao da erro, mas costuma sair com assunto duplicado e geometria torta. "
        "A proporcao treinada mais proxima e %s.",
        largura,
        altura,
        _perto_na_grade(largura, altura),
    )


@router.post("/v1/images/generations", dependencies=[Depends(conferir)])
def gerar(pedido: PedidoDeImagem):
    """`def` e nao `async def`: rodar difusao no event loop congela o
    processo, e ate o `/health/` para de responder."""
    global _ultimo_dispositivo

    if not pedido.prompt.strip():
        raise HTTPException(422, "prompt vazio.")

    _avisar_de_prompt_em_portugues(pedido.prompt)

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
                imagens = _com_prazo_duro(
                    lambda: gerar_imagens(dispositivo, pedido, quantas, largura, altura)
                )
            except WorkerTravado as exc:
                logger.critical("%s Encerrando o processo para o systemd subir outro.", exc)
                _morrer_em_seguida()
                return respostas.falha_do_worker("worker_travado", str(exc))
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
                imagens = _com_prazo_duro(
                    lambda: gerar_imagens(dispositivo, pedido, quantas, largura, altura)
                )

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


def _avisar_de_prompt_em_portugues(prompt: str) -> None:
    """Prompt em portugues e o defeito de qualidade mais caro desta rota.

    Os codificadores de texto do SDXL (CLIP ViT-L e OpenCLIP ViT-bigG) foram
    treinados em legendas da web, esmagadoramente em ingles. Um prompt em
    portugues nao da erro: ele gera uma imagem a partir do pouco que sobrou de
    sinal, e o resultado e generico, mal composto, com aquela "cara de IA" que
    ninguem sabe apontar de onde vem.

    O worker NAO traduz: traduzir em silencio mudaria o pedido de quem chamou,
    e um prompt trocado sem aviso e pior que um prompt ruim. Ele avisa, e quem
    integra decide — foi este aviso que revelou que o prompt NEGATIVO deste
    servico estava em portugues desde o inicio, sem efeito nenhum.
    """
    if _familia_do_pipeline == "outra":
        # O aviso e sobre os CLIP do SDXL. O Z-Image, por exemplo, le o prompt
        # com um modelo de linguagem (Qwen3) que entende portugues.
        return

    baixo = f" {prompt.lower()} "
    pistas = [pista.strip() for pista in _PISTAS_DE_PORTUGUES if pista in baixo]

    if len(pistas) >= 2:
        logger.warning(
            "O prompt parece estar em portugues (%s). Os codificadores de texto "
            "do SDXL foram treinados em ingles: isto nao da erro, mas gera uma "
            "imagem pior. Mande o prompt em ingles. Prompt: %.120s",
            ", ".join(pistas[:4]),
            prompt,
        )


def _com_prazo_duro(trabalho):
    """Roda `trabalho` numa thread e desiste dela depois de `IMAGEM_TEMPO_TRAVADO`.

    A thread nao e cancelada — Python nao sabe fazer isso, e o torch menos
    ainda. Desistir aqui so faz sentido junto com matar o processo, que e o
    que quem chama faz.
    """
    if IMAGEM_TEMPO_TRAVADO <= 0:
        return trabalho()

    resultado: dict = {}

    def alvo():
        try:
            resultado["ok"] = trabalho()
        except BaseException as exc:  # repassada inteira: o OOM de CUDA inclusive
            resultado["erro"] = exc

    thread = threading.Thread(target=alvo, name="imagem", daemon=True)
    thread.start()
    thread.join(IMAGEM_TEMPO_TRAVADO)

    if thread.is_alive():
        raise WorkerTravado(
            f"o trabalho de imagem passou de {IMAGEM_TEMPO_TRAVADO}s (IMAGEM_TEMPO_TRAVADO) "
            f"sem terminar nem falhar: travou na carga do modelo ou dentro de um passo. "
            f"Causa comum: memoria. Veja `journalctl --user -u worker-gpu` e o "
            f"`memory.events` da unit."
        )
    if "erro" in resultado:
        raise resultado["erro"]
    return resultado["ok"]


def _morrer_em_seguida(atraso: float = 2.0) -> None:
    """Encerra o processo com erro, depois de a resposta 500 sair.

    Codigo 1, e nao 0: e o que o systemd le como falha, e o que dispara o
    `Restart=always` e o `OnFailure=` (o aviso de queda).
    """
    temporizador = threading.Timer(atraso, os._exit, args=(1,))
    temporizador.daemon = True
    temporizador.start()


def gerar_imagens(
    dispositivo: str, pedido: PedidoDeImagem, quantas: int, largura: int, altura: int
) -> list[bytes]:
    """As imagens em PNG. Quem chamou converte para o formato que quiser."""
    import torch

    pipe = _obter_pipeline(dispositivo)
    # Depois de carregar: so aqui se sabe a familia do modelo, e a grade de
    # proporcoes so vale para o SDXL.
    _avisar_de_tamanho_fora_da_grade(largura, altura)

    gerador = None
    if pedido.seed is not None:
        # Semente fixa reproduz uma imagem exata — util ao ajustar o prompt.
        # Sem ela cada opcao do lote e diferente, que e o ponto de um lote.
        gerador = torch.Generator(device="cpu").manual_seed(pedido.seed)

    saida = pipe(
        prompt=pedido.prompt,
        negative_prompt=pedido.negative_prompt or None,
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
        "guidance": IMAGEM_GUIDANCE,
        # Os tres que decidem qualidade e nao aparecem em lugar nenhum senao
        # aqui. `vae` vazio num modelo SDXL e o defeito silencioso que o log
        # avisa na carga; publicado, da para conferir de fora, sem journal.
        "vae": IMAGEM_VAE or "(o do modelo)",
        "amostrador": IMAGEM_SCHEDULER or "(o do modelo)",
        "precisao": IMAGEM_DTYPE,
        "quantizacao": IMAGEM_QUANTIZAR,
        "lado_maximo": IMAGEM_LADO_MAXIMO,
        "area_maxima_mp": IMAGEM_AREA_MAXIMA_MP,
        # A grade de proporcoes do SDXL, publicada para o cliente validar
        # contra ELA e nao contra uma copia propria que envelhece.
        "grade": [f"{largura}x{altura}" for largura, altura in GRADE_DO_SDXL],
        "permite_cpu": IMAGEM_PERMITIR_CPU,
        "tempo_maximo": IMAGEM_TEMPO_MAXIMO,
        "tempo_travado": IMAGEM_TEMPO_TRAVADO,
    }
