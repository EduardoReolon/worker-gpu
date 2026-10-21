"""Rota de edicao: `POST /v1/images/edits`, arbitrada.

O dialeto de edicao de imagem da OpenAI, com dois campos a mais
(`reference_image` e `reference_strength`). Quem usa e o PubliBot, para o
video por cenas: cada cena e um primeiro quadro que depois vira animacao.

    reference_image (ate 3)   os personagens: a cena sai com a aparencia deles
    image + mask              pinta SO a parte transparente da mascara, e o
                              resto sai exatamente como veio (a foto real
                              na janela, o ceu do lugar de fundo)

## Dois modelos, escolhidos por pedido

- **Sem referencia**, o `IMAGEM_MODELO` (o de mais qualidade, hoje o
  Z-Image) — inclusive para pintar a mascara: o diffusers tem pintura para ele
  com os MESMOS pesos.
- **Com referencia**, o `IMAGEM_REFERENCIA_MODELO` com IP-Adapter. O Z-Image
  nao aceita IP-Adapter (ele so existe para o SDXL e familia); o padrao e o
  RealVisXL, o SDXL de foto mais realista.

Um so fica carregado por vez: alternar custa uma recarga.

## "Exatamente como veio"

Pintar passa a imagem inteira pelo VAE, e a parte que devia ficar intacta
volta levemente diferente. Por isso, depois de gerar, os pixels da parte
opaca da mascara sao COLADOS de volta da base: a foto sai identica, pixel a
pixel.
"""

from __future__ import annotations

import base64
import io
import logging
import time

from fastapi import APIRouter, Depends, Form, UploadFile
from fastapi.responses import JSONResponse

import imagem
import ollama
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    IMAGEM_MAXIMO,
    IMAGEM_REFERENCIAS_MAXIMAS,
    OLLAMA_DESCARREGAR_PARA_IMAGEM,
)
from prazo import WorkerTravado
from seguranca import conferir

logger = logging.getLogger("worker-gpu.edicao")

router = APIRouter()

ARQUIVO_MAXIMO = 20 * 1024 * 1024
FORCA_PADRAO = 0.6


class PedidoInvalido(ValueError):
    pass


def _invalido(mensagem: str, codigo: str = "pedido_invalido", status: int = 422):
    return JSONResponse(
        {"detail": mensagem, "error": {"code": codigo, "message": mensagem}}, status_code=status
    )


def _abrir(arquivo: UploadFile, campo: str):
    from PIL import Image, UnidentifiedImageError

    conteudo = arquivo.file.read(ARQUIVO_MAXIMO + 1)
    if not conteudo:
        raise PedidoInvalido(f"{campo} vazio.")
    if len(conteudo) > ARQUIVO_MAXIMO:
        raise PedidoInvalido(f"{campo} maior que 20 MB.")
    try:
        figura = Image.open(io.BytesIO(conteudo))
        figura.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise PedidoInvalido(f"{campo} nao e uma imagem legivel: {exc}") from exc
    return figura


def mascara_para_o_diffusers(mascara):
    """A mascara da OpenAI -> (a do diffusers, a de manter).

    Na OpenAI, transparente (alfa 0) = pintar e opaco = manter. No diffusers,
    branco = pintar. Devolve as duas em tons de cinza: `pintar` (branco onde
    pinta) para o pipeline, e `manter` (branco onde mantem) para colar a base
    de volta depois.
    """
    if "A" not in mascara.getbands():
        raise PedidoInvalido(
            "a mascara precisa de canal alfa: transparente (alfa 0) = pintar, opaco = manter."
        )
    alfa = mascara.getchannel("A")
    manter = alfa.point(lambda valor: 255 if valor >= 128 else 0)
    pintar = manter.point(lambda valor: 255 - valor)
    return pintar, manter


def colar_o_que_fica(base, gerada, manter):
    """A parte opaca da mascara volta da base, pixel a pixel."""
    from PIL import Image

    return Image.composite(base, gerada.convert("RGB"), manter)


def editar_imagens(
    dispositivo: str,
    *,
    prompt: str,
    negativo: str | None,
    quantas: int,
    largura: int,
    altura: int,
    semente: int | None,
    base=None,
    mascara=None,
    referencias: list | None = None,
    forca: float = FORCA_PADRAO,
) -> list[bytes]:
    """As imagens em PNG. Escolhe o perfil (com ou sem referencia) e o modo
    (gera do zero ou pinta a mascara)."""
    import torch

    perfil = imagem.perfil_referencia() if referencias else imagem.perfil_principal()
    modo = "pintura" if mascara is not None else "base"
    pipe = imagem._obter_pipeline(dispositivo, perfil, modo)

    argumentos = {
        "prompt": prompt,
        "negative_prompt": negativo or None,
        "num_images_per_prompt": quantas,
        "num_inference_steps": perfil.passos,
        "guidance_scale": perfil.guidance,
        "width": largura,
        "height": altura,
        "callback_on_step_end": imagem.vigia_do_relogio(dispositivo, largura, altura),
    }
    if semente is not None:
        argumentos["generator"] = torch.Generator(device="cpu").manual_seed(semente)

    manter = None
    if mascara is not None:
        pintar, manter = mascara_para_o_diffusers(mascara)
        argumentos["image"] = base
        argumentos["mask_image"] = pintar

    if referencias:
        pipe.set_ip_adapter_scale(forca)
        # A lista de fora e uma por IP-Adapter (aqui, um); a de dentro, as
        # imagens dele — os personagens da cena.
        argumentos["ip_adapter_image"] = [referencias]

    saida = pipe(**argumentos)

    resultado = []
    for gerada in saida.images:
        if manter is not None:
            gerada = colar_o_que_fica(base, gerada, manter)
        memoria = io.BytesIO()
        gerada.save(memoria, format="PNG")
        resultado.append(memoria.getvalue())
    return resultado


@router.post("/v1/images/edits", dependencies=[Depends(conferir)])
def editar(
    prompt: str = Form(...),
    model: str | None = Form(default=None),
    size: str | None = Form(default=None),
    n: str | None = Form(default="1"),
    response_format: str | None = Form(default="b64_json"),
    negative_prompt: str | None = Form(default=None),
    reference_strength: str | None = Form(default=None),
    seed: str | None = Form(default=None),
    image: UploadFile | None = None,
    mask: UploadFile | None = None,
    reference_image: list[UploadFile] | None = None,
):
    """`model` e `response_format` sao aceitos e ignorados: o modelo e o da
    maquina, escolhido pelo pedido (com ou sem referencia), e a resposta e
    sempre `b64_json`.

    `def` e nao `async def`: rodar difusao no event loop congela o processo.
    """
    global _ultimo_perfil

    avisos: list[str] = []
    try:
        if not prompt.strip():
            raise PedidoInvalido("prompt vazio.")
        try:
            quantas = max(1, min(int(n or 1), IMAGEM_MAXIMO))
        except ValueError as exc:
            raise PedidoInvalido(f"n={n!r} nao e um inteiro.") from exc
        try:
            forca = float(reference_strength) if reference_strength else FORCA_PADRAO
        except ValueError as exc:
            raise PedidoInvalido(f"reference_strength={reference_strength!r}.") from exc
        if not 0 <= forca <= 1:
            raise PedidoInvalido("reference_strength vai de 0 a 1.")
        try:
            semente = int(seed) if seed not in (None, "") else None
        except ValueError as exc:
            raise PedidoInvalido(f"seed={seed!r} nao e um inteiro.") from exc

        base = _abrir(image, "image").convert("RGB") if image is not None else None
        mascara = _abrir(mask, "mask") if mask is not None else None
        if mascara is not None and base is None:
            raise PedidoInvalido("mask sem image: a mascara diz o que pintar NA image.")
        if base is not None and mascara is None:
            raise PedidoInvalido(
                "image sem mask nao e suportado: mande a mascara (transparente = pintar)."
            )
        if mascara is not None and mascara.size != base.size:
            raise PedidoInvalido(
                f"a mascara ({mascara.size[0]}x{mascara.size[1]}) precisa ter o tamanho da "
                f"image ({base.size[0]}x{base.size[1]})."
            )
        if mascara is not None:
            mascara_para_o_diffusers(mascara)  # recusa ja, antes de tomar a placa

        referencias = [
            _abrir(arquivo, "reference_image").convert("RGB") for arquivo in reference_image or []
        ]
        if len(referencias) > IMAGEM_REFERENCIAS_MAXIMAS:
            raise PedidoInvalido(
                f"{len(referencias)} reference_image; o maximo e {IMAGEM_REFERENCIAS_MAXIMAS}."
            )
    except PedidoInvalido as exc:
        return _invalido(str(exc))

    if referencias and imagem.perfil_referencia() is None:
        avisos.append(
            "reference_image ignorado: IMAGEM_REFERENCIA_MODELO esta vazio neste worker; a "
            "cena saiu so pela descricao."
        )
        referencias = []

    tamanho = size or (f"{base.size[0]}x{base.size[1]}" if base is not None else "1344x768")
    try:
        largura, altura = imagem._medidas(tamanho)
    except Exception as exc:  # HTTPException 422 do `_medidas`
        return _invalido(getattr(exc, "detail", str(exc)), codigo="tamanho_invalido")

    if base is not None and base.size != (largura, altura):
        from PIL import Image

        base = base.resize((largura, altura), Image.LANCZOS)
        mascara = mascara.resize((largura, altura), Image.NEAREST)

    imagem._avisar_de_prompt_em_portugues(prompt)
    perfil = imagem.perfil_referencia() if referencias else imagem.perfil_principal()

    try:
        with ARBITRO.usar("imagem", modelo=perfil.modelo):
            if OLLAMA_DESCARREGAR_PARA_IMAGEM:
                ollama.descarregar_tudo()
            dispositivo = imagem.resolver_dispositivo()
            try:
                imagens = imagem._com_prazo_duro(
                    lambda: editar_imagens(
                        dispositivo,
                        prompt=prompt,
                        negativo=negative_prompt,
                        quantas=quantas,
                        largura=largura,
                        altura=altura,
                        semente=semente,
                        base=base,
                        mascara=mascara,
                        referencias=referencias,
                        forca=forca,
                    )
                )
            except WorkerTravado as exc:
                logger.critical("%s Encerrando o processo para o systemd subir outro.", exc)
                imagem._morrer_em_seguida()
                return respostas.falha_do_worker("worker_travado", str(exc))
            except imagem.TempoEsgotado as exc:
                logger.warning("%s", exc)
                return respostas.indisponivel("timeout", str(exc), retry_after=None)
            except Exception as exc:
                if not imagem._e_falta_de_vram(exc):
                    logger.exception("Falha ao editar a imagem")
                    return respostas.falha_do_worker(
                        "falha_na_edicao", f"{type(exc).__name__}: {exc}"
                    )
                logger.warning("VRAM insuficiente para editar (%s).", exc)
                imagem.descarregar()
                return respostas.indisponivel(
                    "sem_vram", "sem VRAM para editar agora. O trabalho volta para a fila."
                )
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    finally:
        imagem._agendar_descarga()

    _ultimo_perfil = perfil.nome
    logger.info(
        "%s imagem(ns) editada(s) %sx%s (%s, %s, %s referencia(s)%s).",
        len(imagens),
        largura,
        altura,
        dispositivo,
        perfil.modelo,
        len(referencias),
        ", com mascara" if mascara is not None else "",
    )
    corpo = {
        "created": int(time.time()),
        "data": [{"b64_json": base64.b64encode(png).decode("ascii")} for png in imagens],
    }
    if avisos:
        corpo["avisos"] = avisos
    return corpo


_ultimo_perfil = ""


def estado() -> dict:
    return {
        "referencias_maximas": IMAGEM_REFERENCIAS_MAXIMAS,
        "ultimo_perfil": _ultimo_perfil or None,
    }
