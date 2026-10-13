"""Anima uma imagem com o FramePack, sem a pagina (gradio).

Roda no ambiente do FramePack, NAO no do worker: o FramePack pede torch 2.6 e
Python 3.10, que nao convivem com as versoes daqui. O worker (`animacao.py`)
toma a placa no arbitro e chama este arquivo como processo separado:

    <FRAMEPACK_DIR>/venv/bin/python animar_framepack.py \\
        --framepack-dir ~/codes/FramePack --imagem cena.png \\
        --prompt "The man waves." --segundos 3 --saida clipe.mp4

E o `worker()` do `demo_gradio.py` do FramePack (lllyasviel/FramePack), com os
mesmos padroes da pagina, menos a fila do gradio. Imprime `PROGRESSO <0-100>`
a cada passo, que o worker le para o `progress` do trabalho.

Os modelos ficam em `<FRAMEPACK_DIR>/hf_download` — os mesmos que a pagina
baixou: nada e baixado de novo.
"""

from __future__ import annotations

import argparse
import os
import sys


def _argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--framepack-dir", required=True)
    p.add_argument("--imagem", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--segundos", type=float, default=3.0)
    p.add_argument("--saida", required=True)
    p.add_argument("--resolucao", type=int, default=640)
    p.add_argument("--passos", type=int, default=25)
    p.add_argument("--memoria-preservada", type=float, default=6.0)
    p.add_argument("--teacache", action="store_true")
    p.add_argument("--semente", type=int, default=31337)
    return p.parse_args()


def _progresso(porcento: float) -> None:
    print(f"PROGRESSO {porcento:.0f}", flush=True)


def _medidas(altura: int, largura: int, resolucao: int) -> tuple[int, int]:
    """O formato da grade do FramePack (feita para 640) mais perto da imagem,
    na escala pedida, em multiplos de 32."""
    from diffusers_helper.bucket_tools import find_nearest_bucket

    alto, largo = find_nearest_bucket(altura, largura, resolution=640)
    escala = resolucao / 640
    return max(256, round(alto * escala / 32) * 32), max(256, round(largo * escala / 32) * 32)


def main() -> int:
    args = _argumentos()
    raiz = os.path.abspath(os.path.expanduser(args.framepack_dir))
    os.environ["HF_HOME"] = os.path.join(raiz, "hf_download")
    sys.path.insert(0, raiz)
    os.chdir(raiz)

    import numpy as np
    import torch
    from diffusers import AutoencoderKLHunyuanVideo
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode
    from diffusers_helper.memory import (
        DynamicSwapInstaller,
        fake_diffusers_current_device,
        gpu,
        load_model_as_complete,
        move_model_to_device_with_memory_preservation,
        offload_model_from_device_for_memory_preservation,
        unload_complete_models,
    )
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.utils import (
        crop_or_pad_yield_mask,
        resize_and_center_crop,
        save_bcthw_as_mp4,
        soft_append_bcthw,
    )
    from PIL import Image
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
        LlamaModel,
        LlamaTokenizerFast,
        SiglipImageProcessor,
        SiglipVisionModel,
    )

    hv = "hunyuanvideo-community/HunyuanVideo"
    f16 = torch.float16
    text_encoder = LlamaModel.from_pretrained(hv, subfolder="text_encoder", torch_dtype=f16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        hv, subfolder="text_encoder_2", torch_dtype=f16
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(hv, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(hv, subfolder="tokenizer_2")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(hv, subfolder="vae", torch_dtype=f16).cpu()
    redux = "lllyasviel/flux_redux_bfl"
    feature_extractor = SiglipImageProcessor.from_pretrained(redux, subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained(
        redux, subfolder="image_encoder", torch_dtype=f16
    ).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
    ).cpu()
    for modelo in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        modelo.eval()
        modelo.requires_grad_(False)
    vae.enable_slicing()
    vae.enable_tiling()
    transformer.high_quality_fp32_output_for_inference = True
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    _progresso(2)

    janela, passos, cfg, gs, rs = 9, args.passos, 1.0, 10.0, 0.0
    secoes = int(max(round((args.segundos * 30) / (janela * 4)), 1))

    with torch.no_grad():
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)
        llama_vec, clip_pooler = encode_prompt_conds(
            args.prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        llama_vec_n, clip_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_pooler)
        llama_vec, llama_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        imagem = np.array(Image.open(args.imagem).convert("RGB"))
        altura, largura = _medidas(*imagem.shape[:2], args.resolucao)
        imagem = resize_and_center_crop(imagem, target_width=largura, target_height=altura)
        imagem_pt = torch.from_numpy(imagem).float() / 127.5 - 1
        imagem_pt = imagem_pt.permute(2, 0, 1)[None, :, None]

        load_model_as_complete(vae, target_device=gpu)
        inicio_latente = vae_encode(imagem_pt, vae)
        load_model_as_complete(image_encoder, target_device=gpu)
        visao = hf_clip_vision_encode(imagem, feature_extractor, image_encoder).last_hidden_state

        dt = transformer.dtype
        llama_vec, llama_vec_n = llama_vec.to(dt), llama_vec_n.to(dt)
        clip_pooler, clip_pooler_n = clip_pooler.to(dt), clip_pooler_n.to(dt)
        visao = visao.to(dt)
        _progresso(5)

        rnd = torch.Generator("cpu").manual_seed(args.semente)
        historico = torch.zeros(
            size=(1, 16, 1 + 2 + 16, altura // 8, largura // 8), dtype=torch.float32
        ).cpu()
        pixels = None
        gerados = 0
        paddings = list(reversed(range(secoes)))
        if secoes > 4:
            paddings = [3] + [2] * (secoes - 3) + [1, 0]

        for ordem, padding in enumerate(paddings):
            ultima = padding == 0
            tamanho = padding * janela
            indices = torch.arange(0, sum([1, tamanho, janela, 1, 2, 16])).unsqueeze(0)
            pre, _vazio, latentes_i, pos, idx_2x, idx_4x = indices.split(
                [1, tamanho, janela, 1, 2, 16], dim=1
            )
            limpos_i = torch.cat([pre, pos], dim=1)
            limpos_pre = inicio_latente.to(historico)
            recentes = historico[:, :, : 1 + 2 + 16]
            limpos_pos, limpos_2x, limpos_4x = recentes.split([1, 2, 16], dim=2)
            limpos = torch.cat([limpos_pre, limpos_pos], dim=2)

            unload_complete_models()
            move_model_to_device_with_memory_preservation(
                transformer, target_device=gpu, preserved_memory_gb=args.memoria_preservada
            )
            transformer.initialize_teacache(enable_teacache=args.teacache, num_steps=passos)

            def callback(d, ordem=ordem):
                feito = (ordem + (d["i"] + 1) / passos) / len(paddings)
                _progresso(5 + 90 * feito)

            novos = sample_hunyuan(
                transformer=transformer, sampler="unipc", width=largura, height=altura,
                frames=janela * 4 - 3, real_guidance_scale=cfg, distilled_guidance_scale=gs,
                guidance_rescale=rs, num_inference_steps=passos, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_mask,
                prompt_poolers=clip_pooler, negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_mask_n, negative_prompt_poolers=clip_pooler_n,
                device=gpu, dtype=torch.bfloat16, image_embeddings=visao,
                latent_indices=latentes_i, clean_latents=limpos, clean_latent_indices=limpos_i,
                clean_latents_2x=limpos_2x, clean_latent_2x_indices=idx_2x,
                clean_latents_4x=limpos_4x, clean_latent_4x_indices=idx_4x, callback=callback,
            )  # fmt: skip
            if ultima:
                novos = torch.cat([inicio_latente.to(novos), novos], dim=2)
            gerados += int(novos.shape[2])
            historico = torch.cat([novos.to(historico), historico], dim=2)

            offload_model_from_device_for_memory_preservation(
                transformer, target_device=gpu, preserved_memory_gb=8
            )
            load_model_as_complete(vae, target_device=gpu)
            reais = historico[:, :, :gerados]
            if pixels is None:
                pixels = vae_decode(reais, vae).cpu()
            else:
                quadros_da_secao = janela * 2 + 1 if ultima else janela * 2
                atuais = vae_decode(reais[:, :, :quadros_da_secao], vae).cpu()
                pixels = soft_append_bcthw(atuais, pixels, janela * 4 - 3)
            unload_complete_models()
            if ultima:
                break

        save_bcthw_as_mp4(pixels, os.path.abspath(args.saida), fps=30, crf=18)
    _progresso(100)
    print(f"PRONTO {pixels.shape[2]} quadros {largura}x{altura}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
