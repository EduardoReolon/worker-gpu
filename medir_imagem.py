"""Mede quanto custa gerar uma imagem NESTA maquina, antes de decidir.

Irmao do `medir.py`, que faz o mesmo para a conversao de PDF. Existe porque as
duas perguntas que importam aqui nao tem resposta generica:

- **da para gerar em CPU?** Depende do modelo, do tamanho e da paciencia. Num
  SDXL a 1024x1024 a resposta costuma ser nao — e "nao" aqui significa horas,
  nao minutos;
- **quanto o tamanho economiza?** Nao e linear no numero de pixels: parte do
  custo e fixa (texto, VAE) e parte cresce com a area.

    ./venv/bin/python medir_imagem.py                    # o do .env, como esta
    ./venv/bin/python medir_imagem.py --cpu              # forca CPU
    ./venv/bin/python medir_imagem.py --tamanhos 512x512,768x432,1024x576
    ./venv/bin/python medir_imagem.py --modelo runwayml/stable-diffusion-v1-5
    ./venv/bin/python medir_imagem.py --passos 4 --guidance 0

**Ele mede a SEGUNDA geracao de cada combinacao**, nunca a primeira. O
pipeline so termina de montar dentro da primeira chamada, e incluir isso
misturaria "carregar o modelo" com "desenhar a imagem" — dois numeros com
consequencias diferentes: o primeiro se paga uma vez, o segundo se paga
sempre. (Foi exatamente esse engano que inflou a primeira medicao do Docling.)

Gera as imagens em `medicoes/`, para voce olhar se a qualidade compensa o
tempo. Numero sem imagem ao lado nao ajuda a escolher.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PROMPT_PADRAO = (
    "clean editorial photograph of a modern laboratory bench, soft daylight, "
    "shallow depth of field, muted colors"
)


def _do_env() -> None:
    """Os mesmos valores que a unit vai usar, para a medicao valer."""
    arquivo = RAIZ / ".env"
    if not arquivo.is_file():
        return
    # Lidas TODAS antes de aplicar, e a ULTIMA ocorrencia de cada chave vence.
    #
    # E o que o systemd faz com `EnvironmentFile`, e precisa ser igual: com
    # `setdefault` linha a linha a PRIMEIRA venceria, e um `.env` com a chave
    # repetida — que e o que acontece quando alguem faz `cat >> .env` — poria
    # esta ferramenta medindo uma configuracao e o servico rodando outra. Sem
    # erro nenhum, e com a ferramenta jurando que mediu o que esta em uso.
    do_arquivo: dict[str, str] = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        do_arquivo[chave.strip()] = valor.strip().strip('"').strip("'")

    # `setdefault` e nao atribuicao: uma variavel de verdade no ambiente manda
    # mais que o arquivo, que e como se passa `IMAGEM_DEVICE=cpu ./bancada.py`.
    for chave, valor in do_arquivo.items():
        os.environ.setdefault(chave, valor)


def _medidas(texto: str) -> tuple[int, int]:
    largura, _, altura = texto.lower().partition("x")
    return int(largura), int(altura)


def _montar(modelo: str, dispositivo: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    meia = dispositivo == "cuda"
    argumentos = {
        "torch_dtype": torch.float16 if meia else torch.float32,
        "use_safetensors": True,
    }
    if meia:
        argumentos["variant"] = "fp16"

    try:
        pipe = AutoPipelineForText2Image.from_pretrained(modelo, **argumentos)
    except Exception:
        if not meia:
            raise
        argumentos.pop("variant")
        pipe = AutoPipelineForText2Image.from_pretrained(modelo, **argumentos)

    if dispositivo == "cuda":
        # O MESMO arranjo do servico. Medir com `.to("cuda")` daria um numero
        # melhor e errado: seria a medicao de uma configuracao que o
        # o servico nao usa, porque ela nao cabe ao lado do Ollama.
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
    else:
        pipe.to("cpu")

    pipe.set_progress_bar_config(disable=True)
    return pipe


def _pico_de_vram() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "-"
        return f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    except Exception:
        return "-"


def main() -> int:
    _do_env()

    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--modelo", default=os.environ.get("IMAGEM_MODELO", ""))
    analisador.add_argument("--cpu", action="store_true", help="Forca CPU.")
    analisador.add_argument("--cuda", action="store_true", help="Forca a placa.")
    analisador.add_argument(
        "--tamanhos",
        default=os.environ.get("IMAGEM_TAMANHO", "1024x576"),
        help="Lista separada por virgula. Ex.: 512x512,768x432,1024x576",
    )
    analisador.add_argument("--passos", type=int, default=int(os.environ.get("IMAGEM_PASSOS", 25)))
    analisador.add_argument(
        "--guidance", type=float, default=float(os.environ.get("IMAGEM_GUIDANCE", 7.0))
    )
    analisador.add_argument("--prompt", default=PROMPT_PADRAO)
    opcoes = analisador.parse_args()

    modelo = opcoes.modelo or "stabilityai/stable-diffusion-xl-base-1.0"

    try:
        import torch
    except ImportError:
        print("ERRO: torch nao esta instalado neste venv.", file=sys.stderr)
        print("  ./venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 1

    if opcoes.cpu:
        dispositivo = "cpu"
    elif opcoes.cuda:
        dispositivo = "cuda"
    else:
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    if dispositivo == "cuda" and not torch.cuda.is_available():
        print("ERRO: --cuda pedido, mas o torch nao ve placa nenhuma.", file=sys.stderr)
        return 1

    print(f"Modelo     : {modelo}")
    print(f"Dispositivo: {dispositivo}")
    print(f"Passos     : {opcoes.passos}   guidance: {opcoes.guidance}")
    if dispositivo == "cpu":
        print("\nEm CPU isto pode levar MUITO tempo. Ctrl+C interrompe.")
    print()

    inicio = time.perf_counter()
    pipe = _montar(modelo, dispositivo)
    carga = time.perf_counter() - inicio
    print(f"Carga do modelo: {carga:.1f}s (uma vez por subida do servico)\n")

    saida = RAIZ / "medicoes"
    saida.mkdir(exist_ok=True)

    print(f"{'tamanho':>12}  {'1a':>8}  {'seguinte':>9}  {'pico VRAM':>10}")
    print("-" * 46)

    for texto in opcoes.tamanhos.split(","):
        largura, altura = _medidas(texto.strip())
        tempos = []

        for rodada in (1, 2):
            if dispositivo == "cuda":
                torch.cuda.reset_peak_memory_stats()
            comeco = time.perf_counter()
            resultado = pipe(
                prompt=opcoes.prompt,
                num_inference_steps=opcoes.passos,
                guidance_scale=opcoes.guidance,
                width=largura,
                height=altura,
            )
            tempos.append(time.perf_counter() - comeco)

            if rodada == 2:
                arquivo = saida / f"{largura}x{altura}-{dispositivo}.png"
                resultado.images[0].save(arquivo)

        print(f"{largura}x{altura:<6}  {tempos[0]:7.1f}s  {tempos[1]:8.1f}s  {_pico_de_vram():>10}")

    print(f"\nImagens em {saida}/ — olhe antes de escolher pelo tempo.")
    # Os dois valores moram em arquivos diferentes, e trocar no lugar errado
    # nao da erro nenhum — so nao muda nada. O TAMANHO viaja no pedido, entao
    # quem manda e o PubliBot; o MODELO e desta maquina.
    print(
        "\nPara usar o que voce escolher:\n"
        "    IMAGEM_TAMANHO=<o que ganhou>   no .env do PUBLIBOT (viaja no pedido)\n"
        "    IMAGEM_MODELO=<idem>            no .env do WORKER (e local)\n"
        "e reinicie:  systemctl --user restart worker-gpu"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
