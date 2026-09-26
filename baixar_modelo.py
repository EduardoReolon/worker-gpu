"""Baixa os pesos do modelo de imagem ANTES do primeiro pedido.

Existe por um defeito de experiencia, nao de codigo. O worker carrega o
modelo de forma preguicosa, no primeiro pedido — e o primeiro pedido de todos
nao carrega: **baixa** varios GB (7 no SDXL, ~20 no Z-Image). Quem clica em "gerar tres opcoes de
capa" na tela de revisao fica olhando o navegador girar por varios minutos,
sem nada no terminal do `manage.py dev` (o servico e uma unit do systemd: o
log dele esta no journal), e no fim leva um erro de tempo esgotado.

Rodar isto uma vez, na maquina da placa, tira o download do caminho da
requisicao:

    ./venv/bin/python baixar_modelo.py

Depois disso o primeiro pedido so LE do disco — dezenas de segundos, nao
minutos. Ele respeita o `IMAGEM_MODELO` do `.env`, entao trocar de modelo e
trocar a variavel e rodar isto de novo.

Nao carrega o modelo, nem na GPU nem na RAM: so popula o cache do HuggingFace.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _do_env() -> None:
    """Le o `.env` ao lado, para valer o mesmo modelo que a unit vai usar.

    Sem isto, rodar este script a mao baixaria o modelo padrao e o servico
    carregaria outro — e o download "ja feito" aconteceria de novo dentro da
    primeira requisicao, que e exatamente o que este arquivo existe para
    evitar.
    """
    arquivo = Path(__file__).resolve().parent / ".env"
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


def main() -> int:
    _do_env()

    modelo = os.environ.get("IMAGEM_MODELO", "stabilityai/stable-diffusion-xl-base-1.0")
    precisao = os.environ.get("IMAGEM_DTYPE", "float16").strip().lower()
    vae = os.environ.get("IMAGEM_VAE", "").strip()
    print(f"Baixando {modelo} ...")
    print("Sao alguns GB. Da para interromper e retomar: o download e incremental.")

    try:
        from diffusers import AutoencoderKL, DiffusionPipeline
    except ImportError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        print("  ./venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 1

    inicio = time.perf_counter()
    # `download` e nao `from_pretrained`: escolhe os MESMOS arquivos que o
    # pipeline pediria (variante, formato, componentes), sem carregar nada.
    # Carregar para descartar custava a RAM do modelo inteiro — no Z-Image em
    # float32, a tentativa sem variante, mais de 40 GB.
    #
    # A variante `fp16` so existe a parte nos repositorios SDXL, e o servico
    # so a pede em float16; em bfloat16 ele carrega o ramo principal.
    variantes = ["fp16", None] if precisao == "float16" else [None]
    for variante in variantes:
        try:
            DiffusionPipeline.download(modelo, variant=variante, use_safetensors=True)
            break
        except Exception as erro:
            if variante is None:
                print(f"ERRO ao baixar {modelo}: {erro}", file=sys.stderr)
                return 1
            # Nem todo repositorio publica a variante fp16.
            print("Sem variante fp16; baixando os pesos completos.")

    if vae:
        print(f"Baixando o VAE {vae} ...")
        try:
            # Pequeno (~300 MB): carregar aqui nao pesa, e garante os arquivos
            # que o servico vai pedir.
            AutoencoderKL.from_pretrained(vae)
        except Exception as erro:
            print(f"ERRO ao baixar o VAE {vae}: {erro}", file=sys.stderr)
            return 1

    ativa = os.environ.get("TRANSCRICAO_ATIVA", "sim").strip().lower()
    whisper = os.environ.get("TRANSCRICAO_MODELO", "large-v3").strip()
    if ativa in {"1", "true", "yes", "sim", "on"} and not os.path.isdir(whisper):
        # Mesmo motivo da imagem: sem isto, o primeiro audio baixa ~3 GB DENTRO
        # da requisicao.
        print(f"Baixando o Whisper {whisper} ...")
        try:
            from faster_whisper import download_model

            download_model(whisper)
        except Exception as erro:
            print(f"ERRO ao baixar o Whisper {whisper}: {erro}", file=sys.stderr)
            return 1

    print(f"Pronto em {time.perf_counter() - inicio:.0f}s. O cache esta em ~/.cache/huggingface.")
    print("A primeira geracao (e a primeira transcricao) agora so le do disco.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
