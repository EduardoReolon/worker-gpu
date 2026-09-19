"""Baixa os pesos do modelo de imagem ANTES do primeiro pedido.

Existe por um defeito de experiencia, nao de codigo. O worker carrega o
modelo de forma preguicosa, no primeiro pedido — e o primeiro pedido de todos
nao carrega: **baixa**, cerca de 7 GB. Quem clica em "gerar tres opcoes de
capa" na tela de revisao fica olhando o navegador girar por varios minutos,
sem nada no terminal do `manage.py dev` (o servico e uma unit do systemd: o
log dele esta no journal), e no fim leva um erro de tempo esgotado.

Rodar isto uma vez, na maquina da placa, tira o download do caminho da
requisicao:

    ./venv/bin/python baixar_modelo.py

Depois disso o primeiro pedido so LE do disco — dezenas de segundos, nao
minutos. Ele respeita o `IMAGEM_MODELO` do `.env`, entao trocar de modelo e
trocar a variavel e rodar isto de novo.

Nao carrega nada na GPU: so popula o cache do HuggingFace.
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

    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        os.environ.setdefault(chave.strip(), valor.strip().strip('"').strip("'"))


def main() -> int:
    _do_env()

    modelo = os.environ.get("IMAGEM_MODELO", "stabilityai/stable-diffusion-xl-base-1.0")
    print(f"Baixando {modelo} ...")
    print("Sao alguns GB. Da para interromper e retomar: o download e incremental.")

    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        print("  ./venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 1

    inicio = time.perf_counter()
    try:
        # Carrega para a RAM e descarta. Nao ha forma de "so baixar" que
        # garanta que o conjunto de arquivos e o mesmo que o pipeline pede:
        # a variante fp16, o tokenizer e o VAE sao escolhidos por esta chamada.
        AutoPipelineForText2Image.from_pretrained(
            modelo,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    except Exception:
        # Mesmo recurso do servico: nem todo repositorio publica a variante
        # fp16. Sem esta segunda tentativa, um modelo valido pareceria
        # inexistente.
        print("Sem variante fp16; baixando os pesos completos.")
        try:
            AutoPipelineForText2Image.from_pretrained(
                modelo, torch_dtype=torch.float32, use_safetensors=True
            )
        except Exception as erro:
            print(f"ERRO ao baixar {modelo}: {erro}", file=sys.stderr)
            return 1

    print(f"Pronto em {time.perf_counter() - inicio:.0f}s. O cache esta em ~/.cache/huggingface.")
    print("A primeira geracao agora so le do disco.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
