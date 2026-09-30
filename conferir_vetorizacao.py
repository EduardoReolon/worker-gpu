"""Teste de conformidade: o vetor do worker e o do servidor sao o mesmo?

O worker so INDEXA; a consulta continua no servidor do PubliBot (fastembed,
ONNX). Os dois vetores vao para o MESMO indice, e se divergirem a busca piora
sem erro nenhum. Este script manda a frase de referencia ao `/v1/embeddings`
do worker em execucao — o caminho de verdade, com o lock e a placa — e compara.

    ./venv/bin/python conferir_vetorizacao.py
    ./venv/bin/python conferir_vetorizacao.py --servidor vetor-do-servidor.json

Sem `--servidor`, mostra os 5 primeiros numeros (crus e normalizados) para
comparar a olho com o que o servidor imprime. Com `--servidor` (um JSON com a
lista de 1024 numeros, crua ou normalizada), calcula o cosseno: precisa dar
>= 0,999.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

FRASE = "passage: A curadoria garante que so entra no indice o que pode sustentar um artigo."
LIMIAR = 0.999


def _do_env() -> None:
    """Le o `.env` como o systemd le: a ULTIMA ocorrencia de cada chave vence,
    e uma variavel de verdade no ambiente manda mais que o arquivo."""
    arquivo = RAIZ / ".env"
    if not arquivo.is_file():
        return

    do_arquivo: dict[str, str] = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        do_arquivo[chave.strip()] = valor.strip().strip('"').strip("'")

    for chave, valor in do_arquivo.items():
        os.environ.setdefault(chave, valor)


def _normalizar(vetor: list[float]) -> list[float]:
    norma = math.sqrt(sum(x * x for x in vetor)) or 1.0
    return [x / norma for x in vetor]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--servidor", help="JSON com o vetor do servidor (lista de 1024 numeros)")
    parser.add_argument("--url", help="URL base do worker (padrao: BIND_HOST:BIND_PORT do .env)")
    argumentos = parser.parse_args()

    _do_env()
    import httpx

    url = argumentos.url or (
        f"http://{os.environ.get('BIND_HOST', '127.0.0.1')}:{os.environ.get('BIND_PORT', '8090')}"
    )
    modelo = os.environ.get("VETORIZACAO_MODELO", "intfloat/multilingual-e5-large")

    resposta = httpx.post(
        f"{url}/v1/embeddings",
        json={"model": modelo, "input": [FRASE]},
        headers={"Authorization": f"Bearer {os.environ.get('WORKER_SHARED_SECRET', '')}"},
        timeout=600,
    )
    if resposta.status_code != 200:
        print(f"O worker respondeu {resposta.status_code}: {resposta.text[:300]}", file=sys.stderr)
        if resposta.status_code == 503:
            print("Se for `modelo_carregando`, espere o download e rode de novo.", file=sys.stderr)
        return 1

    vetor = resposta.json()["data"][0]["embedding"]
    normalizado = _normalizar(vetor)
    print(f"Frase   : {FRASE}")
    print(f"Modelo  : {resposta.json()['model']}  ({len(vetor)} dimensoes)")
    print(f"Crus    : {[round(x, 6) for x in vetor[:5]]}")
    print(f"Normais : {[round(x, 6) for x in normalizado[:5]]}")

    if not argumentos.servidor:
        print(
            "\nCompare com os 5 primeiros do servidor (crus ou normalizados, conforme ele imprime)."
        )
        print("Para o cosseno exato: --servidor vetor.json")
        return 0

    do_servidor = json.loads(Path(argumentos.servidor).read_text(encoding="utf-8"))
    if len(do_servidor) != len(vetor):
        print(
            f"Dimensoes diferentes: servidor {len(do_servidor)}, worker {len(vetor)}.",
            file=sys.stderr,
        )
        return 1

    cosseno = sum(a * b for a, b in zip(normalizado, _normalizar(do_servidor), strict=True))
    veredito = (
        "CONFORME" if cosseno >= LIMIAR else "NAO CONFORME — nao ligue a vetorizacao no worker"
    )
    print(f"\nCosseno : {cosseno:.6f}  ({veredito}, limiar {LIMIAR})")
    return 0 if cosseno >= LIMIAR else 1


if __name__ == "__main__":
    raise SystemExit(main())
