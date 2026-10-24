"""Texto para fala com o Kokoro, num processo a parte.

Roda no ambiente da voz (`voz/venv`), nao no do worker: o Kokoro pede torch e
um Python que nao precisam ser os daqui. O worker (`voz.py`) chama:

    voz/venv/bin/python falar_kokoro.py --voz pf_dora --saida fala.mp3 < texto.txt

O texto vem pela entrada padrao (sem limite de linha de comando nem aspas).
Grava MP3 (ou WAV, se o libsndfile nao tiver MP3) a 24 kHz e imprime o
formato em que gravou.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--voz", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--velocidade", type=float, default=1.0)
    args = p.parse_args()
    texto = sys.stdin.read().strip()
    if not texto:
        print("texto vazio", file=sys.stderr)
        return 2

    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    # A primeira letra da voz e o idioma: p = portugues do Brasil, a = ingles...
    pipeline = KPipeline(lang_code=args.voz[0], repo_id="hexgrad/Kokoro-82M")
    partes = [
        np.asarray(audio, dtype=np.float32)
        for _, _, audio in pipeline(texto, voice=args.voz, speed=args.velocidade)
        if audio is not None
    ]
    if not partes:
        print("o Kokoro nao gerou audio", file=sys.stderr)
        return 1
    som = np.concatenate(partes)
    try:
        sf.write(args.saida, som, 24000, format="MP3")
        print("FORMATO mp3", flush=True)
    except (sf.LibsndfileError, ValueError, TypeError):
        sf.write(args.saida, som, 24000, format="WAV")
        print("FORMATO wav", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
