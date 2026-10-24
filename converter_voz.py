"""Conversao de voz com o Chatterbox, num processo a parte.

Roda no ambiente da conversao (`voz/conversao/venv`), nao no do worker: o
`chatterbox-tts` prende torch e transformers em versoes exatas. O worker
(`voz.py`) chama, com a placa na mao:

    python converter_voz.py --origem gravacao.mp3 --alvo voz/vozes/carla/referencia.wav \
        --saida convertida

Troca so o TIMBRE: as palavras, o ritmo, as pausas e a entonacao continuam os
da gravacao; a voz passa a ser a do `--alvo` (o Chatterbox usa os primeiros
10 s dele). Grava MP3 (ou WAV, se o libsndfile nao tiver MP3) e imprime o
formato. Sai com 2 para o que e erro do pedido (audio ilegivel, longo demais).

O processo carrega o modelo a cada chamada e sai: a VRAM volta inteira.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--origem", required=True)
    p.add_argument("--alvo", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--dispositivo", default="cuda")
    p.add_argument("--segundos-maximos", type=float, default=120)
    args = p.parse_args()

    import librosa
    import soundfile as sf

    try:
        duracao = librosa.get_duration(path=args.origem)
    except Exception as exc:  # o librosa diz pouco; o tipo do arquivo basta
        print(f"nao consegui ler o audio ({exc}); mande MP3, WAV, FLAC ou OGG", file=sys.stderr)
        return 2
    if duracao < 0.5:
        print("o audio tem menos de meio segundo", file=sys.stderr)
        return 2
    if duracao > args.segundos_maximos:
        print(
            f"o audio tem {duracao:.0f} s; o maximo e {args.segundos_maximos:.0f} s",
            file=sys.stderr,
        )
        return 2

    import perth

    if perth.PerthImplicitWatermarker is None:
        # O marcador d'agua do Chatterbox nao importou (falta o `pkg_resources`),
        # e o modelo quebraria com um TypeError que nao diz nada.
        print(
            "o marcador d'agua do Chatterbox nao carregou (falta o pkg_resources). Rode: "
            "./deploy/instalar_voz.sh --conversao",
            file=sys.stderr,
        )
        return 1

    from chatterbox.vc import ChatterboxVC

    modelo = ChatterboxVC.from_pretrained(args.dispositivo)
    som = modelo.generate(audio=args.origem, target_voice_path=args.alvo)
    som = som.squeeze(0).numpy()
    try:
        sf.write(args.saida, som, modelo.sr, format="MP3")
        print("FORMATO mp3", flush=True)
    except (sf.LibsndfileError, ValueError, TypeError):
        sf.write(args.saida, som, modelo.sr, format="WAV")
        print("FORMATO wav", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
