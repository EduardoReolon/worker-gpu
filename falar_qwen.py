"""Texto para fala com o Qwen3-TTS, num processo a parte.

Roda no ambiente do Qwen (`voz/qwen/venv`), nao no do worker: o `qwen-tts`
prende uma versao exata do transformers que nao precisa ser a daqui. O worker
(`voz.py`) chama, com a placa na mao:

    python falar_qwen.py falar --voz-dir voz/vozes/carla --saida fala < texto.txt
    python falar_qwen.py desenhar --voz-dir voz/vozes/carla --descricao "..."
    python falar_qwen.py importar --voz-dir voz/vozes/eu --gravacao eu.wav [--texto "..."]

Uma voz e uma pasta com `referencia.wav` (de 3 a ~20 s de fala) e, de
preferencia, `referencia.txt` (o que se diz nela). Sem o texto, o clone usa so
a "impressao" da voz (`x_vector_only_mode`): funciona, mas imita menos.

- `falar`: le o texto da entrada padrao, fala na voz da pasta e grava MP3 (ou
  WAV, se o libsndfile nao tiver MP3). Imprime o formato em que gravou.
- `desenhar`: cria a voz a partir de uma descricao ("mulher de 30 anos, voz
  calorosa..."), pelo modelo VoiceDesign, e grava a referencia. E o caminho de
  ter uma voz propria sem gravar ninguem.
- `importar`: transforma uma gravacao sua em referencia (mono, ate 20 s).

O processo carrega o modelo a cada chamada e sai: a VRAM volta inteira ao
terminar, sem precisar de quem descarregue.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# O texto que a voz desenhada le para virar referencia: ~12 s, frases de
# entonacao variada (afirmacao, pergunta, exclamacao), sem nome proprio.
#
# Bem brasileiro de proposito ("a gente", "pra", "né", "tá"): o modelo so
# conhece "Portuguese", e um texto neutro puxa para o de Portugal. O jeito de
# falar da referencia e o que o clone copia depois.
TEXTO_DE_REFERENCIA = (
    "Oi, tudo bem? Que bom que você chegou! Hoje a gente vai falar de uma coisa "
    "simples, mas que faz muita diferença no dia a dia, né? Você já parou pra "
    "pensar nisso? Então presta atenção, que tá valendo a pena."
)
# Somado a descricao de toda voz desenhada. O instruct entende melhor ingles.
SOTAQUE_PADRAO = (
    "Brazilian Portuguese speaker with a natural accent from Brazil "
    "(not European Portuguese, not from Portugal)."
)
# Trecho por chamada ao modelo. Textos longos numa chamada so tendem a
# desandar no fim (voz que muda, palavra repetida); frases agrupadas ate aqui
# soam continuas e cada uma fica curta.
CARACTERES_POR_TRECHO = 300
PAUSA_ENTRE_TRECHOS = 0.25  # segundos
REFERENCIA_MAXIMA = 20.0  # segundos


def trechos(texto: str, limite: int = CARACTERES_POR_TRECHO) -> list[str]:
    """Frases agrupadas ate `limite` caracteres. Uma frase maior que o limite
    e cortada nas virgulas e, em ultimo caso, nos espacos."""
    frases = [f.strip() for f in re.split(r"(?<=[.!?…;])\s+|\n+", texto) if f.strip()]
    pedacos: list[str] = []
    for frase in frases:
        if len(frase) <= limite:
            pedacos.append(frase)
            continue
        atual = ""
        for parte in re.split(r"(?<=,)\s+|\s+", frase):
            if atual and len(atual) + 1 + len(parte) > limite:
                pedacos.append(atual)
                atual = parte
            else:
                atual = f"{atual} {parte}".strip()
        if atual:
            pedacos.append(atual)
    grupos: list[str] = []
    for pedaco in pedacos:
        if grupos and len(grupos[-1]) + 1 + len(pedaco) <= limite:
            grupos[-1] = f"{grupos[-1]} {pedaco}"
        else:
            grupos.append(pedaco)
    return grupos


def _modelo(nome: str, dispositivo: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    na_placa = dispositivo.startswith("cuda")
    return Qwen3TTSModel.from_pretrained(
        nome,
        device_map="cuda:0" if na_placa else "cpu",
        dtype=torch.bfloat16 if na_placa else torch.float32,
        # O FlashAttention reduz a VRAM, mas compilar o pacote e uma novela;
        # o SDPA do proprio torch chega perto e ja vem instalado.
        attn_implementation="sdpa",
    )


def _gravar(caminho: str, som, taxa: int) -> str:
    import soundfile as sf

    try:
        sf.write(caminho, som, taxa, format="MP3")
        return "mp3"
    except (sf.LibsndfileError, ValueError, TypeError):
        sf.write(caminho, som, taxa, format="WAV")
        return "wav"


def falar(args) -> int:
    import numpy as np

    texto = sys.stdin.read().strip()
    if not texto:
        print("texto vazio", file=sys.stderr)
        return 2
    pasta = Path(args.voz_dir)
    referencia = pasta / "referencia.wav"
    if not referencia.exists():
        print(f"a voz nao tem {referencia}", file=sys.stderr)
        return 2
    arquivo_do_texto = pasta / "referencia.txt"
    texto_da_referencia = (
        arquivo_do_texto.read_text(encoding="utf-8").strip() if arquivo_do_texto.exists() else ""
    )

    modelo = _modelo(args.modelo, args.dispositivo)
    prompt = modelo.create_voice_clone_prompt(
        ref_audio=str(referencia),
        ref_text=texto_da_referencia or None,
        x_vector_only_mode=not texto_da_referencia,
    )
    partes = []
    taxa = 24000
    for trecho in trechos(texto):
        wavs, taxa = modelo.generate_voice_clone(
            text=trecho, language=args.idioma, voice_clone_prompt=prompt
        )
        partes.append(np.asarray(wavs[0], dtype=np.float32))
        partes.append(np.zeros(int(taxa * PAUSA_ENTRE_TRECHOS), dtype=np.float32))
    if not partes:
        print("o Qwen nao gerou audio", file=sys.stderr)
        return 1
    som = np.concatenate(partes[:-1])  # sem a pausa depois do ultimo
    print(f"FORMATO {_gravar(args.saida, som, taxa)}", flush=True)
    return 0


def desenhar(args) -> int:
    import soundfile as sf

    pasta = Path(args.voz_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    modelo = _modelo(args.modelo, args.dispositivo)
    instrucao = f"{args.descricao.strip().rstrip('.')}. {args.sotaque}".strip()
    wavs, taxa = modelo.generate_voice_design(
        text=TEXTO_DE_REFERENCIA, language=args.idioma, instruct=instrucao
    )
    sf.write(pasta / "referencia.wav", wavs[0], taxa, format="WAV")
    (pasta / "referencia.txt").write_text(TEXTO_DE_REFERENCIA + "\n", encoding="utf-8")
    (pasta / "descricao.txt").write_text(args.descricao.strip() + "\n", encoding="utf-8")
    print("PRONTO", flush=True)
    return 0


def importar(args) -> int:
    import numpy as np
    import soundfile as sf

    try:
        som, taxa = sf.read(args.gravacao, dtype="float32", always_2d=True)
    except Exception as exc:  # o libsndfile diz pouco; o tipo do arquivo basta
        print(f"nao consegui ler a gravacao ({exc}); mande WAV, FLAC, OGG ou MP3", file=sys.stderr)
        return 2
    som = som.mean(axis=1)  # mono
    if len(som) < taxa * 3:
        print("a gravacao tem menos de 3 s; grave de 5 a 15 s de fala", file=sys.stderr)
        return 2
    som = som[: int(taxa * REFERENCIA_MAXIMA)]
    pico = float(np.abs(som).max()) or 1.0
    som = som * (0.9 / pico) if pico > 0.9 or pico < 0.3 else som
    pasta = Path(args.voz_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    sf.write(pasta / "referencia.wav", som, taxa, format="WAV")
    texto = (args.texto or "").strip()
    transcricao = pasta / "referencia.txt"
    if texto:
        transcricao.write_text(texto + "\n", encoding="utf-8")
    elif transcricao.exists():
        transcricao.unlink()
    print("PRONTO", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="modo", required=True)
    for nome in ("falar", "desenhar", "importar"):
        s = sub.add_parser(nome)
        s.add_argument("--voz-dir", required=True)
        s.add_argument("--modelo", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        s.add_argument("--dispositivo", default="cuda")
        s.add_argument("--idioma", default="Portuguese")
        if nome == "falar":
            s.add_argument("--saida", required=True)
        if nome == "desenhar":
            s.add_argument("--descricao", required=True)
            s.add_argument("--sotaque", default=SOTAQUE_PADRAO)
        if nome == "importar":
            s.add_argument("--gravacao", required=True)
            s.add_argument("--texto", default="")
    args = p.parse_args()
    return {"falar": falar, "desenhar": desenhar, "importar": importar}[args.modo](args)


if __name__ == "__main__":
    sys.exit(main())
