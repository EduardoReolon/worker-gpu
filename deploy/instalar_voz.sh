#!/usr/bin/env bash
#
# Instala o Kokoro (voz: texto para fala) no ambiente proprio dele, em voz/.
#
#   ./deploy/instalar_voz.sh
#
# O `instalar.sh` chama este sozinho quando o `.env` tem VOZ_ATIVA=sim.
# Idempotente. Ambiente a parte pelo mesmo motivo da animacao: o worker so
# chama o Python DESTE ambiente (`falar_kokoro.py`), como processo separado.
# Torch de CPU: o Kokoro e pequeno (82M) e roda mais rapido que tempo real sem
# a placa — e assim narrar nunca disputa VRAM.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$(pwd)"

if [[ -f "$RAIZ/.env" ]]; then
    set -a; source <(grep -E '^[A-Z_]+=' "$RAIZ/.env"); set +a
fi
DESTINO="${VOZ_DIR:-$RAIZ/voz}"
DESTINO="${DESTINO/#\~/$HOME}"
VOZ="${VOZ_PADRAO:-pf_dora}"

echo "==> Kokoro em $DESTINO"

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# O Kokoro le portugues pelo espeak-ng (as regras de pronuncia).
if ! command -v espeak-ng >/dev/null; then
    echo "  instalando o espeak-ng (pronuncia do portugues)"
    sudo apt-get install -y -q espeak-ng
fi

mkdir -p "$DESTINO"
if [[ ! -x "$DESTINO/venv/bin/python" ]]; then
    uv venv --python 3.12 "$DESTINO/venv"
fi
PYTHON="$DESTINO/venv/bin/python"

echo "==> Dependencias (torch de CPU)"
uv pip install --python "$PYTHON" -q torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PYTHON" -q "kokoro>=0.9.4" soundfile "misaki[en]"

echo "==> Baixando o modelo e testando a voz $VOZ (~350 MB na primeira vez)"
echo "Teste de voz do worker." | "$PYTHON" "$RAIZ/falar_kokoro.py" \
    --voz "$VOZ" --saida "$DESTINO/teste.mp3"
ls -lh "$DESTINO/teste.mp3" | awk '{print "  " $5 "  " $9}'

echo
echo "Pronto. No .env do worker: VOZ_ATIVA=sim (e VOZ_PADRAO=$VOZ)."
echo "Ouca o teste: $DESTINO/teste.mp3"
