#!/usr/bin/env bash
#
# Instala o Kokoro (voz: texto para fala) no ambiente proprio dele, em voz/.
#
#   ./deploy/instalar_voz.sh           o Kokoro
#   ./deploy/instalar_voz.sh --qwen    o Kokoro e o Qwen3-TTS (ou VOZ_QWEN=sim no .env)
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

# ---------------------------------------------------------------------------
# Qwen3-TTS (opcional): VOZ_QWEN=sim no .env, ou ./deploy/instalar_voz.sh --qwen
# ---------------------------------------------------------------------------
# Ambiente proprio de novo: o qwen-tts prende o transformers numa versao exata.
# Torch com CUDA (o do PyPI), porque o Qwen fala na placa. Nao testa a voz aqui:
# testar tomaria a placa por fora do arbitro, com o worker possivelmente
# animando. O teste e criar a primeira voz (./deploy/criar_voz.sh), que passa
# pelo worker e espera a vez.
QWEN_LIGADO="${VOZ_QWEN:-nao}"
[[ "${1:-}" == "--qwen" ]] && QWEN_LIGADO=sim
case "${QWEN_LIGADO,,}" in
    1|true|sim|yes|on) ;;
    *) exit 0 ;;
esac

QWEN_DIR="${VOZ_QWEN_DIR:-$DESTINO/qwen}"
QWEN_DIR="${QWEN_DIR/#\~/$HOME}"
echo
echo "==> Qwen3-TTS em $QWEN_DIR"
if ! command -v sox >/dev/null; then
    echo "  instalando o sox (o qwen-tts le e corta audio com ele)"
    sudo apt-get install -y -q sox libsox-fmt-all
fi
mkdir -p "$QWEN_DIR" "${VOZ_VOZES_DIR:-$DESTINO/vozes}"
if [[ ! -x "$QWEN_DIR/venv/bin/python" ]]; then
    uv venv --python 3.12 "$QWEN_DIR/venv"
fi
QPY="$QWEN_DIR/venv/bin/python"
uv pip install --python "$QPY" -q torch torchaudio
uv pip install --python "$QPY" -q "qwen-tts>=0.1.1" soundfile

MODELO="${VOZ_QWEN_MODELO:-Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
DESENHO="${VOZ_QWEN_MODELO_DESENHO:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
echo "==> Baixando $MODELO e $DESENHO (~9 GB na primeira vez; so disco, sem a placa)"
"$QPY" - "$MODELO" "$DESENHO" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print("  ", snapshot_download(repo))
PY
"$QPY" -c "import qwen_tts, torch; print('  qwen-tts ok; CUDA:', torch.cuda.is_available())"

echo
echo "Pronto. No .env do worker: VOZ_QWEN=sim, e reinicie o worker."
echo "Depois crie a primeira voz:  ./deploy/criar_voz.sh carla \"mulher brasileira de 30 anos, voz calorosa\""
