#!/usr/bin/env bash
#
# Instala o Kokoro (voz: texto para fala) no ambiente proprio dele, em voz/.
#
#   ./deploy/instalar_voz.sh           o Kokoro
#   ./deploy/instalar_voz.sh --qwen    o Kokoro e o Qwen3-TTS (ou VOZ_QWEN=sim no .env)
#   ./deploy/instalar_voz.sh --conversao   e a conversao de voz (ou VOZ_CONVERSAO=sim)
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
_ligado() { [[ "${1,,}" =~ ^(1|true|sim|yes|on)$ ]]; }
QWEN_LIGADO="${VOZ_QWEN:-nao}"
CONVERSAO_LIGADA="${VOZ_CONVERSAO:-nao}"
for argumento in "$@"; do
    [[ "$argumento" == "--qwen" ]] && QWEN_LIGADO=sim
    [[ "$argumento" == "--conversao" ]] && CONVERSAO_LIGADA=sim
done

if _ligado "$QWEN_LIGADO"; then
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
fi

# ---------------------------------------------------------------------------
# Conversao de voz (opcional): VOZ_CONVERSAO=sim, ou --conversao
# ---------------------------------------------------------------------------
# O conversor do Chatterbox (MIT): a gravacao de uma pessoa no timbre de uma
# voz qwen:<nome>. Ambiente proprio: o chatterbox-tts prende torch 2.6 e um
# transformers que nao sao os do Qwen. Baixa o modelo (~1 GB), sem a placa.
if _ligado "$CONVERSAO_LIGADA"; then
    CONV_DIR="${VOZ_CONVERSAO_DIR:-$DESTINO/conversao}"
    CONV_DIR="${CONV_DIR/#\~/$HOME}"
    echo
    echo "==> Conversao de voz (Chatterbox) em $CONV_DIR"
    mkdir -p "$CONV_DIR"
    if [[ ! -x "$CONV_DIR/venv/bin/python" ]]; then
        uv venv --python 3.12 "$CONV_DIR/venv"
    fi
    CPY="$CONV_DIR/venv/bin/python"
    # O `setuptools<81` e do marcador d'agua do Chatterbox (`resemble-perth`):
    # ele importa o `pkg_resources`, que o venv do uv nao traz e que o
    # setuptools 81 tirou. Sem ele o marcador vira None e o modelo nao carrega.
    uv pip install --python "$CPY" -q "chatterbox-tts>=0.1.7" soundfile "setuptools<81"
    "$CPY" - <<'PY'
from huggingface_hub import hf_hub_download
for arquivo in ("s3gen.safetensors", "conds.pt"):
    print("  ", hf_hub_download(repo_id="ResembleAI/chatterbox", filename=arquivo))
PY
    "$CPY" -c "import perth, torch; assert perth.PerthImplicitWatermarker, 'perth sem pkg_resources'; import chatterbox.vc; print('  chatterbox ok; CUDA:', torch.cuda.is_available())"
    echo
    echo "Pronto. No .env do worker: VOZ_CONVERSAO=sim, e reinicie o worker."
fi
