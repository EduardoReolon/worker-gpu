#!/usr/bin/env bash
#
# Instala o FramePack (animacao de imagem para video) no ambiente proprio dele.
#
#   ./deploy/instalar_animacao.sh            instala ou confere
#   ./deploy/instalar_animacao.sh --modelos  e ja baixa os modelos (~30 GB)
#
# O `instalar.sh` chama este sozinho quando o `.env` tem ANIMACAO_ATIVA=sim:
# numa maquina formatada, clonar o worker, preencher o `.env` e rodar o
# `instalar.sh` refaz tudo.
#
# Por que um ambiente a parte (e nao o venv do worker): o FramePack pede
# torch 2.6 e Python 3.10, e as versoes daqui sao outras. O worker so chama o
# Python DESTE ambiente (`animar_framepack.py`), como processo separado.
#
# Idempotente: o que ja existe e conferido, nao refeito. Versoes fixas — o
# commit do FramePack e o torch — para uma reinstalacao daqui a meses instalar
# o mesmo que foi medido.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$(pwd)"

# O commit medido (out/2025). Trocar de commit = medir de novo.
FRAMEPACK_COMMIT="97fe5dbe06ac1f337ece08935b1076a35eefeeb9"
FRAMEPACK_REPO="https://github.com/lllyasviel/FramePack"
# Mais novo que isto, o torchvision tirou o `write_video` que o FramePack usa
# para salvar o MP4 (o erro so aparece no fim da geracao, depois de 20 min).
TORCH="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
TORCH_INDICE="https://download.pytorch.org/whl/cu126"

if [[ -f "$RAIZ/.env" ]]; then
    set -a; source <(grep -E '^[A-Z_]+=' "$RAIZ/.env"); set +a
fi
DESTINO="${FRAMEPACK_DIR:-$HOME/codes/FramePack}"
DESTINO="${DESTINO/#\~/$HOME}"

echo "==> FramePack em $DESTINO"

if ! command -v uv >/dev/null; then
    echo "  instalando o uv (ele baixa o Python 3.10 sem mexer no do sistema)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d "$DESTINO/.git" ]]; then
    git clone "$FRAMEPACK_REPO" "$DESTINO"
fi
git -C "$DESTINO" fetch -q origin
git -C "$DESTINO" checkout -q "$FRAMEPACK_COMMIT"
echo "  commit $FRAMEPACK_COMMIT"

if [[ ! -x "$DESTINO/venv/bin/python" ]]; then
    uv venv --python 3.10 "$DESTINO/venv"
fi
PYTHON="$DESTINO/venv/bin/python"

echo "==> Dependencias (torch fixo; o resto pelo requirements do FramePack)"
# shellcheck disable=SC2086
uv pip install --python "$PYTHON" -q $TORCH --index-url "$TORCH_INDICE"
uv pip install --python "$PYTHON" -q -r "$DESTINO/requirements.txt"
# O requirements nao fixa o torch: reafirmado depois, para nada o ter subido.
# shellcheck disable=SC2086
uv pip install --python "$PYTHON" -q $TORCH --index-url "$TORCH_INDICE"

"$PYTHON" - <<'PY'
import torch, torchvision
assert hasattr(torchvision.io, "write_video"), "torchvision sem write_video: versao errada"
print(f"  torch {torch.__version__}, torchvision {torchvision.__version__}, "
      f"CUDA {'sim' if torch.cuda.is_available() else 'NAO'}")
PY

if [[ "${1:-}" == "--modelos" ]]; then
    echo "==> Baixando os modelos para $DESTINO/hf_download (~30 GB, uma vez so)"
    HF_HOME="$DESTINO/hf_download" "$PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("hunyuanvideo-community/HunyuanVideo", "lllyasviel/flux_redux_bfl",
             "lllyasviel/FramePackI2V_HY"):
    print(f"  {repo}")
    snapshot_download(repo)
PY
else
    echo "  Modelos: baixam no primeiro pedido (~30 GB), ou agora com --modelos."
fi

echo
echo "Pronto. No .env do worker: ANIMACAO_ATIVA=sim (FRAMEPACK_DIR=$DESTINO)."
