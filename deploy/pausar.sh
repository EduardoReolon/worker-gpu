#!/usr/bin/env bash
# Devolve a placa e a memoria para voce. Para jogar, editar video, ou so
# porque o computador esta seu.
#
#   ./deploy/pausar.sh            para o worker e solta o modelo do Ollama
#   ./deploy/pausar.sh --retomar  sobe tudo de volta
#
# Existe porque "parar o worker" nao e um comando, sao dois — e esquecer o
# segundo deixa 5 GB de VRAM presos no Ollama, que e justamente o que voce
# queria de volta. O `systemctl stop` sozinho nao toca no Ollama: eles sao
# servicos separados, e o worker so descarrega o Ollama quando esta com o lock
# na mao para gerar imagem.
#
# Nao desabilita nada: um reboot volta ao normal sozinho. E de proposito —
# uma pausa que sobrevive ao boot vira um worker desligado que ninguem lembra
# de ligar, e o sintoma aparece no cliente como 503 sem fim.
set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_URL="$(grep -E '^OLLAMA_URL=' "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- || true)"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

# O mesmo escopo que o `instalar.sh` usou. Na duvida, tenta os dois: um
# `systemctl --user` numa unit de sistema falha sem fazer nada.
if systemctl --user list-unit-files worker-gpu.service >/dev/null 2>&1 &&
    systemctl --user cat worker-gpu.service >/dev/null 2>&1; then
    SYSTEMCTL=(systemctl --user)
else
    SYSTEMCTL=(sudo systemctl)
fi

_vram() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null |
            sed 's/^/    VRAM em uso: /'
    fi
}

_modelos_carregados() {
    curl -s --max-time 5 "$OLLAMA_URL/api/ps" 2>/dev/null |
        python3 -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(m.get("name","") for m in d.get("models") or []))' 2>/dev/null
}

if [[ "${1:-}" == "--retomar" ]]; then
    echo "==> Subindo o worker"
    "${SYSTEMCTL[@]}" start worker-gpu.service
    sleep 3
    if [[ "$("${SYSTEMCTL[@]}" is-active worker-gpu.service 2>/dev/null)" == "active" ]]; then
        echo "    de pe. O primeiro pedido paga a carga do modelo, como sempre."
    else
        echo "ERRO: nao subiu. Veja o journal:" >&2
        echo "    ${SYSTEMCTL[*]} status worker-gpu.service" >&2
        exit 1
    fi
    exit 0
fi

if [[ -n "${1:-}" ]]; then
    echo "ERRO: argumento desconhecido '${1}'. Use --retomar ou nenhum." >&2
    exit 1
fi

echo "==> Parando o worker"
# O `TimeoutStopSec=90` da unit vale aqui: se uma difusao estiver no meio, o
# systemd espera ela terminar um passo antes de derrubar.
"${SYSTEMCTL[@]}" stop worker-gpu.service
echo "    parado."

echo "==> Soltando o modelo do Ollama"
CARREGADOS="$(_modelos_carregados)"
if [[ -z "$CARREGADOS" ]]; then
    echo "    nada carregado."
else
    for MODELO in $CARREGADOS; do
        # `keep_alive: 0` sem prompt: o proprio protocolo do Ollama para
        # "carregue nada e esqueca este modelo". Nao ha comando especial.
        curl -s --max-time 30 "$OLLAMA_URL/api/generate" \
            -d "{\"model\": \"$MODELO\", \"keep_alive\": 0}" >/dev/null 2>&1
        echo "    soltei $MODELO"
    done
fi

sleep 2
RESTOU="$(_modelos_carregados)"
if [[ -n "$RESTOU" ]]; then
    echo
    echo "AVISO: ainda carregado no Ollama: $RESTOU" >&2
    echo "  Alguem pediu texto entre o stop e agora, ou o Ollama recusou soltar." >&2
    echo "  Para garantir:  sudo systemctl stop ollama" >&2
fi

_vram
echo
echo "Pronto, a placa e sua. Para voltar:"
echo "    $0 --retomar"
