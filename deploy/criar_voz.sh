#!/usr/bin/env bash
#
# Cria uma voz do Qwen3-TTS no worker e grava um teste para ouvir.
#
#   ./deploy/criar_voz.sh carla "mulher brasileira de 30 anos, voz calorosa e calma"
#   ./deploy/criar_voz.sh eduardo --gravacao minha_voz.wav "o que eu falo na gravacao"
#   ./deploy/criar_voz.sh carla "..." --substituir      troca uma que ja existe
#
# Pela descricao, o modelo VoiceDesign inventa a voz (nao imita ninguem). Pela
# gravacao, clona a sua: grave de 5 a 15 s, sem musica nem eco, e passe o texto
# exato do que disse (sem ele funciona, mas imita menos). So clone a voz de
# quem autorizou.
#
# Vai pelo worker, e nao direto no modelo: criar e falar usam a placa e passam
# pelo arbitro. Com uma animacao rodando, o script espera a vez.
# Depois, no publi-bot, a voz da narracao e qwen:<nome>.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_do_env() { grep -E "^${1}=" "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r'; }
URL="http://127.0.0.1:$(_do_env BIND_PORT || true)"
[[ "$URL" == "http://127.0.0.1:" ]] && URL="http://127.0.0.1:8090"
SEGREDO="$(_do_env WORKER_SHARED_SECRET || true)"
[[ -z "$SEGREDO" ]] && { echo "WORKER_SHARED_SECRET nao esta no .env"; exit 1; }

NOME="$1"; shift
CAMPOS=(-F "nome=$NOME")
if [[ "$1" == "--gravacao" ]]; then
    [[ -f "${2:-}" ]] || { echo "nao achei a gravacao: ${2:-}"; exit 1; }
    CAMPOS+=(-F "gravacao=@$2")
    [[ -n "${3:-}" && "$3" != "--substituir" ]] && CAMPOS+=(-F "texto=$3")
else
    CAMPOS+=(-F "descricao=$1")
fi
[[ " $* " == *" --substituir "* ]] && CAMPOS+=(-F "substituir=true")

# Pede, e com a placa ocupada espera o Retry-After e tenta de novo.
_pedir() {
    local saida="$1"; shift
    while true; do
        local cabecalho codigo espera
        cabecalho="$(mktemp)"
        codigo="$(curl -s -o "$saida" -D "$cabecalho" -w '%{http_code}' --max-time 600 \
            -H "Authorization: Bearer $SEGREDO" "$@" || true)"
        espera="$(grep -i '^retry-after:' "$cabecalho" | tr -dc '0-9' || true)"
        rm -f "$cabecalho"
        if [[ "$codigo" == "503" && -n "$espera" ]]; then
            echo "  placa ocupada ($(jq -r '.error.ocupante // .error.code' "$saida" 2>/dev/null)); tento de novo em ${espera}s" >&2
            sleep "$espera"
            continue
        fi
        echo "$codigo"
        return
    done
}

echo "==> Criando qwen:$NOME (a primeira vez carrega o modelo: ~1 min)"
RESPOSTA="$(mktemp)"
CODIGO="$(_pedir "$RESPOSTA" "${CAMPOS[@]}" "$URL/v1/audio/voices" | tail -1)"
if [[ "$CODIGO" != "201" ]]; then
    echo "  falhou ($CODIGO): $(cat "$RESPOSTA")"
    exit 1
fi
jq . "$RESPOSTA"

TESTE="$RAIZ/voz/teste-$NOME.mp3"
echo "==> Testando: $TESTE"
CODIGO="$(_pedir "$TESTE" -H "Content-Type: application/json" \
    -d "{\"voice\": \"qwen:$NOME\", \"input\": \"Você acorda cansado mesmo depois de dormir oito horas? Isso tem explicação, e a gente vai falar sobre ela hoje.\"}" \
    "$URL/v1/audio/speech" | tail -1)"
if [[ "$CODIGO" != "200" ]]; then
    echo "  a voz foi criada, mas o teste falhou ($CODIGO): $(cat "$TESTE")"
    exit 1
fi
ls -lh "$TESTE" | awk '{print "  " $5 "  " $9}'
echo
echo "Pronto. No publi-bot, a voz da narracao: qwen:$NOME"
