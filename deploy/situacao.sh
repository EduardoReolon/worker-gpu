#!/usr/bin/env bash
# O que o worker esta fazendo agora, numa tela so — feito para SSH.
#
#   ./deploy/situacao.sh          uma vez
#   watch -n 10 ./deploy/situacao.sh    atualizando a cada 10 s (Ctrl+C sai)
#
# Responde "esta processando ou travou?": quem segura a placa e ha quanto
# tempo, o progresso da animacao, se a GPU trabalha, e se o processo do
# FramePack esta vivo e gastando CPU.
set -uo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_do_env() { grep -E "^${1}=" "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r'; }
URL="http://$(_do_env BIND_HOST || true):$(_do_env BIND_PORT || true)"
[[ "$URL" == "http://:" ]] && URL="http://127.0.0.1:8090"

echo "== Worker ($URL)"
if SAUDE="$(curl -sf --max-time 5 "$URL/health/")"; then
    echo "$SAUDE" | jq -r '
      "placa      : " + (if .ocupada then "OCUPADA por \(.ocupante) (\(.modelo // "-")) ha \(.ha_segundos / 60 | floor) min" else "livre" end),
      "memoria    : \(.memoria.rss_mb // "?") MB (swap \(.memoria.swap_mb // "?") MB)",
      (if .animacao then
         (.animacao.em_curso[]? | "animacao   : \(.id) \(.status) \(.progress)% ha \(.ha_segundos / 60 | floor) min (\(.seconds) s de video)"),
         (if (.animacao.em_curso | length) == 0 then "animacao   : nenhuma em curso" else empty end)
       else empty end),
      "ollama     : \([.ollama.carregados_detalhe[]? | .name + " ctx=" + (.context_length|tostring)] | join(", ") | if . == "" then "nada carregado" else . end)"'
else
    echo "nao respondeu: $(systemctl --user is-active worker-gpu 2>/dev/null)"
fi

echo
echo "== GPU"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | sed 's/^/uso, VRAM usada, total, temp: /'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
        sed 's/^/  processo: /'
fi

echo
echo "== Processos pesados (CPU %, RAM, tempo)"
ps -eo pid,pcpu,rss,etime,args --sort=-pcpu |
    awk 'NR>1 && NR<=6 {printf "  %-7s %5s%% %6.1f GB  %10s  %s\n", $1, $2, $3/1048576, $4, substr($0, index($0,$5), 70)}'

echo
echo "== Ultimas linhas do journal"
journalctl --user -u worker-gpu -n 6 --no-pager -o cat 2>/dev/null | grep -v FutureWarning | sed 's/^/  /'
