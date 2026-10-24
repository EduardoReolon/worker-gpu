#!/usr/bin/env bash
#
# Guarda o que e desta maquina e NAO esta no git: o que se perde ao formatar.
#
#   ./deploy/backup.sh                 grava em BACKUP_DESTINO (.env) ou na pasta pessoal
#   ./deploy/backup.sh /media/pendrive grava ali
#   ./deploy/backup.sh --agendar       toda semana, sozinho, em BACKUP_DESTINO
#   ./deploy/backup.sh --desagendar    para de fazer sozinho
#
# Entra no arquivo:
#   .env           o segredo do publi-bot, o BIND_HOST, o que esta ligado
#   voz/vozes/     as vozes (a referencia de cada uma; uma voz criada por
#                  descricao NAO se recria igual)
#   medicoes/      as afericoes de desempenho
#
# Nao entra: venvs e modelos (os instaladores refazem e baixam de novo).
# Guarda os 5 ultimos no destino e apaga os mais velhos.
#
# Para "nao esquecer", BACKUP_DESTINO deve ser FORA deste disco: uma pasta
# sincronizada com a nuvem (Google Drive, Dropbox, Nextcloud), um HD externo
# ou um compartilhamento de rede. No mesmo disco, a formatacao leva junto.
#
# Restaurar, depois de clonar o worker de novo:
#   tar xzf worker-gpu-backup-AAAA-MM-DD.tar.gz -C <pasta do worker>
#   ./deploy/instalar.sh

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_do_env() { grep -E "^${1}=" "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"'\r'; }
GUARDAR=5

UNIDADE="$HOME/.config/systemd/user"
if [[ "${1:-}" == "--agendar" ]]; then
    DESTINO="$(_do_env BACKUP_DESTINO || true)"
    if [[ -z "$DESTINO" ]]; then
        echo "Defina BACKUP_DESTINO no .env (uma pasta fora deste disco) antes de agendar."
        exit 1
    fi
    mkdir -p "$UNIDADE"
    cat > "$UNIDADE/worker-gpu-backup.service" <<EOF
[Unit]
Description=worker-gpu — backup do .env, das vozes e das medicoes

[Service]
Type=oneshot
ExecStart=$RAIZ/deploy/backup.sh
EOF
    cat > "$UNIDADE/worker-gpu-backup.timer" <<EOF
[Unit]
Description=worker-gpu — backup semanal

[Timer]
OnCalendar=weekly
# Computador desligado na hora: faz quando ligar.
Persistent=true
RandomizedDelaySec=1h

[Install]
WantedBy=timers.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now worker-gpu-backup.timer
    echo "Agendado: toda semana em $DESTINO."
    systemctl --user list-timers worker-gpu-backup.timer --no-pager | head -3
    exit 0
fi
if [[ "${1:-}" == "--desagendar" ]]; then
    systemctl --user disable --now worker-gpu-backup.timer 2>/dev/null || true
    rm -f "$UNIDADE/worker-gpu-backup.service" "$UNIDADE/worker-gpu-backup.timer"
    systemctl --user daemon-reload
    echo "Backup semanal desligado."
    exit 0
fi

DESTINO="${1:-$(_do_env BACKUP_DESTINO || true)}"
DESTINO="${DESTINO:-$HOME}"
DESTINO="${DESTINO/#\~/$HOME}"
mkdir -p "$DESTINO"

# As vozes podem morar fora do padrao (VOZ_VOZES_DIR / VOZ_DIR no .env).
VOZES="$(_do_env VOZ_VOZES_DIR || true)"
if [[ -z "$VOZES" ]]; then
    VOZ_DIR="$(_do_env VOZ_DIR || true)"
    VOZES="${VOZ_DIR:-$RAIZ/voz}/vozes"
fi
VOZES="${VOZES/#\~/$HOME}"
[[ "$VOZES" == /* ]] || VOZES="$RAIZ/$VOZES"

ITENS=()
[[ -f "$RAIZ/.env" ]] && ITENS+=(.env) || echo "  aviso: sem .env"
[[ -d "$RAIZ/medicoes" ]] && ITENS+=(medicoes)
FORA=()
if [[ -d "$VOZES" ]]; then
    case "$VOZES/" in
        "$RAIZ/"*) ITENS+=("${VOZES#"$RAIZ/"}") ;;
        *) FORA=(-C "$(dirname "$VOZES")" "$(basename "$VOZES")") ;;
    esac
else
    echo "  aviso: sem vozes em $VOZES"
fi
if [[ ${#ITENS[@]} -eq 0 && ${#FORA[@]} -eq 0 ]]; then
    echo "Nada para guardar."
    exit 1
fi

ARQUIVO="$DESTINO/worker-gpu-backup-$(date +%F-%H%M).tar.gz"
tar czf "$ARQUIVO" -C "$RAIZ" "${ITENS[@]}" "${FORA[@]}"
chmod 600 "$ARQUIVO"  # tem o segredo do .env
echo "Guardado: $ARQUIVO ($(du -h "$ARQUIVO" | cut -f1))"
tar tzf "$ARQUIVO" | grep -E '^(\.env|[^/]+/?$|.*/referencia\.wav)$' | sed 's/^/  /'
[[ ${#FORA[@]} -gt 0 ]] && echo "  (as vozes moram fora do worker: restaure $(basename "$VOZES")/ em $(dirname "$VOZES"))"

# Os 5 mais novos ficam.
ls -1t "$DESTINO"/worker-gpu-backup-*.tar.gz 2>/dev/null | tail -n +$((GUARDAR + 1)) | xargs -r rm -f

if [[ "$(stat -c %d "$DESTINO")" == "$(stat -c %d "$RAIZ")" ]]; then
    echo "Atencao: $DESTINO fica no mesmo disco do worker. Se nao for uma pasta"
    echo "sincronizada com a nuvem, copie o arquivo para fora antes de formatar."
fi
