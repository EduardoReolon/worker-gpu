#!/usr/bin/env bash
#
# Instala o worker como unit do systemd, nesta maquina.
#
#   ./deploy/instalar.sh              unit de USUARIO (systemctl --user)
#   ./deploy/instalar.sh --sistema    unit de SISTEMA (precisa de sudo)
#
# Unit de usuario e o padrao, e e o que faz sentido num computador pessoal:
# nao pede sudo e usa o venv que ja esta na sua pasta. Mas ela sobe no LOGIN,
# nao no boot. Para mante-la de pe com a maquina ligada e ninguem logado:
#
#     sudo loginctl enable-linger "$USER"
#
# Use `--sistema` numa maquina dedicada, que precisa subir no boot sempre.
#
# Idempotente: rodar de novo reescreve a unit e reinicia o servico.

set -euo pipefail

cd "$(dirname "$0")/.."
RAIZ="$(pwd)"

ESCOPO="usuario"
if [[ "${1:-}" == "--sistema" ]]; then
    ESCOPO="sistema"
elif [[ -n "${1:-}" ]]; then
    echo "ERRO: argumento desconhecido '${1}'. Use --sistema ou nenhum." >&2
    exit 1
fi

echo "==> Conferindo o que precisa existir"

if [[ ! -x "$RAIZ/venv/bin/uvicorn" ]]; then
    echo "ERRO: $RAIZ/venv/bin/uvicorn nao existe." >&2
    echo "  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -f "$RAIZ/.env" ]]; then
    echo "ERRO: $RAIZ/.env nao existe." >&2
    echo "  cp .env.example .env    e defina WORKER_SHARED_SECRET" >&2
    exit 1
fi

# Lido aqui, e nao no fim: as conferencias abaixo precisam dos valores, e
# conferir depois de instalar a unit ja e tarde.
set -a; source <(grep -E '^[A-Z_]+=' "$RAIZ/.env"); set +a
ENDERECO="${BIND_HOST:-127.0.0.1}"

# Sem o segredo o servico sobe e responde 500 a TODA chamada, com uma mensagem
# que fala de configuracao sem dizer qual arquivo preencher.
if ! grep -qE '^WORKER_SHARED_SECRET=.+' "$RAIZ/.env"; then
    echo "ERRO: WORKER_SHARED_SECRET esta vazio em $RAIZ/.env." >&2
    echo "  Gere um com:  python3 -c \"import secrets; print(secrets.token_urlsafe(48))\"" >&2
    echo "  O MESMO valor vai na configuracao de cada cliente." >&2
    exit 1
fi

# `BIND_HOST=0.0.0.0` publica na internet um endpoint que roda modelo na sua
# placa. Recusar aqui e mais barato que descobrir depois.
if grep -qE '^BIND_HOST=(0\.0\.0\.0|::)[[:space:]]*$' "$RAIZ/.env"; then
    echo "ERRO: BIND_HOST=0.0.0.0 em $RAIZ/.env." >&2
    echo "  Use 127.0.0.1 (so esta maquina) ou o endereco da Tailscale." >&2
    exit 1
fi

# O endereco precisa existir NESTA maquina. Tres coisas caem aqui: um valor de
# exemplo nunca substituido, o endereco da Tailscale com o tailscaled parado,
# e um IP que mudou de lugar. Sem esta conferencia, o uvicorn morre no boot, o
# systemd o reinicia a cada 10s, e a unica pista fica no journal.
PYTHON_DE_CONFERENCIA="$RAIZ/venv/bin/python"
[[ -x "$PYTHON_DE_CONFERENCIA" ]] || PYTHON_DE_CONFERENCIA="$(command -v python3 || true)"

if [[ -n "$PYTHON_DE_CONFERENCIA" ]]; then
    if ! FALHA="$("$PYTHON_DE_CONFERENCIA" -c '
import socket, sys

host = sys.argv[1]
familia = socket.AF_INET6 if ":" in host else socket.AF_INET
sonda = socket.socket(familia, socket.SOCK_STREAM)
try:
    sonda.bind((host, 0))
except OSError as erro:
    sys.exit(str(erro))
finally:
    sonda.close()
' "$ENDERECO" 2>&1)"; then
        echo "ERRO: esta maquina nao consegue escutar em BIND_HOST=$ENDERECO." >&2
        echo "  $FALHA" >&2
        echo >&2
        echo "  Edite BIND_HOST em $RAIZ/.env:" >&2
        echo "    127.0.0.1                 atende so esta maquina" >&2
        echo "    \$(tailscale ip -4)        atende os outros pela Tailscale" >&2
        echo >&2
        echo "  Se ja era o endereco da Tailscale:  tailscale status" >&2
        exit 1
    fi
fi

# A unit passa `--port ${BIND_PORT}` ao uvicorn, e o systemd troca uma
# variavel AUSENTE por string vazia — sem reclamar. O uvicorn recebe
# `--port ""` e morre no boot, so no journal.
if [[ -z "${BIND_PORT:-}" ]]; then
    echo "ERRO: BIND_PORT nao esta definida em $RAIZ/.env." >&2
    echo "  A unit passaria um valor vazio ao uvicorn, que morre no boot." >&2
    echo "  Acrescente:  BIND_PORT=8090" >&2
    exit 1
fi

if [[ "$ESCOPO" == "sistema" ]]; then
    DESTINO="/etc/systemd/system/worker-gpu.service"
    SYSTEMCTL=(sudo systemctl)
    INSTALAR=(sudo install -m 0644)
    LINHA_DE_USUARIO="User=$(id -un)"
    ALVO="multi-user.target"
else
    DESTINO="$HOME/.config/systemd/user/worker-gpu.service"
    SYSTEMCTL=(systemctl --user)
    INSTALAR=(install -m 0644)
    # Unit de usuario ja roda como voce; `User=` ali e erro de carregamento.
    LINHA_DE_USUARIO="# (unit de usuario: roda como quem a iniciou)"
    ALVO="default.target"
    mkdir -p "$(dirname "$DESTINO")"
fi

echo "==> Gerando a unit ($ESCOPO)"
TEMPORARIO="$(mktemp)"
trap 'rm -f "$TEMPORARIO"' EXIT

# `|` como separador: os valores sao caminhos, e com `/` cada um precisaria de
# escape.
sed -e "s|RAIZ|$RAIZ|g" \
    -e "s|LINHA_DE_USUARIO|$LINHA_DE_USUARIO|" \
    -e "s|ALVO_DE_INSTALACAO|$ALVO|" \
    "$RAIZ/deploy/worker-gpu.service" > "$TEMPORARIO"

"${INSTALAR[@]}" "$TEMPORARIO" "$DESTINO"
echo "  $DESTINO"

echo "==> Habilitando e subindo"
"${SYSTEMCTL[@]}" daemon-reload
"${SYSTEMCTL[@]}" enable worker-gpu.service >/dev/null
"${SYSTEMCTL[@]}" restart worker-gpu.service

echo "==> Conferindo /health/"
URL="http://${ENDERECO}:${BIND_PORT}/health/"

# O worker carrega os modelos de forma preguicosa, entao ele responde antes de
# ter peso nenhum na memoria — subir rapido aqui nao diz nada sobre o primeiro
# pedido, que ainda pode baixar alguns GB.
for _ in $(seq 1 15); do
    if RESPOSTA="$(curl -sf "$URL" 2>/dev/null)"; then
        echo "  $RESPOSTA"
        echo
        echo "Pronto. Nos clientes:"
        echo "  URL base : http://${ENDERECO}:${BIND_PORT}"
        echo "  Segredo  : o mesmo WORKER_SHARED_SECRET daqui"
        echo
        echo "Veja INTEGRACAO.md para adaptar um cliente."
        exit 0
    fi
    # Uma unit que ja morreu nao vai responder daqui a 28 segundos.
    if [[ "$("${SYSTEMCTL[@]}" is-active worker-gpu.service 2>/dev/null)" == "failed" ]]; then
        break
    fi
    sleep 2
done

echo "ERRO: o worker nao respondeu em $URL." >&2
echo >&2
# O motivo aqui, e nao um comando para rodar depois: a conferencia existe
# justamente para pegar a falha.
echo "  Fim do journal:" >&2
if [[ "$ESCOPO" == "sistema" ]]; then
    sudo journalctl -u worker-gpu.service -n 30 --no-pager 2>&1 | sed 's/^/    /' || true
else
    journalctl --user -u worker-gpu.service -n 30 --no-pager 2>&1 | sed 's/^/    /' || true
fi
echo >&2
echo "  A unit ficou habilitada e o systemd vai reinicia-la a cada 10s." >&2
echo "  Para parar enquanto investiga:" >&2
echo "    ${SYSTEMCTL[*]} disable --now worker-gpu.service" >&2
exit 1
