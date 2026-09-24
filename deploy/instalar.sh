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

# Um `.env` com CRLF (copiado de um `.env.example` que o git converteu, ou
# salvo por um editor do Windows) poe um `\r` no fim de cada valor. O systemd
# nao o tira: o uvicorn recebe `--port "8090\r"` e morre no boot.
if grep -q $'\r' "$RAIZ/.env"; then
    echo "ERRO: $RAIZ/.env tem fim de linha CRLF (Windows)." >&2
    echo "  sed -i 's/\\r\$//' $RAIZ/.env" >&2
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

# Limites de memoria, do `.env`. Eles moram na unit e nao dao para vir por
# `${}`: o systemd nao expande variaveis de ambiente em `MemoryMax=`. Entao o
# instalador os substitui aqui, e o `.env` continua sendo o unico lugar que se
# edita.
#
# A ULTIMA ocorrencia vence, como o systemd faz com EnvironmentFile — um `.env`
# com a chave repetida (`cat >> .env`) tem que valer o mesmo nas duas leituras.
_do_env() {
    grep -E "^${1}=" "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- |
        tr -d '"'"'"' \t\r' || true
}

MEMORIA_FREIO="$(_do_env WORKER_MEMORY_HIGH)"
MEMORIA_PAREDE="$(_do_env WORKER_MEMORY_MAX)"
MEMORIA_FREIO="${MEMORIA_FREIO:-10G}"
MEMORIA_PAREDE="${MEMORIA_PAREDE:-14G}"

echo "==> Limites de memoria: freio $MEMORIA_FREIO, parede $MEMORIA_PAREDE"
echo "    (WORKER_MEMORY_HIGH / WORKER_MEMORY_MAX no .env)"
echo "    Acima do freio o kernel aperta; acima da parede ele mata, e o"
echo "    Restart=always sobe de novo em 10s. Veja 'Convivendo com a maquina'."

# Aviso de queda, tambem do `.env`. Antes ele se ligava descomentando uma linha
# no molde — que e arquivo versionado, entao o `git pull` seguinte recusava ou
# conflitava. A chave mora no `.env`, junto com o resto do que e desta maquina.
AVISO="$(_do_env WORKER_AVISO_QUEDA)"
AVISO="${AVISO:-nao}"
LINHA_DE_AVISO="# (aviso de queda desligado: WORKER_AVISO_QUEDA=sim no .env)"

if [[ "$AVISO" == "sim" ]]; then
    # O aviso e uma notificacao na SUA sessao grafica. Uma unit de sistema roda
    # fora dela, e o `notify-send` como root nao tem para quem mostrar — ligar
    # assim seria um aviso que nunca aparece, que e pior que nenhum.
    if [[ "$ESCOPO" == "sistema" ]]; then
        echo "ERRO: WORKER_AVISO_QUEDA=sim so vale para a unit de usuario." >&2
        echo "  Instale sem --sistema, ou ponha WORKER_AVISO_QUEDA=nao." >&2
        exit 1
    fi
    if ! command -v notify-send >/dev/null 2>&1; then
        echo "ERRO: WORKER_AVISO_QUEDA=sim, mas notify-send nao existe." >&2
        echo "  Debian/Ubuntu: sudo apt install libnotify-bin" >&2
        echo "  Fedora:        sudo dnf install libnotify" >&2
        exit 1
    fi
    install -m 0644 "$RAIZ/deploy/worker-gpu-aviso.service" \
        "$(dirname "$DESTINO")/worker-gpu-aviso.service"
    LINHA_DE_AVISO="OnFailure=worker-gpu-aviso.service"
    echo "==> Aviso de queda ligado"
    echo "    $(dirname "$DESTINO")/worker-gpu-aviso.service"
elif [[ "$AVISO" != "nao" ]]; then
    echo "ERRO: WORKER_AVISO_QUEDA='$AVISO' em $RAIZ/.env. Use sim ou nao." >&2
    exit 1
fi

echo "==> Gerando a unit ($ESCOPO)"
TEMPORARIO="$(mktemp)"
trap 'rm -f "$TEMPORARIO"' EXIT

# `|` como separador: os valores sao caminhos, e com `/` cada um precisaria de
# escape.
sed -e "s|RAIZ|$RAIZ|g" \
    -e "s|LINHA_DE_USUARIO|$LINHA_DE_USUARIO|" \
    -e "s|ALVO_DE_INSTALACAO|$ALVO|" \
    -e "s|LINHA_DE_AVISO|$LINHA_DE_AVISO|" \
    -e "s|MEMORIA_FREIO|$MEMORIA_FREIO|" \
    -e "s|MEMORIA_PAREDE|$MEMORIA_PAREDE|" \
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

# Quem esta na porta. O journal diz por que o worker nao subiu; isto diz se o
# motivo e que o lugar ja estava ocupado — e a sonda de endereco la em cima nao
# responde essa pergunta, porque ela faz `bind` na porta ZERO: confere o
# endereco, nunca a porta. Numa reinstalacao a porta e legitimamente nossa, e
# por isso a sonda pode continuar como esta; na FALHA, saber de quem ela e
# separa "meu processo antigo nao morreu" de "outro servico chegou primeiro".
echo "  Quem esta escutando na porta ${BIND_PORT}:" >&2
if command -v ss >/dev/null 2>&1; then
    ss -ltnp 2>/dev/null | grep ":${BIND_PORT}\b" | sed 's/^/    /' >&2 || echo "    (ninguem)" >&2
elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"${BIND_PORT}" -sTCP:LISTEN -P -n 2>/dev/null | sed 's/^/    /' >&2 || echo "    (ninguem)" >&2
else
    echo "    (nem ss nem lsof nesta maquina)" >&2
fi
echo >&2
echo "  A unit ficou habilitada e o systemd vai reinicia-la a cada 10s." >&2
echo "  Para parar enquanto investiga:" >&2
echo "    ${SYSTEMCTL[*]} disable --now worker-gpu.service" >&2
exit 1
