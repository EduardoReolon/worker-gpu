#!/usr/bin/env bash
#
# Monta a gravacao de referencia de uma voz a partir de varios audios curtos
# (as falas gravadas no publi-bot, por exemplo) e cria a voz no worker.
#
#   ./deploy/montar_referencia.sh eduardo ~/gravacoes
#   ./deploy/montar_referencia.sh eduardo ~/gravacoes --substituir
#
# 1. Mede o ruido de fundo de cada audio e fica com os mais limpos, ate ~15 s
#    (os bem mais ruidosos que o melhor ficam de fora, se der 8 s sem eles).
# 2. Atenua o ruido constante (chuva, motor, ventilador), corta o silencio
#    das pontas e iguala o volume entre eles.
# 3. Junta, com uma pausa curta entre um e outro.
# 4. Transcreve pelo Whisper do worker (o clone imita melhor com o texto).
# 5. Mostra o texto, grava voz/montagem/<nome>.wav e .txt, e cria a voz
#    (pelo ./deploy/criar_voz.sh) se voce confirmar.
#
# Aceita qualquer audio que o ffmpeg le (mp3, wav, webm, ogg, m4a...).
# Se o texto vier com erro, corrija o .txt e crie com o comando que ele mostra.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
NOME="$1"; PASTA="$2"; shift 2
EXTRA=("$@")
ALVO_SEGUNDOS=15
MAXIMO_SEGUNDOS=20

command -v ffmpeg >/dev/null || { echo "Falta o ffmpeg: sudo apt-get install -y ffmpeg"; exit 1; }
[[ -d "$PASTA" ]] || { echo "Nao achei a pasta $PASTA"; exit 1; }

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_do_env() { grep -E "^${1}=" "$RAIZ/.env" 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' \r'; }
HOST="$(_do_env BIND_HOST || true)"
[[ -z "$HOST" || "$HOST" == "0.0.0.0" ]] && HOST="127.0.0.1"
PORTA="$(_do_env BIND_PORT || true)"
URL="http://$HOST:${PORTA:-8090}"
SEGREDO="$(_do_env WORKER_SHARED_SECRET || true)"

TRABALHO="$(mktemp -d)"
trap 'rm -rf "$TRABALHO"' EXIT

# --- 1. Ruido e duracao de cada audio -----------------------------------------
echo "==> Medindo os audios de $PASTA"
LISTA="$TRABALHO/lista.tsv"
: > "$LISTA"
while IFS= read -r -d '' arquivo; do
    # O piso de ruido: quanto mais negativo, mais limpo o fundo.
    # O astats sem opcoes (as de filtrar a saida nao existem no ffmpeg 4); a
    # ultima linha de piso e a do conjunto. A duracao sai do que foi
    # decodificado ("time="): o webm do navegador nao traz "Duration".
    medida="$(ffmpeg -hide_banner -i "$arquivo" -af astats -f null - 2>&1 || true)"
    ruido="$(grep -oE 'Noise floor dB: -?[0-9.inf]+' <<<"$medida" | tail -1 | awk '{print $4}' || true)"
    duracao="$(grep -oE 'time=[0-9:.]+' <<<"$medida" | tail -1 | awk -F'[=:]' '{print $2*3600+$3*60+$4}' || true)"
    if [[ -z "$duracao" || -z "$ruido" ]] || awk "BEGIN {exit !($duracao < 0.3)}"; then
        echo "  ignorado (nao e audio): $(basename "$arquivo")"
        continue
    fi
    [[ "$ruido" == "-inf" ]] && ruido=-120
    printf '%s\t%s\t%s\n' "$ruido" "$duracao" "$arquivo" >> "$LISTA"
done < <(find "$PASTA" -maxdepth 1 -type f -print0 | sort -z)

[[ -s "$LISTA" ]] || { echo "Nenhum audio em $PASTA"; exit 1; }

# --- 2. Os mais limpos ate ~15 s (na ordem dos nomes, para soar continuo) ------
ESCOLHIDOS="$TRABALHO/escolhidos.tsv"
# Um audio bem mais ruidoso que o mais limpo (15 dB acima) so entra se, sem
# ele, nao der nem 8 s: chuva na referencia vira chuva em toda fala.
sort -t$'\t' -k1,1g "$LISTA" | awk -F'\t' -v alvo="$ALVO_SEGUNDOS" -v max="$MAXIMO_SEGUNDOS" '
    NR == 1 { limpo = $1 }
    total >= alvo { next }
    total + $2 > max { next }
    $1 > limpo + 15 && total >= 8 { next }
    { total += $2; print }
' | sort -t$'\t' -k3 > "$ESCOLHIDOS"

echo "    ruido(dB)  seg  arquivo   (* = escolhido)"
while IFS=$'\t' read -r ruido duracao arquivo; do
    marca=" "; grep -qF "$arquivo" "$ESCOLHIDOS" && marca="*"
    printf '  %s %8.1f %5.1f  %s\n' "$marca" "$ruido" "$duracao" "$(basename "$arquivo")"
done < <(sort -t$'\t' -k1,1g "$LISTA")

TOTAL="$(awk -F'\t' '{t += $2} END {printf "%.1f", t}' "$ESCOLHIDOS")"
if awk "BEGIN {exit !($TOTAL < 5)}"; then
    echo "So ${TOTAL} s de audio aproveitavel; grave mais (de 10 a 15 s no total)."
    exit 1
fi

# --- 3. Limpa cada um e junta -----------------------------------------------
echo "==> Limpando e juntando ${TOTAL} s"
LIMPEZA="highpass=f=80,afftdn=nr=12:nf=-30:tn=1,"
LIMPEZA+="silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.05,"
LIMPEZA+="areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,areverse,"
LIMPEZA+="loudnorm=I=-20:TP=-2:LRA=7"
ffmpeg -hide_banner -loglevel error -f lavfi -i "anullsrc=r=24000:cl=mono" -t 0.35 "$TRABALHO/pausa.wav"
LISTA_CONCAT="$TRABALHO/concat.txt"
: > "$LISTA_CONCAT"
i=0
while IFS=$'\t' read -r _ _ arquivo; do
    i=$((i + 1))
    ffmpeg -hide_banner -loglevel error -i "$arquivo" -af "$LIMPEZA" -ar 24000 -ac 1 \
        "$TRABALHO/parte$i.wav"
    [[ $i -gt 1 ]] && echo "file '$TRABALHO/pausa.wav'" >> "$LISTA_CONCAT"
    echo "file '$TRABALHO/parte$i.wav'" >> "$LISTA_CONCAT"
done < "$ESCOLHIDOS"

SAIDA_DIR="$RAIZ/voz/montagem"
mkdir -p "$SAIDA_DIR"
WAV="$SAIDA_DIR/$NOME.wav"
TXT="$SAIDA_DIR/$NOME.txt"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$LISTA_CONCAT" -c pcm_s16le "$WAV"
echo "    $WAV"

# --- 4. Transcreve pelo worker -------------------------------------------------
echo "==> Transcrevendo pelo worker ($URL)"
TEXTO=""
if [[ -n "$SEGREDO" ]]; then
    while true; do
        cabecalho="$TRABALHO/cabecalho"
        codigo="$(curl -s -o "$TRABALHO/texto" -D "$cabecalho" -w '%{http_code}' --max-time 300 \
            -H "Authorization: Bearer $SEGREDO" -F "file=@$WAV" -F "language=pt" \
            -F "response_format=text" "$URL/v1/audio/transcriptions" || true)"
        espera="$(grep -i '^retry-after:' "$cabecalho" 2>/dev/null | tr -dc '0-9' || true)"
        if [[ "$codigo" == "503" && -n "$espera" ]]; then
            echo "  placa ocupada; tento de novo em ${espera}s"
            sleep "$espera"
            continue
        fi
        break
    done
    if [[ "$codigo" == "200" ]]; then
        TEXTO="$(tr '\n' ' ' < "$TRABALHO/texto" | sed 's/  */ /g; s/^ //; s/ $//')"
    else
        echo "  nao transcreveu ($codigo): $(head -c 300 "$TRABALHO/texto" 2>/dev/null)"
    fi
fi
if [[ -z "$TEXTO" ]]; then
    echo "  Sem o texto. Escreva em $TXT o que se diz no audio, e crie com:"
    echo "  ./deploy/criar_voz.sh $NOME --gravacao $WAV \"\$(cat $TXT)\" ${EXTRA[*]}"
    exit 1
fi
printf '%s\n' "$TEXTO" > "$TXT"
echo
echo "  Texto: $TEXTO"
echo
echo "Ouca $WAV e confira o texto (palavra por palavra; corrija em $TXT se precisar)."

# --- 5. Cria a voz ------------------------------------------------------------
COMANDO=(./deploy/criar_voz.sh "$NOME" --gravacao "$WAV" "$TEXTO" "${EXTRA[@]}")
if [[ -t 0 ]]; then
    read -r -p "Criar qwen:$NOME agora com esse texto? [s/N] " resposta
    if [[ "${resposta,,}" == s* ]]; then
        cd "$RAIZ" && exec "${COMANDO[@]}"
    fi
fi
echo "Para criar depois (com o texto do .txt):"
echo "  ./deploy/criar_voz.sh $NOME --gravacao $WAV \"\$(cat $TXT)\" ${EXTRA[*]}"
