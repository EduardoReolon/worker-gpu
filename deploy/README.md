# Operar o worker: guia dos scripts

Tudo o que se roda à mão, de dentro da **pasta do worker**
(`cd ~/codes/python/worker-gpu`). Os detalhes de cada assunto estão no
`README.md` da raiz (operar a máquina), no `INTEGRACAO.md` (as rotas) e no
`GRAVAR_VOZ.md` (gravar uma voz).

Para os comandos com `curl`, defina antes o endereço e o segredo do worker:

```bash
W="http://$(grep ^BIND_HOST .env | cut -d= -f2):$(grep ^BIND_PORT .env | cut -d= -f2)"
S="$(grep ^WORKER_SHARED_SECRET .env | cut -d= -f2 | tr -d '"')"
```

## Os scripts

| Script | Para quê |
|---|---|
| `deploy/instalar.sh` | instala ou reinstala o serviço (e o que o `.env` liga) |
| `deploy/instalar_animacao.sh` | o FramePack (animação), à parte |
| `deploy/instalar_voz.sh` | o Kokoro, o Qwen3-TTS (`--qwen`) e a conversão de voz (`--conversao`) |
| `deploy/situacao.sh` | o que o worker está fazendo agora, ao vivo |
| `deploy/pausar.sh` | devolve a placa e a memória para você (jogar, editar vídeo) |
| `deploy/backup.sh` | guarda o `.env`, as vozes e as medições; agenda backup semanal |
| `deploy/criar_voz.sh` | cria uma voz do Qwen (por descrição ou por gravação) |
| `deploy/montar_referencia.sh` | junta vários áudios curtos numa referência limpa e cria a voz |
| `afericao.py` | mede a máquina durante uma animação e sugere o `.env` |
| `baixar_modelo.py` | baixa o modelo de imagem antes do primeiro pedido |
| `medir.py`, `medir_imagem.py`, `bancada.py`, `conferir_vetorizacao.py` | medições e comparações (veja o fim) |

---

## 1. Instalar do zero (ou depois de formatar)

### O que vem antes (o deploy não instala)

- driver da NVIDIA (`nvidia-smi` precisa funcionar);
- git, Python 3.12, `curl`, `jq`, `ffmpeg`:
  `sudo apt-get install -y git python3-venv curl jq ffmpeg`;
- Tailscale, logado (`tailscale ip -4` mostra o endereço);
- Ollama e os modelos de texto (`ollama pull ...`), com o override de
  `127.0.0.1` do `README.md` (seção **O Ollama**);
- subir sem login: `sudo loginctl enable-linger $USER`.

### O worker

```bash
git clone <repositório> ~/codes/python/worker-gpu
cd ~/codes/python/worker-gpu
git checkout claude-cli

python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

**Com backup** (recomendado): restaure o `.env`, as vozes e as medições.

```bash
tar xzf /caminho/worker-gpu-backup-AAAA-MM-DD-HHMM.tar.gz -C .
```

**Sem backup**: `cp .env.example .env` e preencha. O mínimo é:

- `WORKER_SHARED_SECRET`: o mesmo que está no publi-bot;
- `BIND_HOST`: o IP do Tailscale (`tailscale ip -4`);
- `BIND_PORT=8090`.

Depois:

```bash
./venv/bin/python baixar_modelo.py          # modelo de imagem (~7 a 20 GB)
./deploy/instalar.sh                        # o serviço, e o que o .env liga
./deploy/instalar_animacao.sh --modelos     # se ANIMACAO_ATIVA=sim: ~30 GB
```

O `instalar.sh` lê o `.env` e instala sozinho:

- a animação, com `ANIMACAO_ATIVA=sim` (os modelos dela só com `--modelos`,
  acima);
- o Kokoro, com `VOZ_ATIVA=sim`;
- o Qwen, com `VOZ_QWEN=sim`;
- a conversão de voz, com `VOZ_CONVERSAO=sim`.

Ele confere o `/health/` no fim. Se o IP do Tailscale mudou, atualize o
`BIND_HOST` e também o endereço do worker no publi-bot.

## 2. Atualizar

```bash
git pull origin claude-cli
systemctl --user restart worker-gpu
```

| Mudou o quê | Rode também |
|---|---|
| `BIND_HOST`, `BIND_PORT`, `WORKER_MEMORY_*`, `WORKER_AVISO_QUEDA` | `./deploy/instalar.sh` (essas entram na unit) |
| ligou `VOZ_QWEN` ou `VOZ_CONVERSAO` | `./deploy/instalar_voz.sh --qwen` / `--conversao` |
| ligou `ANIMACAO_ATIVA` | `./deploy/instalar_animacao.sh --modelos` |
| trocou `IMAGEM_MODELO` | `./venv/bin/python baixar_modelo.py` |
| `requirements.txt` | `./venv/bin/pip install -r requirements.txt` |

O `restart` volta na hora, mas o worker leva alguns segundos para responder:
um `curl` logo em seguida pode sair vazio.

## 3. O dia a dia

```bash
./deploy/situacao.sh            # ao vivo, a cada 1 s (Ctrl+C sai); "5" = a cada 5 s; "-1" = uma vez
./deploy/pausar.sh              # para o worker e solta a VRAM do Ollama
./deploy/pausar.sh --retomar    # volta
systemctl --user status worker-gpu --no-pager
journalctl --user -u worker-gpu -n 50 --no-pager      # o log
journalctl --user -u worker-gpu -f                    # o log, ao vivo
curl -s "$W/health/" | jq                             # o estado completo
```

O `pausar.sh` não sobrevive a um reboot, de propósito: o computador religado
volta a atender sozinho.

## 4. Backup e restauração

O que não está no git e se perde ao formatar: o `.env`, as vozes
(`voz/vozes/`) e as medições. Venvs e modelos, os instaladores refazem.

```bash
./deploy/backup.sh                     # em BACKUP_DESTINO (.env) ou na pasta pessoal
./deploy/backup.sh /media/pendrive     # num lugar escolhido
```

**Automático, toda semana.** Ponha `BACKUP_DESTINO=/pasta/fora/deste/disco`
no `.env` (uma pasta sincronizada com a nuvem, um HD externo ou a rede) e
rode:

```bash
./deploy/backup.sh --agendar           # liga (com o computador desligado, faz quando ligar)
./deploy/backup.sh --desagendar        # desliga
systemctl --user list-timers worker-gpu-backup.timer   # quando é o próximo
```

Ele guarda os 5 mais novos. Depois de criar ou trocar uma voz, rode um
`./deploy/backup.sh` na hora: uma voz criada por descrição não se recria
igual.

**Restaurar**, dentro da pasta do worker:

```bash
tar xzf worker-gpu-backup-AAAA-MM-DD-HHMM.tar.gz -C .
./deploy/instalar.sh
```

O arquivo tem o segredo do `.env`: não o deixe em pasta pública.

## 5. Vozes

Uma voz do Qwen é uma pasta em `voz/vozes/<nome>/` (`referencia.wav` e
`referencia.txt`). No publi-bot, ela se chama `qwen:<nome>`. As do Kokoro
(`pf_dora`, `pm_alex`, `pm_santa`) já vêm prontas. Como gravar bem:
`GRAVAR_VOZ.md`.

### Criar

Pela **descrição**: o modelo inventa a voz, que não imita ninguém.

```bash
./deploy/criar_voz.sh carla "mulher brasileira de 30 anos, voz calorosa, tom de vendedora animada"
```

Por uma **gravação sua** de 10 a 15 s, com o texto exato do que se diz nela:

```bash
./deploy/criar_voz.sh eduardo --gravacao minha_voz.wav "o texto exato que eu falei"
```

Por **vários áudios curtos** (ex.: as falas baixadas do publi-bot). O
script escolhe os mais limpos, tira ruído constante, junta e transcreve:

```bash
./deploy/montar_referencia.sh eduardo ~/gravacoes
```

Os três gravam um teste em `voz/teste-<nome>.mp3` para você ouvir. Com uma
animação rodando, eles esperam a placa sozinhos.

- **Trocar** uma voz: o mesmo comando com `--substituir` no fim. A antiga se
  perde; para guardá-la antes, `cp -r voz/vozes/carla voz/vozes/carla-antiga`.
- **Sotaque** (vozes por descrição): cada criação sai um pouco diferente.
  Se puxar para Portugal ou para o carioca, rode de novo ou descreva a
  pronúncia (veja `GRAVAR_VOZ.md`). Na conversão, o sotaque vem da sua
  gravação.
- **Apagar**: `rm -r voz/vozes/<nome>`.
- **Voz padrão** (pedido sem voz): `VOZ_PADRAO=qwen:carla` no `.env` e
  `systemctl --user restart worker-gpu`.

### Listar e ouvir

```bash
curl -s -H "Authorization: Bearer $S" "$W/v1/audio/voices" | jq -r '.data[].id'

curl -s -o fala.mp3 -H "Authorization: Bearer $S" -H "Content-Type: application/json" \
  -d '{"voice": "qwen:carla", "input": "Oi! Hoje tem condição especial."}' "$W/v1/audio/speech"
```

### Converter uma gravação para outra voz

Saem as suas palavras, a sua entonação e o seu sotaque, com o timbre da voz
pedida:

```bash
curl -s -o convertida.mp3 -w '%{http_code}\n' --max-time 600 -H "Authorization: Bearer $S" \
  -F "voice=qwen:carla" -F "audio=@fala.mp3" "$W/v1/audio/convert"
```

O publi-bot faz isso sozinho com a fala gravada de cada cena.

### Acabamento

Toda fala que sai do worker ganha um acabamento de locução (volume -16 LUFS,
voz "de rádio"), pelo `ffmpeg`. Para desligar, `VOZ_ACABAMENTO=nao` no `.env`
e reinicie. Para conferir:

```bash
curl -s "$W/health/" | jq .voz.acabamento
```

## 6. Medir e comparar

| Comando | Responde |
|---|---|
| `python3 afericao.py --minutos 10` | rode **com uma animação em curso**: GPU, disco, memória, progresso; sugere `WORKER_MEMORY_MAX` e gera um prompt para uma LLM. Pelo SSH, use `nohup ... &`. Sai em `medicoes/` |
| `./venv/bin/python medir_imagem.py` | quanto custa uma imagem nesta máquina (tamanhos, CPU ou GPU) |
| `./venv/bin/python bancada.py` | qual configuração de imagem fica melhor, lado a lado, numa página para olhar |
| `./venv/bin/python medir.py arquivo.pdf` | quanto o Docling leva por PDF (CPU ou GPU) |
| `./venv/bin/python conferir_vetorizacao.py` | o vetor do worker bate com o do servidor do publi-bot |

## 7. Quando algo dá errado

| Sintoma | Veja |
|---|---|
| `curl` sai vazio, ou o script diz "000" | o worker está subindo ou caiu: `systemctl --user status worker-gpu` e o log |
| `503 gpu_ocupada` | normal: a placa está com outro trabalho. O publi-bot e os scripts tentam de novo |
| `422 voz_desconhecida` | a voz não existe neste worker: liste as vozes (seção 5) |
| `500 falha_na_voz` | `journalctl --user -u worker-gpu -n 60 --no-pager`; a causa aparece no fim |
| o worker não sobe depois de ligar algo no `.env` | o log diz o que falta (em geral, rodar o `instalar_voz.sh` ou o `instalar_animacao.sh`) |
| tudo lento, memória no limite | `./deploy/situacao.sh` e o `afericao.py`; seção **Convivendo com a máquina** do `README.md` |
