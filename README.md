# worker-gpu

Um processo, uma placa, um árbitro.

Esta máquina tem uma GPU e vários clientes. A placa é **indivisível**: um
modelo de texto de 30B já ocupa quase toda a VRAM de 8 GB, e a difusão precisa
de ~5,4 GB. Não cabem juntos — e o que acontece quando se tenta não é um erro,
é o processo caindo para CPU em silêncio, dezenas de vezes mais lento, sem
nada no log dizendo por quê.

Por isso **tudo** entra por aqui, inclusive o texto:

| Rota | O que faz |
|---|---|
| `POST /v1/chat/completions` | texto — repassa ao Ollama |
| `POST /v1/images/generations` | imagem — difusão |
| `POST /parse/` | PDF para Markdown com análise de layout |
| `GET /v1/models` | catálogo |
| `GET /health/` | estado, sem credencial |

As três primeiras disputam **um lock só**. Quem não pega recebe `503` com
`Retry-After` calculado.

> Integrando um cliente? **[`INTEGRACAO.md`](INTEGRACAO.md)** tem o guia
> completo. O resto deste arquivo é sobre operar a máquina.

## Por que um processo, e não três serviços

Foi três, e não funcionava. Cada serviço tinha um lock e protegia a si mesmo;
nenhum sabia dos outros. O Ollama, então, não participava de nada — ele decide
sozinho quando carregar e descarregar modelo.

O resultado era o esperado em retrospecto: a difusão encontrava a placa cheia,
caía para CPU, e um único lote consumia horas de processador e 17 GB de RAM.

Um lock em memória só é correto porque **todo** pedido de GPU entra no mesmo
processo. É também o que torna seguro mandar o Ollama soltar a VRAM: detendo o
lock, ninguém está gerando texto.

## Instalação

Cinco passos, nesta ordem. O `instalar.sh` é o **último** deles — ele instala
a unit do systemd e confere que o serviço sobe; ele não cria o venv, não
instala dependências, não escreve o `.env` e não baixa modelo. Rodado antes da
hora, ele para com o erro dizendo o que falta.

```bash
# 1. Ambiente. São vários GB (torch, docling, diffusers).
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2. Configuração.
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # WORKER_SHARED_SECRET

# 3. Endereço de escuta: edite BIND_HOST no .env (veja a tabela adiante).

# 4. Pesos do modelo de imagem: ~7 GB, uma vez.
./venv/bin/python baixar_modelo.py

# 5. Unit do systemd + conferência de /health/.
./deploy/instalar.sh
```

O passo 4 não é opcional na prática. O serviço carrega o modelo de forma
preguiçosa, e o primeiro pedido de todos não carrega: **baixa**. Sem isso, o
primeiro cliente a pedir uma imagem espera minutos e leva um tempo esgotado. O
`/health/` informa `baixado`.

O Docling baixa os modelos dele sozinho, no primeiro `/parse/` — são bem
menores, mas valem um pedido de aquecimento antes de pôr em produção.

Falta ainda o Ollama, que é um serviço separado desta máquina e tem
configuração própria — a seção seguinte.

> Qual Python? O OCR do Docling não é instalável em 3.14 (veja a seção do
> Docling). Se você for usar OCR, crie o venv com 3.12 ou 3.13.

### O Ollama

Ele passa a ser **interno**: só o worker fala com ele. Tire-o da rede.

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf <<'CONF'
[Service]
# Loopback. Quem publica na rede privada e o worker, que arbitra a placa.
# Deixar o Ollama exposto e deixar uma porta dos fundos sem arbitragem.
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_PARALLEL=1"
CONF

sudo systemctl daemon-reload && sudo systemctl restart ollama
```

**`OLLAMA_KEEP_ALIVE` se ajusta aqui, no serviço do Ollama** — o worker não o
envia. Ele já enviou, e foi removido: `keep_alive` não é campo do dialeto da
OpenAI, e o proxy fala esse dialeto, então a camada compatível descartava a
chave. A variável existia no `.env` do worker e não tinha efeito nenhum.

Ele deixa de ser crítico de qualquer forma: o worker descarrega o modelo quando
precisa da placa para imagem. Mantê-lo alto passa a ser vantagem — o texto não
recarrega à toa.

**`OLLAMA_CONTEXT_LENGTH` é o padrão da máquina**, e vale para quem não pedir
nada. Um cliente que manda `options.num_ctx` no pedido tem a janela dele
respeitada: o worker roteia esse pedido pelo `/api/chat` do Ollama, porque a
camada compatível descartaria o `options` sem avisar. Veja `dialeto.py`.

Isso significa que a janela passou a **custar VRAM de verdade**. Uma cache de
atenção de 16k num modelo de 7B pode ser a diferença entre caber na placa e o
Ollama espalhar camadas para a CPU — que não dá erro, só fica lento. O
`/health/` mostra `context_length` e `size_vram` por modelo carregado.

### O Docling (conversão de PDF)

A rota `/parse/` usa o [Docling](https://github.com/docling-project/docling):
ele olha a página com modelos de visão e reconstrói a estrutura — ordem de
leitura em coluna dupla, títulos, tabelas, legendas — em vez de ler a camada
de texto. O `INTEGRACAO.md` descreve em detalhe o que ele identifica e o que
descarta; aqui está só o que você regula nesta máquina.

| Variável | Padrão | O que muda |
|---|---|---|
| `CONVERSAO_ATIVA` | `sim` | `nao` remove a rota. Use se esta máquina só gera imagem |
| `DOCLING_DEVICE` | `auto` | `cpu`, `cuda` ou `auto` |
| `DOCLING_THREADS` | `0` (o Docling decide) | só vale em CPU |
| `DOCLING_OCR` | `nao` | ligue **apenas** se o acervo tem PDF digitalizado |
| `MAX_PDF_BYTES` | 100 MB | acima disso a resposta é `413` |

**GPU não é requisito.** A análise de layout roda em CPU; a placa muda o
tempo, não o resultado. Meça antes de decidir:

```bash
./venv/bin/python medir.py um-artigo.pdf          # como está configurado
./venv/bin/python medir.py um-artigo.pdf --cpu
./venv/bin/python medir.py um-artigo.pdf --cuda
```

Em `cuda`, lembre que a conversão passa a **disputar a placa** com o texto e a
imagem — o lock é o mesmo. Numa máquina com uma placa só e Ollama residente,
`cpu` costuma ser a escolha certa mesmo sendo mais lenta: ela converte em
paralelo à geração de texto, em vez de esperar a vez.

O **OCR** é a parte cara e vem desligado. PDF com camada de texto não precisa
dele; PDF digitalizado sem ele converte para quase nada, **sem erro**. Se
ligar, o motor precisa estar instalado — o worker recusa subir sem ele, em vez
de falhar só na primeira digitalização.

> O OCR do Docling (`rapidocr`) só é declarado para Python < 3.14. Se o venv
> for 3.14, `DOCLING_OCR=sim` não sobe. Use 3.12 ou 3.13.

O primeiro `/parse/` depois de cada reinício carrega os modelos: dezenas de
segundos a mais, uma vez. `/health/` mostra `conversao.carregado`.

### O endereço de escuta

| Valor | Quem alcança |
|---|---|
| `127.0.0.1` | só esta máquina |
| `$(tailscale ip -4)` | os outros clientes, pela rede privada |
| `0.0.0.0` | a internet inteira. O instalador recusa |

Trocar exige **reinstalar a unit** (`./deploy/instalar.sh`): o endereço entra
na linha de comando do uvicorn.

### Sobe sozinho no boot?

| Instalação | Sobe quando |
|---|---|
| `./deploy/instalar.sh` (padrão, unit de **usuário**) | você faz login |
| o mesmo, **mais** `sudo loginctl enable-linger $USER` | no boot, sem login |
| `./deploy/instalar.sh --sistema` | no boot, sempre |

Numa máquina pessoal que também atende outros, `enable-linger` é o que você
quer.

## Medir antes de decidir

```bash
./venv/bin/python medir_imagem.py --tamanhos 512x288,768x432,1024x576
./venv/bin/python medir.py um-artigo.pdf --cpu --threads 1
```

Os dois medem a **segunda** execução de cada combinação: a primeira inclui a
montagem do pipeline, que se paga uma vez, e misturar as duas produz um número
que não serve para decidir nada.

### O que a medição já mostrou nesta placa

| tamanho | área relativa | tempo | pico VRAM |
|---|---|---|---|
| 512×288 | 1× | 14,5 s | 5,3 GB |
| 768×432 | 2,25× | 13,3 s | 5,4 GB |
| 1024×576 | 4× | 17,8 s | 5,4 GB |

Quatro vezes mais pixels por 23% mais tempo — e o menor foi *mais lento* que o
do meio. O custo dominante é mover pesos entre RAM e VRAM, que é fixo por
geração; a difusão em si é o troco.

**Consequência prática:** reduzir o tamanho para "economizar" não economiza. O
que muda o tempo é o modelo. E o pico de VRAM não cai com o tamanho — é por
isso que o árbitro, e não uma imagem menor, é a resposta para dividir a placa.

Em CPU, o mesmo 512×288 levou 160 s: 11 vezes mais, com qualidade pior.

## Qualidade da imagem

As imagens saindo "com cara de IA" quase nunca é só o modelo. Quatro coisas
desta configuração degradam o resultado antes de o modelo entrar na conta, e as
quatro são silenciosas — a imagem sai, com 200, só pior.

| O quê | Por quê | Conserto |
|---|---|---|
| **prompt em português** | os codificadores de texto do SDXL (CLIP ViT-L, OpenCLIP ViT-bigG) foram treinados em legendas da web, esmagadoramente inglesas. `borrado` não está no vocabulário; `blurry` está | mande o prompt em inglês. O worker **avisa no journal** quando detecta português |
| **VAE em float16** | o VAE que vem no SDXL estoura em float16 — e é em float16 que o serviço carrega, porque float32 não cabe. Dá manchas, faixas de cor e às vezes imagem preta | `IMAGEM_VAE=madebyollin/sdxl-vae-fp16-fix`. O worker avisa quando vê um VAE de SDXL sem ajuste |
| **tamanho fora da área de treino** | o SDXL foi treinado numa **grade de proporções a área constante** (~1,05 MP). A de 16:9 que ele conhece é **1344×768**, não 1024×576 | peça um tamanho da grade. `IMAGEM_LADO_MAXIMO=1536` para as largas caberem; `/health/` publica a grade em `imagem.grade` |
| **amostrador padrão** | o Euler do diffusers precisa de mais passos para o mesmo resultado | `IMAGEM_SCHEDULER=dpm++2m_karras` |

O prompt negativo deste serviço esteve em português por muito tempo, e por isso
não fazia nada. Era o pior tipo de configuração: aparecia no `.env`, parecia
ativa, e o efeito era o de não existir.

### O modelo

O SDXL base é o padrão porque é o que existe sem escolha, não porque é bom para
foto realista — ele é de 2023 e é um dos piores da família nisso. Um **ajuste
fino da mesma família** troca sem mexer em código, sem mudar a VRAM e sem mudar
o pipeline:

```bash
IMAGEM_VAE=madebyollin/sdxl-vae-fp16-fix
IMAGEM_MODELO=SG161222/RealVisXL_V5.0        # ou RunDiffusion/Juggernaut-XL-v9
```

Depois, `./venv/bin/python baixar_modelo.py` e reiniciar.

Outras famílias cabem em 8 GB e exigem mais: **SD 3.5 Medium** segue melhor o
prompt e escreve texto legível, mas precisa de outro pipeline; **FLUX.1-schnell**
é o melhor em realismo e são 12B parâmetros — só entra quantizado, e devagar.
Não meça por reputação: meça na bancada.

### A bancada

```bash
systemctl --user stop worker-gpu

./venv/bin/python bancada.py --modelos stabilityai/stable-diffusion-xl-base-1.0,SG161222/RealVisXL_V5.0
./venv/bin/python bancada.py --vaes ',madebyollin/sdxl-vae-fp16-fix'
./venv/bin/python bancada.py --schedulers euler,dpm++2m_karras --passos 25,40

systemctl --user start worker-gpu
```

Ela gera as variantes com **prompt e semente fixos** e escreve uma página com
as imagens lado a lado, agrupadas por cena. Duas sementes por variante, no
mínimo: a variação entre sementes do mesmo modelo é frequentemente maior que a
variação entre modelos, e quem olha uma imagem de cada escolhe a sorte.

Ela existe para tirar o PubliBot do caminho — ajustar qualidade pela tela de
revisão dele passa por LLM, fila, worker e navegador, e quando a imagem sai
ruim não dá para saber de quem foi a culpa.

Olhe nesta ordem, que é a ordem em que a geração realista quebra: **mãos e
dedos**, **pele**, **reflexo e metal**, **texto na cena**, **fundo desfocado**.

## Diagnóstico

```bash
curl -s http://<endereco>:8090/health/ | jq
journalctl --user -u worker-gpu -f
```

| Sintoma | Causa provável |
|---|---|
| `503 gpu_ocupada` | funcionando como projetado; o cliente deve voltar depois |
| `503 sem_vram` | o Ollama não soltou a placa. Veja `ollama.carregados` no `/health/` |
| um cliente sempre paga troca de modelo | `/health/` diz `modelo` (em uso) e `ollama.carregados` (residentes); veja a afinidade no `INTEGRACAO.md` |
| `503 ollama_indisponivel` | o Ollama caiu, `OLLAMA_URL` está errado, ou o Ollama respondeu 5xx |
| `503 timeout` no texto | o pedido não coube em `OLLAMA_TIMEOUT`. Prompt grande demais, ou modelo lento demais para o orçamento |
| `/health/` com um bloco `{"erro": ...}` | aquela parte falhou ao ser coletada; o resto do corpo continua válido |
| `ha_segundos` alto e parado | trabalho preso segurando o lock. O lock não tem watchdog: ele solta em `OLLAMA_TIMEOUT` |
| `/health/` dá `timed out` | um handler bloqueante no event loop — nenhum deveria ser `async def` |
| um cliente diz que o prompt de sistema some | janela de contexto pequena. Veja `ollama.carregados_detalhe[].context_length` no `/health/` |
| `baixado: false` | rode `baixar_modelo.py` antes do primeiro uso (isto e o modelo de IMAGEM; os de texto o worker baixa sozinho) |
| `503 baixando_modelo` | o modelo de texto pedido nao estava no disco. Acompanhe em `ollama.baixando` no `/health/` |
| `404` num modelo que deveria existir | o download falhou; a mensagem diz por que. Confira o nome contra `ollama list` |
| `ultimo_dispositivo: cpu` | caiu para CPU. Com `IMAGEM_PERMITIR_CPU=nao` isso não deveria acontecer |
| imagens com manchas ou faixas de cor | `IMAGEM_VAE` vazio num modelo SDXL. Veja **Qualidade da imagem** |
| imagens genéricas, mal compostas | prompt em português. Procure o aviso: `journalctl --user -u worker-gpu \| grep portugues` |
| assunto duplicado, geometria torta | tamanho fora da grade de treino. `journalctl --user -u worker-gpu \| grep grade` |
| uvicorn morre no boot | `BIND_HOST` inexistente, ou `BIND_PORT` vazio |

## Testes

```bash
./venv/bin/python -m pytest -q
```

Os modelos pesados não entram: `gerar_imagens`, `obter_conversor` e o Ollama
são substituídos. O que se exercita é o contrato HTTP, a arbitragem e as
recusas — e é lá que os erros deste repositório doem, porque do outro lado há
clientes que só veem JSON.

`contrato/` guarda os exemplos de resposta que os clientes copiam. Os testes
conferem que as respostas reais ainda têm aquela forma.
