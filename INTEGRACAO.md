# Integrando um cliente ao worker-gpu

Guia para adaptar um sistema que hoje fala com o Ollama direto — ou que não
falava com nada.

O worker é um **árbitro de GPU**, não um provedor. A diferença aparece num
ponto só, e é o ponto inteiro deste documento: **ele recusa quando a placa
está ocupada, e espera que você tente de novo.**

## O que muda no seu cliente

| Antes | Depois |
|---|---|
| `http://gpu:11434/v1/chat/completions` | `http://gpu:8090/v1/chat/completions` |
| sem credencial | `Authorization: Bearer <segredo>` |
| 200 ou erro | 200, ou **503 com `Retry-After`** |
| `stream: true` funcionava | `stream` é recusado |

O corpo e a resposta são os mesmos do dialeto da OpenAI. Se o seu cliente já
usa uma biblioteca compatível, trocar a URL base e a chave costuma bastar —
**menos o 503**, que é o que exige código novo.

## Primeiro contato

Antes de escrever código, confirme os três de fora para dentro. Se algum
falhar aqui, não é o seu cliente que está errado.

```bash
WORKER=http://<maquina>:8090
SEGREDO=<o WORKER_SHARED_SECRET do worker>

# 1. Está de pé? (sem credencial, de propósito)
curl -s $WORKER/health/ | jq

# 2. A credencial vale? Sem ela isto dá 401.
curl -s -H "Authorization: Bearer $SEGREDO" $WORKER/v1/models | jq

# 3. Uma geração de verdade, curta.
curl -s -H "Authorization: Bearer $SEGREDO" -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:7b-instruct","messages":[{"role":"user","content":"diga ok"}]}' \
  $WORKER/v1/chat/completions | jq -r '.choices[0].message.content'
```

**Confira que o `$SEGREDO` não está vazio antes de testar.** Uma variável de
shell vazia não dá erro: o `curl` manda `Authorization: Bearer ` e o worker
responde `401` com toda a razão — e você vai procurar defeito no worker. Um
`grep` num caminho de `.env` errado é a causa mais comum:

```bash
[[ -n "$SEGREDO" ]] && echo "segredo com ${#SEGREDO} caracteres" || echo "VAZIO"
```

| O que veio | O que significa |
|---|---|
| `401` | segredo errado, faltou o cabeçalho, ou a sua variável está vazia |
| `404` na rota | aquela rota está desligada (`IMAGEM_ATIVA`/`CONVERSAO_ATIVA`) |
| `503` | está funcionando — a placa só está ocupada agora |
| nada, e o curl expira | o worker não está escutando nesse endereço |

O `/health/` responde sem credencial de propósito: um diagnóstico que precisa
de segredo não serve para descobrir por que o segredo não funciona.

**Não há rota de teste, e é de propósito.** Uma que não tocasse a placa
provaria só que o processo está de pé — o que o `/health/` já diz, melhor. E
uma que tocasse a placa seria um pedido normal, competindo pelo lock como
qualquer outro, com a desvantagem de gastar GPU sem produzir nada. Os três
comandos acima são o teste: `/health/` para o processo, `/v1/models` para a
credencial, e uma geração curta para o caminho inteiro.

## O contrato do 503

Esta é a parte que decide se a integração vai funcionar sob carga.

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 40

{"error": {"code": "gpu_ocupada",
           "message": "a GPU esta em uso por 'imagem'; tente em 40s",
           "ocupante": "imagem"}}
```

Quatro códigos, e eles pedem coisas diferentes:

| `error.code` | Significa | `Retry-After` | O que fazer |
|---|---|---|---|
| `gpu_ocupada` | outro trabalho está na placa | sim | reagendar para daqui a `Retry-After` |
| `ollama_indisponivel` | o Ollama caiu, reinicia, ou respondeu 5xx | sim | idem; se persistir por horas, alertar |
| `sem_vram` | a placa não tem espaço agora | sim | idem, com paciência maior |
| `timeout` | o trabalho passou do orçamento | **não** | **não repita igual** — reduza o pedido |
| `baixando_modelo` | o modelo pedido não estava no disco e está sendo baixado | sim | reagendar; o `Retry-After` vem do progresso real |

**Todo 503 que sai do worker tem `error.code`.** Não existe 503 sem código:
mesmo um 5xx vindo do Ollama é envelopado nesta forma antes de sair. Se você
receber um sem código, é bug do worker — abra issue.

**O `timeout` vem sem `Retry-After`, de propósito.** Para esse código a
orientação é reduzir o pedido, não voltar igual mais tarde; mandar o cabeçalho
junto seria o contrato se contradizendo dentro da mesma resposta. Leia o
cabeçalho com um padrão (`headers.get("Retry-After", 60)`), nunca com acesso
direto.

**Código que você não conhece: adie com teto, nunca desista.** Se um dia a
lista crescer, um cliente que trate código desconhecido como falha definitiva
quebra no dia do acréscimo. Trate como adiável, com teto de tentativas e log
alto — o teto é o que impede o adiamento eterno, e ele protege você também nos
códigos que você já conhece.

**Nenhum deles é falha do seu trabalho.** Um 503 não deve consumir tentativa,
não deve abrir disjuntor, e não deve marcar o trabalho como falho. Se o seu
sistema conta tentativas, esse é o detalhe que mais importa: tratando 503 como
erro, alguns minutos de disputa esgotam as tentativas de uma fila inteira.

### Por que 503 e não uma fila no worker

Porque o worker **não tem estado durável**. Uma fila em memória perde trabalho
no primeiro restart, e restart aqui é rotina — atualização, troca de modelo,
reboot da máquina.

Você já tem fila persistente e sabe retomar. Enfileirar no worker seria uma
segunda fila, invisível para você e pior que a que já existe.

### O `Retry-After` é calculado

Ele vem do que está rodando agora, não de uma constante. Se uma imagem começou
há 20 segundos e costuma levar 60, você recebe `Retry-After: 40`. Respeite-o:
voltar antes garante outro 503, e voltar muito depois desperdiça placa ociosa.

## Como implementar do seu lado

Se o seu sistema tem orquestrador com passos, o padrão é este — separar
"adiar" de "falhar":

```python
resposta = requests.post(
    f"{WORKER}/v1/chat/completions",
    json=corpo,
    headers={"Authorization": f"Bearer {SEGREDO}"},
    timeout=600,
)

if resposta.status_code == 503:
    dados = resposta.json().get("error", {})
    espera = int(resposta.headers.get("Retry-After", 60))

    if dados.get("code") == "timeout":
        # Repetir igual daria o mesmo resultado. Reduza antes.
        raise TrabalhoPrecisaDeAjuste(dados.get("message", ""))

    if adiamentos >= TETO_DE_ADIAMENTOS:
        # O teto e o que impede o adiamento eterno. Sem ele, um Ollama fora do
        # ar por dias mantem o trabalho girando na fila para sempre.
        raise TrabalhoFalhou(f"{adiamentos} adiamentos por {dados.get('code')}")

    # Adiar NAO gasta tentativa: nada deu errado, so nao era a hora.
    raise Adiar(dados.get("message", ""), tentar_em_segundos=espera)

if resposta.status_code in (400, 404, 422):
    # Isto e "seu pedido esta errado". Repetir nao muda nada.
    raise TrabalhoFalhou(resposta.text[:300])

resposta.raise_for_status()
```

Três detalhes que decidem se isso funciona sob carga:

- **o teto de adiamentos não é opcional.** `Retry-After` chega a ser 5 s, e uma
  geração longa de outro cliente produz dezenas de 503 seguidos. Sem teto, um
  pedido que nunca vai caber gira para sempre;
- **respeite o `Retry-After` como piso, não como valor final.** Um recuo
  próprio por cima (`max(retry_after, 5 * 2 ** adiamentos)`, limitado) evita
  bater na porta a cada 5 s durante uma geração de minutos;
- **`ConnectionError` também é adiável.** Se o worker reiniciar no meio do seu
  pedido, a conexão cai sem status HTTP nenhum. Conte no mesmo teto.

Se o seu sistema não tem orquestrador, o mínimo aceitável é reagendar a tarefa
para `agora + Retry-After` e sair — **nunca** um `sleep` segurando o processo:
a placa pode estar ocupada por minutos, e você prende um worker inteiro
esperando.

## Modelo por cliente, e afinidade

**O modelo é de quem pede.** O worker repassa o corpo ao Ollama praticamente
intacto — o `model` que você mandar é o que roda. Um CRM com modelo por tenant
e um padrão de sistema funciona sem nada especial aqui: o worker arbitra a
placa, não a escolha.

Um limite honesto: trocar de modelo **custa**. Com
`OLLAMA_MAX_LOADED_MODELS=1` — a configuração recomendada numa placa só — um
modelo diferente expulsa o que estava carregado. Não é falha, é tempo: o
próximo pedido espera o carregamento.

### Modelo que ainda não está na máquina

**Não precisa pré-baixar.** Se você pedir um modelo que não está no disco, o
worker o baixa sozinho, em segundo plano, e te responde:

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 240

{"error": {"code": "baixando_modelo",
           "message": "o modelo 'llama3.1:8b' esta sendo baixado (37%)"}}
```

Você reagenda como em qualquer 503. Os pedidos seguintes recebem o mesmo
código com o progresso atualizado, e o `Retry-After` **encolhe conforme o
download anda** — ele é calculado do ritmo real, não é constante. Quando
termina, o próximo pedido passa normalmente.

**O download não toma a placa.** Ele é rede e disco: a rota de imagem e a de
conversão seguem funcionando durante ele, e `ocupada` continua `false` no
`/health/`. Quem quiser acompanhar:

```bash
curl -s http://<worker>:8090/health/ | jq '.ollama.baixando'
# [{"modelo": "llama3.1:8b", "porcento": 37.4, "ha_segundos": 95}]
```

#### Se o download falhar, você recebe 404 — e deve desistir

Essa é a parte que importa para a sua fila. Um nome de modelo que não existe
no registro do Ollama **não** vira 503 eterno:

| Quando | Você recebe | O que fazer |
|---|---|---|
| primeiro pedido, modelo ausente | `503 baixando_modelo` | reagendar |
| enquanto baixa | `503 baixando_modelo` | reagendar |
| o download falhou | **`404`**, com a mensagem da falha | **falhar o trabalho** — o nome está errado, ou a máquina não alcança o registro |
| baixou | `200` | nada |

O worker lembra a falha por alguns minutos justamente para o seu cliente
receber um 404 em vez de disparar um download novo a cada retentativa. Passado
esse prazo ele tenta de novo sozinho, então uma queda de rede se cura sem
ninguém reiniciar nada.

#### O que isso custa

Gigabytes de disco, sem ninguém aprovar. **Um nome de modelo errado que por
acaso exista no registro do Ollama vai ser baixado.** Se a sua lista de
modelos vem de um banco onde alguém digita o nome, esse é o risco real —
confira os nomes contra `GET /v1/models` quando cadastrar, não quando usar.

O dono da máquina desliga isso com `BAIXAR_MODELO_AUTOMATICO=nao` no `.env`, e
aí o comportamento volta a ser o antigo: o Ollama responde 404 na hora.

### A janela de contexto, por pedido

**Funciona: mande `options.num_ctx` no corpo.**

```json
{"model": "qwen2.5:7b-instruct",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."}],
 "options": {"num_ctx": 16384},
 "max_tokens": 800}
```

Não é o dialeto da OpenAI — é o do Ollama, e é de propósito. O worker olha o
seu pedido: se ele traz `options` (ou `keep_alive`), o pedido vai pelo
`/api/chat` do Ollama, onde esses campos existem; se não traz, segue pelo
`/v1/chat/completions`, exatamente como antes. A **resposta é a mesma nos dois
casos** — você não precisa saber qual caminho o seu pedido tomou, e um teste do
worker garante que as duas formas não divergem.

Por que isso não era assim antes: a camada compatível do Ollama desserializa o
corpo num struct tipado e **descarta chave que não conhece, sem erro e sem
log**. `options` inteiro caía nessa. Um pedido com `num_ctx: 16384` respondia
200 tendo rodado com 4096.

| Você manda | Vale | Observação |
|---|---|---|
| `options.num_ctx` | **sim** | leva o pedido pelo dialeto nativo |
| `options.num_predict` | **sim** | ou use `max_tokens`, que é equivalente |
| `options.*` (qualquer opção do Ollama) | **sim** | repassado como está |
| `keep_alive` | **sim** | leva o pedido pelo dialeto nativo |
| `max_tokens`, `temperature`, `top_p`, `seed`, `stop` | **sim** | nos dois caminhos |
| `response_format` (inclusive `json_schema`) | **sim** | nos dois caminhos |
| `n` maior que 1 | **não** | o Ollama gera uma resposta por pedido; vira aviso no journal do worker |

**Se você mandar `options` e um campo equivalente da OpenAI, o `options`
ganha** — ele é a intenção mais específica. `max_tokens: 100` junto de
`options: {"num_predict": 500}` roda com 500.

**Campo que o worker não sabe traduzir vira AVISO no journal dele**, nunca
silêncio. Se você suspeitar que algo não está chegando, peça ao dono da máquina:

```bash
journalctl --user -u worker-gpu -f | grep dialeto
```

#### Não precisa mais de Modelfile

A recomendação anterior era embutir a janela no modelo com
`PARAMETER num_ctx` e um `ollama create`. Ela **continua funcionando**, mas
deixou de ser necessária, e a razão de ter saído de cena é prática: alguém
precisava rodar o `ollama create` a cada máquina nova, e **esquecer não dava
erro nenhum** — voltava a janela padrão, em silêncio. Se você já criou modelos
derivados, eles seguem valendo; se não criou, não crie.

#### Um custo real que aparece agora

Com o `num_ctx` passando a valer de verdade, ele passa a **custar VRAM**. A
cache de atenção de 16k tokens num modelo de 7B não é de graça, e numa placa de
8 GB pode ser a diferença entre caber e o Ollama espalhar camadas para a CPU —
o que não dá erro, só fica muitas vezes mais lento.

Se o tempo de resposta piorar depois de você subir a janela, é isso. Confira:

```bash
curl -s http://<worker>:8090/health/ \
  | jq '.ollama.carregados_detalhe[] | {name, context_length, size_vram}'
```

Peça a janela que você precisa, não a maior que couber.

### Como detectar truncamento

Janela pequena demais não dá erro: ela **come o prompt pelo início**, e o
primeiro a morrer é o prompt de sistema. A resposta volta 200, com JSON válido
e schema respeitado, e o modelo não viu as suas instruções. É o defeito que
passa por revisão automatizada e envenena tudo o que for construído em cima.

Três sinais, do mais barato para o mais caro:

| Sinal | Custo | O que prova |
|---|---|---|
| `usage.prompt_tokens` vs. o que você mandou | nada | truncou, se vier muito abaixo — e travado num número redondo (2048, 4096) é conclusivo |
| `finish_reason == "length"` | nada | a **saída** foi cortada pelo `max_tokens`, não a entrada |
| `ollama.carregados_detalhe[].context_length` no `/health/` | uma viagem | com que janela o modelo **está** carregado |
| um canário no início do prompt de sistema, ecoado na saída | um campo no schema | se o início do prompt sobreviveu, **neste pedido** |

O primeiro é o que a maioria deveria usar: já está na resposta, não custa nada
e não pede campo novo. O segundo é diagnóstico de **máquina** ("está
configurada errada"), e não confirmação de pedido — ele é lido por outra
viagem, depois, e entre a sua inferência e a sua leitura outro cliente pode ter
recarregado o modelo com outra janela.

### Como saber o que está na placa

Três formas, da mais barata para a mais cara:

| De onde | O que diz | Custa |
|---|---|---|
| um **200** seu | o seu modelo é o que está carregado agora | nada |
| o **503** que você levou | `error.modelo` — o que está em uso | nada |
| `GET /health/` | `modelo` (em uso) e `ollama.carregados` (residentes) | uma viagem |
| `GET /health/` | `ollama.carregados_detalhe` — com **`context_length`** | a mesma viagem |

A primeira é a que se esquece: depois de uma resposta bem-sucedida, você já
sabe. Não precisa perguntar.

A segunda é a que fecha o ciclo. Antes, um 503 dizia só "volte em 18s"; agora
diz **com o quê** vale voltar:

```json
{"error": {"code": "gpu_ocupada",
           "message": "a GPU esta em uso por 'texto'; tente em 18s",
           "ocupante": "texto",
           "modelo": "qwen2.5:7b-instruct"}}
```

`modelo` só aparece quando se sabe qual é. **Ausente não quer dizer nenhum** —
quer dizer que a tarefa em curso não tem modelo nomeado (uma conversão de
PDF, por exemplo).

### O padrão: ordenar a fila, não esperar por ela

Sabendo o modelo residente, a sua fila escolhe melhor:

```python
def proximo(fila, modelo_na_placa):
    # Primeiro os que nao pagam troca...
    for trabalho in fila:
        if trabalho.modelo == modelo_na_placa:
            return trabalho
    # ...e, nao havendo, qualquer um: trocar e caro, parar e pior.
    return fila[0] if fila else None
```

Duas regras que impedem isso de virar um problema novo:

- **É uma dica, nunca uma garantia.** Entre você ler e você postar, outro
  cliente pode tomar a placa e trocar o modelo. Nada pode depender disso estar
  certo — no pior caso você paga uma troca, que é exatamente o que aconteceria
  sem afinidade nenhuma.
- **Não deixe um tenant na fila para sempre.** Preferir o modelo carregado sem
  limite é uma receita de inanição: o tenant do modelo menos usado nunca é
  atendido. Limite a sequência (N trabalhos, ou X minutos) e depois pague a
  troca.

Não vale a pena ir além disso do lado do cliente. Prioridade de verdade —
justiça, envelhecimento, reserva com hora marcada — é trabalho do árbitro, e
o `arbitro.py` diz explicitamente que hoje ele não faz nada disso.

## Timeouts do seu lado

| Rota | Timeout sugerido | Orçamento do worker |
|---|---|---|
| `/v1/chat/completions` | 600 s | `OLLAMA_TIMEOUT`, 540 s |
| `/v1/images/generations` | 300 s | `IMAGEM_TEMPO_MAXIMO`, 600 s |
| `/parse/` | 600 s | sem teto |
| `/health/` | 10 s | — |

**O seu timeout precisa ser MAIOR que o orçamento do worker, e não igual.** Os
dois relógios não começam juntos: o seu parte quando você envia, o do worker só
depois de o pedido chegar, passar pela credencial e tomar o lock. Iguais, quem
desiste primeiro é você — e você abandona um trabalho que ainda está segurando
a placa, de modo que a sua retentativa imediata bate num serviço ocupado. Com
540 s do lado do worker e 600 s do seu, quem desiste primeiro é ele, e ele
desiste devolvendo um 503 legível em vez de um socket cortado.

São generosos de propósito: o worker pode estar carregando um modelo (dezenas
de segundos) antes de começar. Um timeout curto desiste de um trabalho que
estava indo bem — e, pior, o worker continua trabalhando e segurando a placa,
de modo que a sua retentativa imediata bate num serviço ocupado.

## As rotas

### Texto

```http
POST /v1/chat/completions
Authorization: Bearer <segredo>

{"model": "qwen2.5:7b-instruct",
 "messages": [{"role": "system", "content": "..."},
              {"role": "user", "content": "..."}],
 "temperature": 0.2,
 "stream": false}
```

Resposta: `contrato/texto-resposta.json`.

`model` é seu: o worker repassa o que você mandar, sem impor nem substituir.
Veja **Modelo por cliente, e afinidade** para o que isso custa quando muda.

`stream: true` recebe **422**. O árbitro precisa saber quando o trabalho
termina para soltar a placa, e uma resposta em streaming só termina quando o
cliente termina de ler.

### Imagem

```http
POST /v1/images/generations
Authorization: Bearer <segredo>

{"prompt": "...", "n": 3, "size": "1024x576"}
```

Resposta: `contrato/imagem-resposta.json`. Sempre `b64_json` — um link
temporário expiraria antes de você publicar a imagem.

Cada lado do `size` precisa ser múltiplo de 8. Não é capricho: o modelo
arredonda por dentro e devolveria uma imagem de tamanho diferente do pedido,
sem avisar.

O tamanho quase não muda o tempo (veja o README): o custo é dominado por mover
pesos entre RAM e VRAM. Peça o tamanho que você quer publicar.

**Peça 1344×768 para 16:9, e não 1024×576.** O SDXL foi treinado em cerca de
1024×1024 pixels de **área**, distribuídos em proporções fixas, e a de 16:9 que
ele conhece é 1344×768. Pedir abaixo da área de treino gera anatomia e
composição piores — e como o tamanho quase não muda o tempo, pedir menos não
economiza nada. O teto por lado é 1344 (`imagem.lado_maximo` no `/health/`).

**O prompt precisa estar em inglês.** Os codificadores de texto do SDXL foram
treinados em legendas da web, esmagadoramente inglesas: um prompt em português
não dá erro, gera uma imagem a partir do pouco sinal que sobrou, e o resultado
é genérico e mal composto. O worker **não traduz** — traduzir em silêncio
mudaria o seu pedido —, mas registra um aviso no journal quando detecta
português. Se você gera o prompt com um LLM, peça a ele em inglês.

### Conversão de PDF (Docling)

```http
POST /parse/
Authorization: Bearer <segredo>
X-Expected-Sha256: <opcional>

multipart/form-data, campo `file`
```

```json
{"markdown": "# Titulo\n\nUm paragrafo.",
 "sha256": "e3b0c4...",
 "bytes": 30,
 "duration_ms": 1200}
```

Exemplo completo em `contrato/conversao-resposta.json`.

#### O que ele realmente faz

Por trás está o **[Docling](https://github.com/docling-project/docling)**, da
IBM. Ele não lê a camada de texto do PDF — ele **olha a página** com modelos de
visão e reconstrói a estrutura antes de escrever qualquer coisa.

A diferença não é de acabamento, é de correção. Num artigo científico de duas
colunas, um extrator de camada de texto (pdftotext, PyPDF2, pdfminer) devolve
as frases **intercaladas**: a primeira linha da coluna esquerda, a primeira da
direita, a segunda da esquerda. O resultado parece texto — tem palavras
corretas, pontuação, parágrafos — e só na leitura atenta se descobre que as
frases não se encadeiam. É o pior tipo de defeito: passa por revisão
automatizada, passa por contagem de caracteres, e envenena tudo o que for
construído em cima.

O que o Docling identifica e o que faz com cada coisa:

| Na página | No Markdown |
|---|---|
| ordem de leitura (coluna dupla, caixas, barras laterais) | texto em ordem linear correta |
| títulos e subtítulos, por hierarquia visual | `#`, `##`, `###` |
| parágrafos | blocos separados por linha em branco |
| listas | `-` e `1.` |
| **tabelas**, com células mescladas | tabela em Markdown (`do_table_structure=True`) |
| figuras e legendas | a legenda vira texto; a imagem não é exportada |
| cabeçalho, rodapé, número de página | **descartados** |
| notas de rodapé | texto, fora do corpo |
| fórmulas e código | blocos próprios |

É por isso que a saída serve para **separar um artigo em seções por título**:
os `#` do Markdown correspondem à hierarquia que estava na página, não a um
palpite sobre tamanho de fonte. Quem consome pode fatiar por heading com
confiança.

#### O que ele *não* faz

- **Não devolve as imagens.** Só o Markdown. Se você precisa das figuras,
  extraia-as do PDF por outro caminho.
- **Não devolve coordenadas nem número de página.** A resposta é texto
  corrido; se você precisa de "onde na página", esta rota não serve.
- **Não decide se o documento é bom.** Um PDF digitalizado torto converte, e
  converte mal.
- **Não classifica nem resume.** Metadados (título, autores, DOI) você extrai
  do Markdown do seu lado — é o que o PubliBot faz.

#### PDF digitalizado: OCR

PDF **sem camada de texto** — o que sai de um scanner — precisa de OCR, e ele
vem **desligado por padrão** (`DOCLING_OCR=false`). Desligado, uma página
digitalizada converte para quase nada, sem erro.

Se o seu acervo tem digitalizações, o dono da máquina liga `DOCLING_OCR=true`
no `.env` do worker. Confira em `/health/`:

```bash
curl -s http://<worker>:8090/health/ | jq .conversao
# {"dispositivo": "auto", "ocr": false, "threads": "auto", "carregado": true}
```

O worker **recusa subir** com OCR ligado e o motor ausente, em vez de falhar só
na primeira digitalização — o caso raro, que é justamente quando ninguém está
olhando.

#### Tempo, GPU e o primeiro pedido

A análise de layout **não precisa de GPU**: a placa muda o tempo, não o
resultado. Dá para operar a rota numa máquina sem placa nenhuma.

Duas coisas afetam o relógio do seu lado:

- **o primeiro pedido carrega os modelos** — dezenas de segundos a mais, uma
  vez por reinício do worker. Depois disso o conversor fica residente;
- **a conversão disputa o mesmo lock** que texto e imagem. Um PDF grande
  chegando enquanto uma imagem é gerada leva `503`, como qualquer outra rota.

Por isso o timeout sugerido é 600 s, e por isso um PDF não deve ser convertido
dentro de uma requisição web do seu sistema: é trabalho de fila.

#### Limites e integridade

- **Tamanho:** acima de `MAX_PDF_BYTES` (padrão 100 MB) a resposta é `413`.
- **Formato:** a rota é para PDF. Outros formatos que o Docling aceita não
  estão expostos aqui.
- **Mande o `X-Expected-Sha256`.** Sem ele, um arquivo truncado no caminho é
  convertido em silêncio, e o Markdown de um documento que ninguém pediu entra
  no seu acervo. Com ele, a resposta é `422` e você sabe que precisa reenviar.
  O `sha256` volta na resposta de qualquer forma — guarde-o: é como você
  descobre depois que dois documentos do acervo são o mesmo arquivo.

### Saúde

`GET /health/`, sem credencial. Use para diagnóstico, não como teste de
disponibilidade antes de cada pedido: entre o `/health/` e o pedido a placa
pode ter sido tomada, e você teria feito duas viagens para chegar no mesmo
503.

Resposta: `contrato/saude-resposta.json`.

**Ele nunca devolve 500.** Um bloco que falha ao ser coletado vira
`{"erro": "<tipo>: <mensagem>"}` no lugar do bloco, com HTTP 200 — o processo
está de pé, e é isso que o 200 afirma. Um diagnóstico que estoura não serve
para descobrir o que estourou. Trate um bloco com `erro` como "esta parte está
doente", e não como "o worker caiu".

Dois campos valem alarme do seu lado:

- **`ha_segundos` acima de ~120 com `ocupante: "texto"`** — trabalho preso. O
  lock não tem watchdog: se a chamada ao Ollama pendurar, ele fica retido até
  `OLLAMA_TIMEOUT` e todo mundo leva 503 nesse tempo;
- **um bloco com `erro`** — a máquina está degradada de um jeito que os 200 não
  denunciam.

## Mantendo o contrato honesto

`contrato/*.json` são os exemplos de resposta. **Copie-os para os testes do
seu cliente** e exercite o seu adaptador contra eles.

Isso pega a divergência dos dois lados: aqui, um teste confere que a resposta
real ainda tem aquela forma; no seu repositório, que o seu código ainda lê
aquela forma.

O que não pega é a sua cópia envelhecer. Por isso todo exemplo tem
`_contrato_versao`, e o worker publica a dele:

```bash
curl -s http://<worker>:8090/health/ | jq -r .contrato_versao
```

Compare de vez em quando — ou num teste, se o seu CI alcança o worker. Versão
maior diferente significa que algo mudou de forma incompatível, e há uma seção
nova neste arquivo explicando o quê.

Acréscimos e um ajuste de cabeçalho. Nada some e nada muda de tipo, mas dois
pontos merecem olhada:

- **`error.code == "timeout"` agora acontece em `/v1/chat/completions`.** Antes
  era documentado e inalcançável: um pedido que estourava o orçamento chegava
  como `ollama_indisponivel`, indistinguível de "o Ollama caiu". Se você já
  trata o código, não precisa fazer nada — ele só passou a aparecer;
- **o 503 de código `timeout` não traz mais `Retry-After`.** Se você lê o
  cabeçalho com acesso direto (`headers["Retry-After"]`), troque por
  `headers.get("Retry-After", 60)`. Afeta também `/v1/images/generations`, onde
  esse código já ocorria;
- **um 5xx vindo do Ollama agora sai envelopado** como `ollama_indisponivel`,
  com `Retry-After`. Antes ele era repassado cru, sem `error.code` e sem
  cabeçalho — a única forma de 503 que saía daqui sem nada para decidir;
- **`/health/` ganhou `ollama.carregados_detalhe`** e passou a nunca responder
  500. `ollama.carregados` continua sendo a lista de nomes, intocada;
## O que mudou na 2.3

Só a rota de imagem, e é sobre qualidade:

- **o tamanho padrão passou de `1024x576` para `1344x768`**, que é a proporção
  16:9 que o SDXL conhece do treino. Afeta você só se você **omite** `size`;
- **o teto por lado subiu de 1024 para 1344**, para a proporção acima ser
  pedível. Nada que passava antes deixou de passar;
- **o prompt negativo do worker estava em português e não fazia nada.** Agora
  está em inglês. Veja o aviso sobre o **seu** prompt na seção de imagem;
- `/health/` publica `imagem.vae`, `imagem.amostrador`, `imagem.guidance` e
  `imagem.lado_maximo`, que são os ajustes que decidem qualidade.

## O que mudou na 2.2

- **modelo ausente passou a ser baixado sozinho**, em segundo plano e sem
  tomar a placa. O pedido que disparou recebe `503 baixando_modelo`; um
  download que falha vira `404`, para o cliente desistir em vez de reagendar.
  Código novo — se o seu cliente trata código desconhecido como adiável com
  teto (como este guia manda), ele já funciona sem mudança;
- **`/health/` ganhou `ollama.baixando`**, com o progresso de cada download.

## O que mudou na 2.1

- **`options` passou a funcionar.** `options.num_ctx` e `options.num_predict`
  eram descartados em silêncio pela camada compatível do Ollama; agora um
  pedido que traz `options` (ou `keep_alive`) é roteado pelo dialeto nativo e
  traduzido de volta. A forma da resposta não muda, e quem não manda `options`
  segue pelo caminho antigo sem diferença nenhuma;
- **`OLLAMA_KEEP_ALIVE` (a variável do worker) foi removido.** Ele era injetado
  no corpo e descartado pela camada compatível: parecia ativo e não tinha
  efeito nenhum. Um `keep_alive` que **você** mandar no pedido é honrado, porque
  ele leva o pedido pelo nativo; a política da máquina se ajusta no Ollama.

## Checklist

- [ ] URL base aponta para o worker, não para o Ollama
- [ ] `Authorization: Bearer` em toda chamada (menos `/health/`)
- [ ] 503 **adia**, não falha, e não consome tentativa
- [ ] `Retry-After` é lido com `.get(..., 60)`, e tratado como piso
- [ ] **Teto de adiamentos**, senão um pedido impossível gira para sempre
- [ ] `error.code == "timeout"` não é repetido igual
- [ ] `error.code` desconhecido **adia com teto**, nunca falha definitiva
- [ ] `baixando_modelo` adia; o **404** que vem depois dele faz o trabalho falhar
- [ ] 400/404/422 **falham**, não adiam
- [ ] `ConnectionError` (restart do worker) entra no mesmo teto
- [ ] Sem `stream: true`
- [ ] Timeout do cliente **maior** que `OLLAMA_TIMEOUT`, não igual
- [ ] Janela de contexto por pedido em `options.num_ctx` (não precisa de Modelfile)
- [ ] Truncamento monitorado por `usage.prompt_tokens`
- [ ] Exemplos de `contrato/` copiados para os seus testes
- [ ] Reagendar, nunca `sleep`
- [ ] Se usa modelo por cliente: afinidade é dica, e com limite de sequência
