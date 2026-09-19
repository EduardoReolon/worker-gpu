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

| O que veio | O que significa |
|---|---|
| `401` | segredo errado, ou faltou o cabeçalho |
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

| `error.code` | Significa | O que fazer |
|---|---|---|
| `gpu_ocupada` | outro trabalho está na placa | reagendar para daqui a `Retry-After` |
| `ollama_indisponivel` | o Ollama caiu ou reinicia | idem; se persistir por horas, alertar |
| `sem_vram` | a placa não tem espaço agora | idem, com paciência maior |
| `timeout` | o trabalho passou do orçamento | **não repita igual** — reduza o pedido |

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
resposta = requests.post(f"{WORKER}/v1/chat/completions", json=corpo,
                         headers={"Authorization": f"Bearer {SEGREDO}"},
                         timeout=600)

if resposta.status_code == 503:
    dados = resposta.json().get("error", {})
    espera = int(resposta.headers.get("Retry-After", 60))

    if dados.get("code") == "timeout":
        # Repetir igual daria o mesmo resultado. Reduza antes.
        raise TrabalhoPrecisaDeAjuste(dados.get("message", ""))

    # Adiar NAO gasta tentativa: nada deu errado, so nao era a hora.
    raise Adiar(dados.get("message", ""), tentar_em_segundos=espera)

resposta.raise_for_status()
```

Se o seu sistema não tem orquestrador, o mínimo aceitável é reagendar a tarefa
para `agora + Retry-After` e sair — **nunca** um `sleep` segurando o processo:
a placa pode estar ocupada por minutos, e você prende um worker inteiro
esperando.

## Modelo por cliente, e afinidade

**O modelo é de quem pede.** O worker repassa o corpo ao Ollama praticamente
intacto — o `model` que você mandar é o que roda. Um CRM com modelo por tenant
e um padrão de sistema funciona sem nada especial aqui: o worker arbitra a
placa, não a escolha.

Dois limites honestos: o modelo precisa existir no Ollama da máquina
(`GET /v1/models` lista), e trocar de modelo **custa**. Com
`OLLAMA_MAX_LOADED_MODELS=1` — a configuração recomendada numa placa só — um
modelo diferente expulsa o que estava carregado. Não é falha, é tempo: o
próximo pedido espera o carregamento.

### Como saber o que está na placa

Três formas, da mais barata para a mais cara:

| De onde | O que diz | Custa |
|---|---|---|
| um **200** seu | o seu modelo é o que está carregado agora | nada |
| o **503** que você levou | `error.modelo` — o que está em uso | nada |
| `GET /health/` | `modelo` (em uso) e `ollama.carregados` (residentes) | uma viagem |

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

| Rota | Timeout sugerido |
|---|---|
| `/v1/chat/completions` | 600 s |
| `/v1/images/generations` | 300 s |
| `/parse/` | 600 s |
| `/health/` | 10 s |

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

## Checklist

- [ ] URL base aponta para o worker, não para o Ollama
- [ ] `Authorization: Bearer` em toda chamada (menos `/health/`)
- [ ] 503 **adia**, não falha, e não consome tentativa
- [ ] `Retry-After` é respeitado
- [ ] `error.code == "timeout"` não é repetido igual
- [ ] Sem `stream: true`
- [ ] Timeouts generosos
- [ ] Exemplos de `contrato/` copiados para os seus testes
- [ ] Reagendar, nunca `sleep`
- [ ] Se usa modelo por cliente: afinidade é dica, e com limite de sequência
