# Como gravar a sua voz para o Qwen3-TTS

Guia para clonar a sua voz no worker (`qwen:<nome>`) e usá-la na narração dos
vídeos do publi-bot. A instalação está em `INTEGRACAO.md`, seção **Voz**.

## O que esperar

- O resultado sai **parecido o bastante para reconhecer que é você**, mas não
  idêntico. O modelo imita a partir de poucos segundos, sem treinar na sua
  voz. O ElevenLabs no modo "Professional" (treinado com 30 min ou mais) ainda
  imita melhor.
- **A transcrição exata da gravação faz muita diferença** na semelhança. Sem
  ela funciona, mas imita menos.
- **A gravação pesa mais que o modelo.** Ruído, eco e música de fundo vão
  parar na voz gerada.
- Falas muito longas às vezes saem com uma emoção exagerada ou estranha. O
  worker já corta o texto em trechos de ~300 caracteres para evitar isso.

## A regra principal: o modelo copia o *jeito* de falar

O modelo de clonagem não recebe instruções de tom, como "fale animado". Ele
imita o **tom, o ritmo e a energia** da gravação. Se você gravar calmo, toda
narração sai calma.

**Para cada estilo que quiser usar, grave uma voz separada** (veja
[Vários estilos](#vários-estilos-uma-voz-para-cada)).

## Como gravar

- **Duração:** de 10 a 15 s de fala contínua. O worker usa no máximo 20 s; o
  mínimo é 3 s.
- **Lugar:** um cômodo pequeno com cortina, sofá ou roupas absorve eco. Dentro
  do carro parado ou de um armário de roupas funciona muito bem. Evite
  banheiro, cozinha e sala vazia.
- **Silêncio:** sem ventilador, ar-condicionado, TV ou música. Nenhuma outra
  voz.
- **Microfone:** o celular serve. Fique a um palmo de distância, um pouco de
  lado para o sopro não bater no microfone.
- **Volume:** constante, sem gritar. A energia tem que vir da entonação, não
  do volume.
- **Tentativas:** grave umas 3 vezes e fique com a mais natural.
- **Corte** o silêncio longo do começo e do fim.

## Como falar

- **Sorria enquanto fala.** Dá para ouvir, e é o que deixa a voz simpática e
  "vendedora".
- **Destaque as palavras-chave** ("poucos minutos", "especial", "só essa
  semana").
- **Varie:** pergunta, afirmação e chamada no mesmo texto, para o modelo
  aprender as variações de entonação.
- **Tom de quem recomenda algo para um amigo**, não de locutor de rádio.
  Exagero soa falso, e o modelo copia o exagero.
- **Não leia "de cabeça baixa".** Decore o texto ou deixe-o na altura dos
  olhos: lendo, a voz fica monótona.

## Vários estilos: uma voz para cada

Cada voz é criada com o seu próprio texto e tom. No publi-bot, cada conta ou
vídeo escolhe a sua (`qwen:eduardo-vendas`, `qwen:eduardo-calmo`...).

### `eduardo-vendas`: divulgação, oferta (o principal)

Energia alta, sorrindo, um pouco mais rápido que conversa normal, pausa curta
antes da chamada final.

> Você sabia que dá pra resolver isso em poucos minutos? Pois é! A gente
> preparou uma condição especial, só essa semana. Corre, chama no WhatsApp e
> garante o seu antes que acabe!

### `eduardo-calmo`: explicativo, dicas, conteúdo

Ritmo de conversa, tranquilo e seguro, como explicando para um cliente na
frente de você.

> Deixa eu te explicar de um jeito simples. Quando a gente faz do jeito certo
> desde o começo, o resultado dura muito mais. E o melhor: sai mais barato no
> final. Faz sentido, né?

### `eduardo-urgente`: promoção relâmpago, últimas vagas

Mais rápido e mais intenso que o de vendas, mas sem gritar. Ênfase nos números
e no prazo.

> Atenção! São só as últimas unidades, e o desconto acaba hoje à meia-noite.
> Não deixa pra depois, porque depois não tem mais. Clica agora e garante já!

### `eduardo-depoimento`: história, bastidores, confiança

Mais baixo e próximo, como contando algo pessoal. Pausas naturais.

> Quando eu comecei, eu não fazia ideia de que ia chegar até aqui. Foi muito
> trabalho, muito erro, mas cada cliente satisfeito valeu a pena. E hoje eu
> quero dividir isso com você.

Comece pelo **vendas**. Os outros só se você sentir falta.

## Converter o arquivo

O worker lê WAV, FLAC, OGG e MP3. **O m4a do gravador do celular não
funciona.** Converta antes:

```bash
ffmpeg -i gravacao.m4a -ac 1 -ar 24000 minha_voz.wav
```

Para cortar o silêncio do começo e do fim no mesmo passo (aqui, começando em
0,5 s e terminando em 14 s):

```bash
ffmpeg -i gravacao.m4a -ss 0.5 -to 14 -ac 1 -ar 24000 minha_voz.wav
```

## Criar a voz

Passe **exatamente** o que você disse, inclusive se trocou alguma palavra na
hora:

```bash
./deploy/criar_voz.sh eduardo-vendas --gravacao vendas.wav "Você sabia que dá pra resolver isso em poucos minutos? Pois é! A gente preparou uma condição especial, só essa semana. Corre, chama no WhatsApp e garante o seu antes que acabe!"
```

O script espera a vez na placa e grava `voz/teste-eduardo-vendas.mp3`. Ouça.
Não gostou? Grave de novo e troque:

```bash
./deploy/criar_voz.sh eduardo-vendas --gravacao vendas2.wav "o texto exato" --substituir
```

Para ver as vozes que existem:

```bash
curl -s -H "Authorization: Bearer $(grep ^WORKER_SHARED_SECRET .env | cut -d= -f2)" \
    http://127.0.0.1:8090/v1/audio/voices | jq -r '.data[].id'
```

No publi-bot, a voz da narração da conta: `qwen:eduardo-vendas`.

## Entonação gravada, timbre de outra voz (conversão)

Outro uso da mesma voz: você grava a fala da cena **com a entonação certa**, e
o worker troca só o **timbre** (`/v1/audio/convert`). As palavras, o ritmo e as
pausas ficam os seus.

- **Na voz da Carla:** converta para `qwen:carla`.
- **Na sua própria voz, com qualidade de estúdio:** converta para a sua voz
  boa (`qwen:eduardo`). A fala pode ser gravada às pressas, no celular ou no
  navegador: o áudio que sai é recriado a partir da referência, e não é a
  gravação limpa.

Para isso funcionar bem:

- **A referência é que precisa de qualidade** (a `referencia.wav` da voz
  alvo). Grave-a uma vez, com cuidado, seguindo este guia. O conversor usa
  os primeiros 10 s dela.
- **A fala da cena pode ser simples, mas não ruim demais.** Microfone do
  celular serve; ruído forte, eco grande, outra voz por cima ou som estourado
  fazem o conversor errar palavras.
- **Na conversão, o tom vem da fala da cena, não da referência.** Então a
  referência da sua voz pode ser mais neutra que a de `eduardo-vendas`.

**Ainda não testado na sua máquina:** o conversor atual (o do Chatterbox)
segura bem as palavras e o ritmo. A subida e a descida da voz ele recria em
parte. Se a entonação se perder, há um conversor que segue o tom da gravação à
risca (o Seed-VC), que pode entrar como alternativa.

## Se não ficou bom

| Sintoma | Provável causa | O que fazer |
|---|---|---|
| Não parece você | transcrição errada ou ausente | confira o texto palavra por palavra |
| Eco ou "som de lata" | cômodo grande ou microfone longe | grave no carro ou no armário, a um palmo |
| Chiado ou ruído de fundo | ventilador, rua, ar-condicionado | desligue tudo e grave de novo |
| Voz monótona | gravação lida, sem energia | decore o texto e sorria |
| Voz exagerada, teatral | gravação exagerada | baixe um tom: amigo recomendando |
| Erro "menos de 3 s" | arquivo curto ou cortado demais | grave de 10 a 15 s |
| Erro "nao consegui ler a gravacao" | formato m4a ou outro | converta com o `ffmpeg` acima |

## Autorização

Só clone a voz de quem autorizou. Para a voz de outra pessoa (um sócio, um
locutor), peça autorização por escrito.
