"""Todo o ambiente do worker num lugar so.

Estava espalhado por dois modulos, cada um lendo `os.environ` no topo. Com
tres consumidores e um arbitro no meio, isso deixa de caber: uma variavel com
dois donos e uma variavel que diverge.

As constantes sao lidas na IMPORTACAO, e nao a cada uso. E deliberado — e o que
faz a unit do systemd valer alguma coisa: o que esta no `.env` no momento em
que o servico sobe e o que vale ate ele reiniciar. Trocar uma variavel e
reiniciar, sempre; nunca "trocar e torcer".
"""

from __future__ import annotations

import os


def _booleano(nome: str, padrao: bool = False) -> bool:
    bruto = os.environ.get(nome, "sim" if padrao else "nao").strip().lower()
    return bruto in {"1", "true", "yes", "sim", "on"}


def _inteiro(nome: str, padrao: int) -> int:
    try:
        return int(os.environ.get(nome, padrao))
    except ValueError:
        return padrao


def _decimal(nome: str, padrao: float) -> float:
    try:
        return float(os.environ.get(nome, padrao))
    except ValueError:
        return padrao


# ---------------------------------------------------------------------------
# Rede e credencial
# ---------------------------------------------------------------------------
BIND_HOST = os.environ.get("BIND_HOST", "127.0.0.1")
BIND_PORT = _inteiro("BIND_PORT", 8090)

# Um segredo para todas as rotas. Nao ha usuarios aqui: quem alcanca a porta ou
# tem o segredo, ou nao entra.
SEGREDO = os.environ.get("WORKER_SHARED_SECRET", "")


# ---------------------------------------------------------------------------
# Ollama (texto)
# ---------------------------------------------------------------------------
# O worker e o unico que fala com o Ollama. Ele escuta em loopback e NAO deve
# estar exposto na rede — quem publica e o worker, que arbitra.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

# Orcamento de tempo de UM pedido de texto, em segundos.
#
# 540 e nao 600, e a diferenca e o ponto: o `INTEGRACAO.md` sugere 600s de
# timeout no cliente. Iguais, os dois relogios disparam juntos — e o do cliente
# comeca antes, porque o daqui so parte depois de o pedido chegar, passar pela
# credencial e tomar o lock. O cliente desistia primeiro e abandonava um
# trabalho que ainda estava segurando a placa, que e exatamente o que a
# documentacao diz querer evitar. Com 540 quem desiste primeiro e o worker, e
# ele desiste devolvendo um 503 legivel em vez de um socket cortado.
OLLAMA_TIMEOUT = _decimal("OLLAMA_TIMEOUT", 540.0)

# Janela de contexto do modelo de texto, em tokens (`options.num_ctx`).
#
# O Ollama carrega com 4096 por padrao, e um pedido maior volta 400 —
# planejar um artigo manda ~5 mil tokens. O cliente diz quanto precisa no
# cabecalho `X-PubliBot-Contexto`; sem ele vale o PADRAO. O worker nunca baixa
# a janela de um pedido para o outro (cada troca recarrega o modelo) e nunca
# passa do MAXIMO, que e o que esta maquina aguenta: acima da VRAM o Ollama
# poe camadas na CPU, e a geracao fica muito mais lenta.
#
# 0 no PADRAO desliga: pedido sem cabecalho vai sem `num_ctx`, como antes.
CONTEXTO_PADRAO = _inteiro("CONTEXTO_PADRAO", 16384)
CONTEXTO_MAXIMO = _inteiro("CONTEXTO_MAXIMO", 32768)

# Baixar sozinho um modelo pedido que nao esta no disco.
#
# Ligado: com um modelo por inquilino, lembrar de dar `ollama pull` em cada um
# a cada maquina nova nao escala, e esquecer aparece como 404 no meio de um
# lote. O worker baixa EM SEGUNDO PLANO, sem tomar o lock da GPU — um download
# e rede e disco, e prender a placa por dez minutos enquanto baixa seria pior
# que o problema. O pedido que disparou o download leva 503 `baixando_modelo`,
# que os clientes ja sabem adiar.
#
# Desligue numa maquina com disco apertado ou sem saida para a internet: um
# nome de modelo errado que por acaso exista no registro do Ollama baixa
# gigabytes que ninguem pediu. Desligado, o comportamento e o de antes — o
# Ollama responde 404 e o cliente desiste.
BAIXAR_MODELO_AUTOMATICO = _booleano("BAIXAR_MODELO_AUTOMATICO", True)

# Teto para UM download, em segundos. Generoso de proposito: um modelo grande
# numa conexao domestica passa de meia hora, e desistir no meio joga fora o que
# ja veio.
OLLAMA_PULL_TIMEOUT = _decimal("OLLAMA_PULL_TIMEOUT", 3600.0)

# Por quanto tempo um download que FALHOU e lembrado, em segundos.
#
# Ele existe contra o laco: sem memoria da falha, cada retentativa do cliente
# dispara um download novo do mesmo modelo que nao existe, para sempre. Com
# memoria eterna, uma queda de rede exigiria reiniciar o servico. Lembrar por
# alguns minutos faz o cliente desistir (404) e ainda assim deixa a maquina se
# curar sozinha depois.
FALHA_DE_DOWNLOAD_LEMBRADA = _inteiro("FALHA_DE_DOWNLOAD_LEMBRADA", 600)

# Descarregar o modelo de texto da VRAM antes de um trabalho de imagem.
#
# E o ponto do arbitro existir. Numa placa de 8 GB um modelo de texto grande
# ocupa quase tudo, e a difusao precisa de ~5,4 GB — nao cabem juntos. Como
# TODO pedido passa por aqui, descarregar e seguro: quem detem o lock sabe que
# ninguem esta gerando texto naquele instante.
OLLAMA_DESCARREGAR_PARA_IMAGEM = _booleano("OLLAMA_DESCARREGAR_PARA_IMAGEM", True)

# Havia aqui um OLLAMA_KEEP_ALIVE, injetado no corpo de cada pedido de texto.
# Foi REMOVIDO: `keep_alive` nao e campo do dialeto da OpenAI, e o proxy fala
# esse dialeto — a camada compativel do Ollama descartava a chave na
# desserializacao. A variavel existia, aparecia no `.env`, e nao tinha efeito
# nenhum. Configuracao que mente e pior que configuracao ausente, porque
# alguem confia nela.
#
# Enquanto o proxy nao falar `/api/chat`, a politica de memoria desta maquina
# se ajusta no Ollama (OLLAMA_KEEP_ALIVE no ambiente DELE, ou `PARAMETER` num
# Modelfile), e nao aqui.


# ---------------------------------------------------------------------------
# Imagem (difusao)
# ---------------------------------------------------------------------------
IMAGEM_ATIVA = _booleano("IMAGEM_ATIVA", True)
IMAGEM_MODELO = os.environ.get("IMAGEM_MODELO", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGEM_DEVICE = os.environ.get("IMAGEM_DEVICE", "auto").lower()
IMAGEM_PASSOS = _inteiro("IMAGEM_PASSOS", 30)
IMAGEM_GUIDANCE = _decimal("IMAGEM_GUIDANCE", 6.0)

# Nao ha prompt negativo aqui, e a ausencia e deliberada. Houve um padrao
# ("text, letters, words, watermark, ...") aplicado a TODO pedido, e ele
# brigava com quem pedia justamente texto — uma planilha com linhas rotuladas
# saia como sopa de letras. O worker executa; quem sabe o que a imagem precisa
# e o cliente, e ele manda `negative_prompt` no pedido quando quiser um.


# Precisao dos pesos na placa: `float16` ou `bfloat16`. Em CPU e sempre
# float32.
#
# float16 e o do SDXL (com o VAE ajustado, `IMAGEM_VAE`). Os modelos novos
# (Z-Image, FLUX, SD 3.5) sao publicados e treinados em bfloat16, e em float16
# alguns estouram — imagem preta ou ruido, sem erro. A RTX 30xx tem bfloat16.
IMAGEM_DTYPE = os.environ.get("IMAGEM_DTYPE", "float16").strip().lower()

# Quantizar os pesos ao carregar, com bitsandbytes: `nao`, `4bit` ou `8bit`.
# So vale em GPU.
#
# E o que faz um modelo de 6B caber numa placa de 8 GB: o Z-Image-Turbo em
# bfloat16 sao ~20 GB (transformer 12 + codificador de texto 8); em 4 bits,
# ~7 GB de RAM e ~5 GB de pico na placa. A perda de qualidade em 4 bits (NF4)
# e pequena perto do salto de modelo. Precisa de `pip install bitsandbytes`.
IMAGEM_QUANTIZAR = os.environ.get("IMAGEM_QUANTIZAR", "nao").strip().lower()

# Quais componentes do pipeline quantizar. Os nomes sao os do `model_index.json`
# do modelo; o padrao serve ao Z-Image e ao FLUX (no FLUX o T5 e
# `text_encoder_2`). O VAE fica de fora: e pequeno e sensivel.
_COMPONENTES = os.environ.get("IMAGEM_QUANTIZAR_COMPONENTES", "transformer,text_encoder")
IMAGEM_QUANTIZAR_COMPONENTES = [nome.strip() for nome in _COMPONENTES.split(",") if nome.strip()]

# VAE alternativo, por nome de repositorio no HuggingFace. Vazio = o do modelo.
#
# Existe por um defeito conhecido e visivel: o VAE que vem no SDXL 1.0
# **estoura em float16**. O servico carrega em float16 na placa (e precisa:
# float32 nao cabe), e o resultado sao manchas, faixas de cor e, em alguns
# casos, imagem preta — defeitos que a pessoa olhando chama de "cara de IA"
# sem saber apontar o que e.
#
# Para qualquer modelo da familia SDXL, o conserto e este:
#
#     IMAGEM_VAE=madebyollin/sdxl-vae-fp16-fix
#
# NAO e o padrao aqui porque `IMAGEM_MODELO` pode ser de outra familia (SD 1.5,
# SD 3.5, FLUX), e um VAE de SDXL nelas nao encaixa. O worker AVISA no log
# quando ve um modelo SDXL sem VAE ajustado.
IMAGEM_VAE = os.environ.get("IMAGEM_VAE", "")

# Amostrador. Vazio = o que vem no modelo (para o SDXL, o Euler do diffusers).
#
# O amostrador decide como os passos caminham do ruido para a imagem, e com
# pouco passo a escolha aparece: `dpm++2m_karras` costuma dar em 30 passos o
# que o Euler da em 50. Nomes aceitos em `imagem.SCHEDULERS`.
IMAGEM_SCHEDULER = os.environ.get("IMAGEM_SCHEDULER", "").strip().lower()
IMAGEM_OCIOSO_SEGUNDOS = _inteiro("IMAGEM_OCIOSO_SEGUNDOS", 300)
IMAGEM_MAXIMO = _inteiro("IMAGEM_MAXIMO", 4)
# Lado maximo aceito num pedido.
#
# 1536 e nao 1024, e a diferenca importa para a qualidade: o SDXL foi treinado
# em recortes de cerca de 1024x1024 PIXELS DE AREA, distribuidos numa grade de
# proporcoes fixas (veja `imagem.GRADE_DO_SDXL`). A mais larga delas e
# 1536x640, e com o teto em 1024 nao havia como nem pedir as largas — o pedido
# voltava 422. Gerar fora da grade nao da erro: da assunto duplicado,
# geometria torta e composicao incoerente.
#
# Este teto e por LADO. Quem limita o custo e o `IMAGEM_AREA_MAXIMA_MP`
# abaixo, e os dois precisam existir: sozinho, um teto de lado em 1536
# deixaria passar 1536x1536, que e o dobro da area de treino.
IMAGEM_LADO_MAXIMO = _inteiro("IMAGEM_LADO_MAXIMO", 1536)

# Area maxima de UMA imagem, em megapixels.
#
# E o teto que protege a placa e a qualidade ao mesmo tempo, e ele existe
# porque o teto por lado nao protege nenhuma das duas: toda a grade de treino
# do SDXL fica em torno de 1,05 MP, e 1536x1536 passaria pelo teto de lado com
# 2,36 MP — o dobro da area de treino e VRAM que esta placa nao tem.
#
# 1,2 deixa a grade inteira passar com folga e barra o que esta claramente
# fora. Suba so depois de medir:
#     ./venv/bin/python bancada.py --tamanhos 1344x768,1536x640
IMAGEM_AREA_MAXIMA_MP = _decimal("IMAGEM_AREA_MAXIMA_MP", 1.2)

# Gerar em CPU quando a VRAM nao couber. Desligado: medido em uso, um lote em
# CPU consumiu horas de processador e 17 GB de RAM (float32), com a maquina
# inutilizavel. Recusar devolve o trabalho para a fila do cliente, que sabe
# esperar.
IMAGEM_PERMITIR_CPU = _booleano("IMAGEM_PERMITIR_CPU", False)

# Teto por geracao. Sem ele nao ha como interromper um laco de difusao, e ate
# um `systemctl restart` fica preso esperando.
IMAGEM_TEMPO_MAXIMO = _inteiro("IMAGEM_TEMPO_MAXIMO", 600)

# Prazo DURO de um trabalho de imagem (carga do modelo + geracao), em
# segundos. 0 desliga.
#
# O `IMAGEM_TEMPO_MAXIMO` so e conferido ENTRE passos da difusao. Um processo
# travado — na carga, que nao tem passo nenhum, ou dentro de um passo — nunca
# chega a conferir, e segurava a placa (e o texto de todos) por tempo
# indefinido. Foi o que aconteceu com o worker estrangulado no `MemoryHigh`.
#
# Estourado este prazo, o worker considera-se QUEBRADO: responde 500
# `worker_travado` e se mata, para o systemd subir um processo limpo em 10s.
# Nao ha como cancelar a thread travada, e um processo que continua com ela
# pendurada nao e confiavel.
#
# Precisa ser maior que o `IMAGEM_TEMPO_MAXIMO`: quem desiste primeiro de uma
# geracao lenta mas saudavel e ele, com um 503 `timeout`. E a carga do
# Z-Image leva de 1 a 3 minutos.
IMAGEM_TEMPO_TRAVADO = _inteiro("IMAGEM_TEMPO_TRAVADO", 900)

# --- Edicao: referencias de personagem e foto na cena (/v1/images/edits) ----
# Dois modelos, escolhidos POR PEDIDO:
#
# - sem imagem de referencia, vale o IMAGEM_MODELO de cima (o de mais
#   qualidade, hoje o Z-Image) — inclusive para pintar so a parte da mascara;
# - com referencia (os personagens), vale o IMAGEM_REFERENCIA_MODELO, que
#   precisa aceitar IP-Adapter. O Z-Image nao aceita: o IP-Adapter so existe
#   para o SDXL e familia. O RealVisXL e o SDXL de foto mais realista.
#
# Um so fica carregado por vez (a placa e de 8 GB): alternar entre os dois
# custa uma recarga. Vazio desliga as referencias: elas passam a ser
# ignoradas, com aviso na resposta.
IMAGEM_EDICAO_ATIVA = _booleano("IMAGEM_EDICAO_ATIVA", True)
IMAGEM_REFERENCIA_MODELO = os.environ.get(
    "IMAGEM_REFERENCIA_MODELO", "SG161222/RealVisXL_V5.0"
).strip()
IMAGEM_REFERENCIA_DTYPE = os.environ.get("IMAGEM_REFERENCIA_DTYPE", "float16").strip().lower()
IMAGEM_REFERENCIA_VAE = os.environ.get(
    "IMAGEM_REFERENCIA_VAE", "madebyollin/sdxl-vae-fp16-fix"
).strip()
IMAGEM_REFERENCIA_SCHEDULER = (
    os.environ.get("IMAGEM_REFERENCIA_SCHEDULER", "dpm++2m_karras").strip().lower()
)
IMAGEM_REFERENCIA_PASSOS = _inteiro("IMAGEM_REFERENCIA_PASSOS", 30)
IMAGEM_REFERENCIA_GUIDANCE = _decimal("IMAGEM_REFERENCIA_GUIDANCE", 6.0)

# O IP-Adapter: repositorio, subpasta, arquivo de pesos e o codificador de
# imagem. O "plus" com o ViT-H copia a aparencia (rosto, roupa) melhor que o
# basico, que copia mais o estilo.
IMAGEM_IP_ADAPTER = os.environ.get("IMAGEM_IP_ADAPTER", "h94/IP-Adapter").strip()
IMAGEM_IP_ADAPTER_SUBPASTA = os.environ.get("IMAGEM_IP_ADAPTER_SUBPASTA", "sdxl_models").strip()
IMAGEM_IP_ADAPTER_PESO = os.environ.get(
    "IMAGEM_IP_ADAPTER_PESO", "ip-adapter-plus_sdxl_vit-h.safetensors"
).strip()
IMAGEM_IP_ADAPTER_CODIFICADOR = os.environ.get(
    "IMAGEM_IP_ADAPTER_CODIFICADOR", "models/image_encoder"
).strip()
IMAGEM_REFERENCIAS_MAXIMAS = _inteiro("IMAGEM_REFERENCIAS_MAXIMAS", 3)


# ---------------------------------------------------------------------------
# Conversao de PDF (Docling)
# ---------------------------------------------------------------------------
CONVERSAO_ATIVA = _booleano("CONVERSAO_ATIVA", True)
DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "auto").lower()
DOCLING_THREADS = _inteiro("DOCLING_THREADS", 0)
DOCLING_OCR = _booleano("DOCLING_OCR", False)

# Devolver a memoria do conversor depois de um tempo sem pedido, em segundos.
# 0 desliga (o conversor fica residente, que era o comportamento anterior).
#
# O Docling ficava carregado PARA SEMPRE depois da primeira conversao — a rota
# de imagem tinha o seu `IMAGEM_OCIOSO_SEGUNDOS` e esta nao tinha nada. Sao
# cerca de 1 a 2 GB de memoria anonima parada, e memoria anonima parada e
# exatamente o que o kernel manda para o swap primeiro.
#
# Mais alto que o da imagem de proposito: recarregar o Docling custa dezenas de
# segundos, e um acervo de PDF costuma vir em lote — soltar entre dois arquivos
# do mesmo lote seria pagar a carga duas vezes por nada.
CONVERSAO_OCIOSO_SEGUNDOS = _inteiro("CONVERSAO_OCIOSO_SEGUNDOS", 900)
MAX_PDF_BYTES = _inteiro("MAX_PDF_BYTES", 100 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Transcricao de audio (faster-whisper)
# ---------------------------------------------------------------------------
TRANSCRICAO_ATIVA = _booleano("TRANSCRICAO_ATIVA", True)

# Nome de modelo do faster-whisper (`large-v3`, `large-v3-turbo`, `medium`...)
# ou um caminho local. `large-v3` em int8 ocupa ~3 GB de VRAM. O `turbo` tem o
# decodificador destilado: bem mais rapido, um pouco pior em portugues.
TRANSCRICAO_MODELO = os.environ.get("TRANSCRICAO_MODELO", "large-v3").strip()

# `auto` usa a placa se o CTranslate2 a enxergar. `cpu` funciona, e leva da
# ordem da duracao do audio.
TRANSCRICAO_DEVICE = os.environ.get("TRANSCRICAO_DEVICE", "auto").strip().lower()

# Tipo de computacao do CTranslate2 na placa. `int8_float16`: pesos em 8 bits,
# contas em float16 — metade da VRAM do float16 puro, sem perda que se ouca.
# Em CPU e sempre `int8`.
TRANSCRICAO_COMPUTACAO = os.environ.get("TRANSCRICAO_COMPUTACAO", "int8_float16").strip()

# Filtro de voz (VAD): pula o silencio antes de transcrever. Alem de poupar
# tempo, e o que evita o Whisper "alucinar" frases em trechos mudos.
TRANSCRICAO_VAD = _booleano("TRANSCRICAO_VAD", True)
TRANSCRICAO_BEAM = _inteiro("TRANSCRICAO_BEAM", 5)

# Devolver a memoria do modelo depois de ocioso. Recarregar custa ~10-20 s.
TRANSCRICAO_OCIOSO_SEGUNDOS = _inteiro("TRANSCRICAO_OCIOSO_SEGUNDOS", 600)

# Teto "macio", conferido entre segmentos: estourado, 503 `timeout`. Um audio
# de uma hora leva poucos minutos numa placa de 8 GB.
TRANSCRICAO_TEMPO_MAXIMO = _inteiro("TRANSCRICAO_TEMPO_MAXIMO", 1200)

# Prazo DURO (carga + transcricao), como o `IMAGEM_TEMPO_TRAVADO`: estourado,
# 500 `worker_travado` e o processo se encerra. Precisa ser maior que o macio
# e MENOR que o timeout do cliente (1800 s no PubliBot), para o 500 legivel
# chegar antes de ele desistir.
TRANSCRICAO_TEMPO_TRAVADO = _inteiro("TRANSCRICAO_TEMPO_TRAVADO", 1500)

MAX_AUDIO_BYTES = _inteiro("MAX_AUDIO_BYTES", 1024 * 1024 * 1024)

# O Whisper disputa a VRAM com o modelo de texto, como a difusao.
OLLAMA_DESCARREGAR_PARA_TRANSCRICAO = _booleano("OLLAMA_DESCARREGAR_PARA_TRANSCRICAO", True)


# ---------------------------------------------------------------------------
# Vetorizacao (embeddings)
# ---------------------------------------------------------------------------
VETORIZACAO_ATIVA = _booleano("VETORIZACAO_ATIVA", True)

# O MESMO modelo do servidor do PubliBot, e o nome e conferido em cada pedido:
# vetor de outro modelo no mesmo indice estraga a busca sem erro nenhum.
VETORIZACAO_MODELO = os.environ.get("VETORIZACAO_MODELO", "intfloat/multilingual-e5-large").strip()

# `auto` usa a placa se o torch a enxergar.
VETORIZACAO_DEVICE = os.environ.get("VETORIZACAO_DEVICE", "auto").strip().lower()

# float32 e nao float16, por fidelidade: o vetor daqui entra no MESMO indice
# que o do servidor (fastembed, ONNX, float32), e o teste de conformidade pede
# cosseno >= 0,999. Em float32 o e5-large ocupa ~2,2 GB de VRAM.
VETORIZACAO_DTYPE = os.environ.get("VETORIZACAO_DTYPE", "float32").strip().lower()

# Textos por lote na placa. 32 paragrafos de 512 tokens cabem com folga.
VETORIZACAO_LOTE = _inteiro("VETORIZACAO_LOTE", 32)

# Teto de textos por pedido. Um documento sao de 1 a algumas centenas.
VETORIZACAO_MAXIMO_TEXTOS = _inteiro("VETORIZACAO_MAXIMO_TEXTOS", 4096)

VETORIZACAO_OCIOSO_SEGUNDOS = _inteiro("VETORIZACAO_OCIOSO_SEGUNDOS", 600)

# Prazo DURO (carga + vetorizacao). Um documento leva segundos; o cliente
# espera 600 s, e o 500 legivel tem que chegar antes.
VETORIZACAO_TEMPO_TRAVADO = _inteiro("VETORIZACAO_TEMPO_TRAVADO", 300)

# Soltar o modelo de texto do Ollama antes de vetorizar: `sim`, `nao`, ou
# `se_faltar` — tenta ao lado dele e so descarrega se a VRAM nao couber.
# `se_faltar` e o padrao porque vetorizar leva segundos, e recarregar um 30B
# depois leva dezenas deles: descarregar a cada documento seria trocar um
# custo pequeno por um grande.
OLLAMA_DESCARREGAR_PARA_VETORIZACAO = (
    os.environ.get("OLLAMA_DESCARREGAR_PARA_VETORIZACAO", "se_faltar").strip().lower()
)


# ---------------------------------------------------------------------------
# YouTube (legenda)
# ---------------------------------------------------------------------------
# Nao usa a placa. Existe porque o YouTube recusa legenda para IP de nuvem, e
# esta maquina esta numa conexao residencial.
YOUTUBE_ATIVA = _booleano("YOUTUBE_ATIVA", True)

# Teto de cada ida ao YouTube, em segundos. A biblioteca nao tem nenhum: sem
# este, uma conexao que para de responder prende a requisicao para sempre.
YOUTUBE_TIMEOUT = _inteiro("YOUTUBE_TIMEOUT", 30)


# ---------------------------------------------------------------------------
# Animacao (imagem para video, pelo FramePack)
# ---------------------------------------------------------------------------
# Desligada de padrao: precisa do FramePack instalado a parte, no ambiente
# proprio dele (ver INTEGRACAO.md, "Animacao"). O worker so arbitra a placa e
# chama o `animar_framepack.py` com o Python daquele ambiente — as versoes que o
# FramePack pede (torch 2.6, Python 3.10) nao convivem com as daqui.
ANIMACAO_ATIVA = _booleano("ANIMACAO_ATIVA", False)
# Dentro da pasta do worker (e fora do git, ver .gitignore): uma pasta so para
# clonar, implantar e apagar. O `deploy/instalar_animacao.sh` instala aqui.
FRAMEPACK_DIR = os.path.expanduser(
    os.environ.get(
        "FRAMEPACK_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "framepack")
    )
)
ANIMACAO_PYTHON = os.path.expanduser(
    os.environ.get("ANIMACAO_PYTHON", os.path.join(FRAMEPACK_DIR, "venv", "bin", "python"))
)
# Lado maior do video, em px (o FramePack trabalha em 640). Menor = mais rapido;
# o Instagram mostra o video menor que isso. Multiplo de 32.
ANIMACAO_RESOLUCAO = _inteiro("ANIMACAO_RESOLUCAO", 640)
ANIMACAO_PASSOS = _inteiro("ANIMACAO_PASSOS", 25)
# TeaCache: bem mais rapido, um pouco menos fiel. Medido numa placa de 8 GB:
# ~9,5 min por trecho de ~1,1 s em 704x544 com ele ligado.
ANIMACAO_TEACACHE = _booleano("ANIMACAO_TEACACHE", True)
# Quanto da VRAM o FramePack deixa livre ao subir o modelo (GB). Maior = mais
# lento e mais folgado; em 8 GB, 6 funcionou (chegou a 7,7 GB).
ANIMACAO_MEMORIA_PRESERVADA = _decimal("ANIMACAO_MEMORIA_PRESERVADA", 6.0)

# GB do transformer do FramePack (~26 GB) fixados na memoria do processo. 0
# desliga.
#
# O transformer fica mapeado do disco e e lido inteiro a cada passo. Sem
# cache para ele inteiro, o kernel descarta sempre o trecho mais antigo — o
# primeiro que o passo seguinte le — e o modelo todo volta do disco a cada
# passo: GPU pela metade, esperando o disco. Fixar N GB faz so o resto
# (26 - N) ser relido.
#
# CUIDADO: medido, isto NAO resolveu. A parte fixa sai do mesmo `MemoryMax`
# que o cache, entao o que falta de cache (transformer + memoria propria -
# teto) nao muda; e o kernel ja relia ~9,5 GB por passo, nao o modelo
# inteiro. So subir o teto ou um transformer menor (fp8) diminuem a falta.
# Fica desligado; a opcao continua para experimentos.
ANIMACAO_RAM_FIXA_GB = _decimal("ANIMACAO_RAM_FIXA_GB", 0.0)
ANIMACAO_SEGUNDOS_MAXIMOS = _inteiro("ANIMACAO_SEGUNDOS_MAXIMOS", 5)
# Teto de um trabalho. Passou: o processo do FramePack e encerrado e o trabalho
# vira `failed` com `timeout`.
ANIMACAO_TEMPO_MAXIMO = _inteiro("ANIMACAO_TEMPO_MAXIMO", 3600)
# Por quanto tempo o video pronto fica para o cliente baixar.
ANIMACAO_GUARDAR_SEGUNDOS = _inteiro("ANIMACAO_GUARDAR_SEGUNDOS", 6 * 3600)


# ---------------------------------------------------------------------------
# Voz (texto para fala, pelo Kokoro)
# ---------------------------------------------------------------------------
# Desligada de padrao. O Kokoro (82M, Apache-2.0) roda na CPU, mais rapido que
# tempo real: nao toma a placa nem passa pelo arbitro. Mora num ambiente proprio
# (`voz/`, instalado pelo deploy/instalar_voz.sh), como o FramePack.
VOZ_ATIVA = _booleano("VOZ_ATIVA", False)
VOZ_DIR = os.path.expanduser(
    os.environ.get("VOZ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "voz"))
)
VOZ_PYTHON = os.path.expanduser(
    os.environ.get("VOZ_PYTHON", os.path.join(VOZ_DIR, "venv", "bin", "python"))
)
# A voz quando o pedido nao traz uma do Kokoro (ou traz uma da OpenAI, "alloy").
# Portugues do Brasil: pf_dora (feminina), pm_alex e pm_santa (masculinas).
VOZ_PADRAO = os.environ.get("VOZ_PADRAO", "pf_dora")
VOZ_VELOCIDADE = _decimal("VOZ_VELOCIDADE", 1.0)
VOZ_TEMPO_MAXIMO = _inteiro("VOZ_TEMPO_MAXIMO", 120)
VOZ_TEXTO_MAXIMO = _inteiro("VOZ_TEXTO_MAXIMO", 4000)


# ---------------------------------------------------------------------------
# Arbitragem
# ---------------------------------------------------------------------------
# Quanto esperar pelo lock antes de devolver 503. Zero = recusa na hora.
#
# Zero e o padrao de proposito: os clientes tem fila propria e duravel, e uma
# espera aqui so consome conexao dos dois lados para chegar no mesmo lugar.
# Subir isto para 2 ou 3 segundos absorve disputa fina sem virar fila.
ESPERA_PELO_LOCK = _decimal("ESPERA_PELO_LOCK", 0.0)

# Estimativas para o `Retry-After`, em segundos. Nao precisam ser exatas: elas
# so evitam mandar o cliente voltar em 60s quando falta meio segundo, ou em
# 60s quando falta um minuto e meio.
DURACAO_ESTIMADA = {
    "texto": _inteiro("ESTIMATIVA_TEXTO", 30),
    "imagem": _inteiro("ESTIMATIVA_IMAGEM", 60),
    "conversao": _inteiro("ESTIMATIVA_CONVERSAO", 40),
    # Um audio de uma hora leva alguns minutos. O `Retry-After` encolhe com o
    # tempo decorrido, entao errar para cima so custa a primeira espera.
    "transcricao": _inteiro("ESTIMATIVA_TRANSCRICAO", 300),
    "vetorizacao": _inteiro("ESTIMATIVA_VETORIZACAO", 15),
    # Uns 20 minutos por 2,4 s de video numa placa de 8 GB.
    "animacao": _inteiro("ESTIMATIVA_ANIMACAO", 1500),
}
