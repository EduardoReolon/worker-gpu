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
IMAGEM_PASSOS = _inteiro("IMAGEM_PASSOS", 25)
IMAGEM_GUIDANCE = _decimal("IMAGEM_GUIDANCE", 7.0)
IMAGEM_NEGATIVO = os.environ.get(
    "IMAGEM_NEGATIVO",
    "texto, letras, palavras, marca d'agua, logotipo, assinatura, moldura, "
    "baixa qualidade, borrado, deformado",
)
IMAGEM_OCIOSO_SEGUNDOS = _inteiro("IMAGEM_OCIOSO_SEGUNDOS", 300)
IMAGEM_MAXIMO = _inteiro("IMAGEM_MAXIMO", 4)
IMAGEM_LADO_MAXIMO = _inteiro("IMAGEM_LADO_MAXIMO", 1024)

# Gerar em CPU quando a VRAM nao couber. Desligado: medido em uso, um lote em
# CPU consumiu horas de processador e 17 GB de RAM (float32), com a maquina
# inutilizavel. Recusar devolve o trabalho para a fila do cliente, que sabe
# esperar.
IMAGEM_PERMITIR_CPU = _booleano("IMAGEM_PERMITIR_CPU", False)

# Teto por geracao. Sem ele nao ha como interromper um laco de difusao, e ate
# um `systemctl restart` fica preso esperando.
IMAGEM_TEMPO_MAXIMO = _inteiro("IMAGEM_TEMPO_MAXIMO", 600)


# ---------------------------------------------------------------------------
# Conversao de PDF (Docling)
# ---------------------------------------------------------------------------
CONVERSAO_ATIVA = _booleano("CONVERSAO_ATIVA", True)
DOCLING_DEVICE = os.environ.get("DOCLING_DEVICE", "auto").lower()
DOCLING_THREADS = _inteiro("DOCLING_THREADS", 0)
DOCLING_OCR = _booleano("DOCLING_OCR", False)
MAX_PDF_BYTES = _inteiro("MAX_PDF_BYTES", 100 * 1024 * 1024)


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
}
