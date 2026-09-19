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
OLLAMA_TIMEOUT = _decimal("OLLAMA_TIMEOUT", 600.0)

# Descarregar o modelo de texto da VRAM antes de um trabalho de imagem.
#
# E o ponto do arbitro existir. Numa placa de 8 GB um modelo de texto grande
# ocupa quase tudo, e a difusao precisa de ~5,4 GB — nao cabem juntos. Como
# TODO pedido passa por aqui, descarregar e seguro: quem detem o lock sabe que
# ninguem esta gerando texto naquele instante.
OLLAMA_DESCARREGAR_PARA_IMAGEM = _booleano("OLLAMA_DESCARREGAR_PARA_IMAGEM", True)

# Quanto tempo o Ollama mantem o modelo carregado depois de responder. Enviado
# em cada pedido, entao vale mesmo sem mexer no servico dele.
#
# Vazio = nao mexer, e ai vale o padrao do Ollama (5 min).
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "")


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
