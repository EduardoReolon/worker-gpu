"""Rota de texto: `POST /v1/chat/completions`, arbitrada.

O worker publica o mesmo endpoint que o Ollama ja publicava, e repassa. Para
quem integra, a unica diferenca e o endereco — e o 503 quando a placa esta com
outro trabalho.

Essa e a mudanca que faz o arbitro funcionar. Enquanto o CRM e o PubliBot
falavam com o Ollama direto, ninguem sabia quando a placa estava ocupada e
nada podia ser descarregado com seguranca.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

import modelos
import ollama
import respostas
from arbitro import ARBITRO, GpuOcupada
from seguranca import conferir

logger = logging.getLogger("worker-gpu.texto")

router = APIRouter()

# Status do Ollama que sao "volte depois", e nao "seu pedido esta errado".
#
# Repassados crus, eles chegariam ao cliente como 503 sem `error.code` e sem
# `Retry-After` — a unica forma de 503 que sai daqui sem nada para decidir. Um
# cliente que ramifica por `error.code` leria isso como "adiavel, codigo
# desconhecido" e reagendaria para sempre.
#
# 502 e 504 entram junto com o 503 porque `OLLAMA_URL` aceita qualquer
# endereco: apontando para um proxy reverso, sao esses dois que aparecem no
# lugar do 503, com exatamente a mesma armadilha.
STATUS_TRANSITORIOS_DO_OLLAMA = (502, 503, 504)


@router.post("/v1/chat/completions", dependencies=[Depends(conferir)])
def chat(corpo: dict):
    """Gera texto. `def` e nao `async def`: a chamada ao Ollama bloqueia, e no
    event loop ela congelaria o processo inteiro — inclusive o `/health/`."""
    if not isinstance(corpo, dict) or not corpo.get("messages"):
        raise HTTPException(422, "corpo sem `messages`.")

    if corpo.get("stream"):
        # Recusar e melhor que ignorar em silencio: quem pediu streaming esta
        # esperando pedacos, e receber tudo de uma vez quebraria a leitura
        # dele de um jeito dificil de diagnosticar.
        raise HTTPException(
            422,
            "streaming nao e suportado: o arbitro precisa saber quando o "
            "trabalho termina para soltar a placa. Use `stream: false`.",
        )

    try:
        # ANTES do lock, e de proposito: se o modelo nao estiver no disco, isto
        # dispara o download em segundo plano e levanta. Baixar com o lock na
        # mao prenderia a placa por dezenas de minutos — a imagem e a conversao
        # ficariam em 503 por causa de um modelo de texto que nem carregou.
        modelos.conferir(corpo.get("model"))
    except modelos.Baixando as erro:
        return respostas.indisponivel("baixando_modelo", str(erro), retry_after=erro.tentar_em)
    except modelos.NaoDisponivel as erro:
        # 404 e nao 503: o modelo nao existe e nao vai passar a existir
        # sozinho. Um 503 aqui poria o cliente a reagendar para sempre.
        return JSONResponse({"error": {"message": str(erro)}}, status_code=404)

    try:
        # O modelo vem do cliente, intacto. Quem chama decide — o CRM tem um
        # modelo por tenant, o PubliBot tem o da conexao. Anota-lo no arbitro
        # e o que permite ao proximo 503 dizer *qual* modelo esta na placa.
        with ARBITRO.usar("texto", modelo=corpo.get("model")):
            status, dados = ollama.conversar(corpo, {})
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    except ollama.OllamaDemorouDemais as erro:
        # Sem `Retry-After`: para este codigo a doc manda reduzir o pedido, e
        # nao voltar igual mais tarde.
        return respostas.indisponivel("timeout", str(erro), retry_after=None)
    except ollama.OllamaIndisponivel as erro:
        # 503 e nao 500: o Ollama desligado e um estado transitorio, e os
        # clientes ja sabem esperar por ele.
        return respostas.indisponivel("ollama_indisponivel", str(erro))

    if status in STATUS_TRANSITORIOS_DO_OLLAMA:
        return respostas.indisponivel("ollama_indisponivel", _mensagem_do_ollama(status, dados))

    # Todo o resto vai VERBATIM, status e corpo. Um 404 de modelo inexistente e
    # um 400 de `json_schema` invalido sao respostas certas para o cliente:
    # "seu pedido esta errado" e informacao, e traduzi-las so apagaria o que o
    # Ollama tinha a dizer.
    return JSONResponse(dados, status_code=status)


def _mensagem_do_ollama(status: int, dados: dict) -> str:
    """A mensagem do upstream, preservada dentro do envelope.

    O envelope troca a FORMA (para o cliente achar `error.code`), nunca o
    diagnostico: quem esta lendo o journal precisa do texto original.
    """
    erro = dados.get("error") if isinstance(dados, dict) else None
    if isinstance(erro, dict) and erro.get("message"):
        detalhe = str(erro["message"])
    elif isinstance(erro, str) and erro:
        detalhe = erro
    else:
        detalhe = str(dados)[:300]

    return f"o Ollama respondeu {status}: {detalhe}"


@router.get("/v1/models", dependencies=[Depends(conferir)])
def catalogo():
    """O catalogo, no formato que os clientes do dialeto OpenAI esperam.

    Inclui os modelos de texto do Ollama e, quando a geracao de imagem esta
    ligada, o modelo de imagem — um cliente perguntando "o que da para usar"
    precisa ver os dois.
    """
    from config import IMAGEM_ATIVA, IMAGEM_MODELO, TRANSCRICAO_ATIVA, TRANSCRICAO_MODELO

    nomes = list(ollama.modelos_no_disco())
    if IMAGEM_ATIVA:
        nomes.append(IMAGEM_MODELO)
    if TRANSCRICAO_ATIVA:
        nomes.append(TRANSCRICAO_MODELO)

    return {"object": "list", "data": [{"id": nome, "object": "model"} for nome in nomes]}
