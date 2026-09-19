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

import ollama
import respostas
from arbitro import ARBITRO, GpuOcupada
from seguranca import conferir

logger = logging.getLogger("worker-gpu.texto")

router = APIRouter()


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
        # O modelo vem do cliente, intacto. Quem chama decide — o CRM tem um
        # modelo por tenant, o PubliBot tem o da conexao. Anota-lo no arbitro
        # e o que permite ao proximo 503 dizer *qual* modelo esta na placa.
        with ARBITRO.usar("texto", modelo=corpo.get("model")):
            status, dados = ollama.conversar(corpo, {})
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    except ollama.OllamaIndisponivel as erro:
        # 503 e nao 500: o Ollama desligado e um estado transitorio, e os
        # clientes ja sabem esperar por ele.
        return respostas.indisponivel("ollama_indisponivel", str(erro))

    return JSONResponse(dados, status_code=status)


@router.get("/v1/models", dependencies=[Depends(conferir)])
def modelos():
    """O catalogo, no formato que os clientes do dialeto OpenAI esperam.

    Inclui os modelos de texto do Ollama e, quando a geracao de imagem esta
    ligada, o modelo de imagem — um cliente perguntando "o que da para usar"
    precisa ver os dois.
    """
    from config import IMAGEM_ATIVA, IMAGEM_MODELO

    nomes = list(ollama.modelos_no_disco())
    if IMAGEM_ATIVA:
        nomes.append(IMAGEM_MODELO)

    return {"object": "list", "data": [{"id": nome, "object": "model"} for nome in nomes]}
