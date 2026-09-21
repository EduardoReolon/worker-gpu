"""O worker de GPU: um processo, uma placa, um arbitro.

Esta maquina tem uma GPU e varios clientes — o PubliBot, o CRM, e o que vier.
A placa e indivisivel: um modelo de texto de 30B ja ocupa quase toda a VRAM de
8 GB, e a difusao precisa de ~5,4 GB. Nao cabem juntos, e o que acontece
quando se tenta nao e um erro — e o processo caindo para CPU em silencio,
dezenas de vezes mais lento.

Por isso **tudo** entra por aqui, inclusive o texto, que antes ia direto ao
Ollama:

    POST /v1/chat/completions    texto        (repassa ao Ollama)
    POST /v1/images/generations  imagem       (difusao)
    POST /parse/                 conversao    (Docling)
    GET  /v1/models              catalogo
    GET  /health/                estado, sem credencial

Um processo so, e nao tres, porque o lock que arbitra a placa precisa ser um
so. Tres servicos separados teriam tres locks, e tres locks sobre uma placa
nao protegem nada — que era exatamente a situacao anterior.

## O contrato com quem integra

**503 com `Retry-After`, nunca fila.** Este processo nao tem estado duravel:
uma fila em memoria perderia trabalho no primeiro restart. Os clientes ja tem
fila persistente e sabem retomar. O `Retry-After` e calculado a partir do que
esta rodando.

**Nada de streaming.** O arbitro precisa saber quando o trabalho termina para
soltar a placa, e uma resposta em streaming so termina quando o cliente
termina de ler.

`INTEGRACAO.md` tem o guia completo para adaptar um cliente.

## Escuta

Em `BIND_HOST`, que precisa ser o endereco da rede privada (Tailscale) ou
`127.0.0.1`. **Nunca 0.0.0.0**: um endpoint que roda modelo na sua placa,
aberto na internet, e placa de graca para quem achar a porta. O instalador
recusa.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

import conversao
import imagem
import modelos
import ollama
import texto
from arbitro import ARBITRO
from config import CONVERSAO_ATIVA, IMAGEM_ATIVA

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s %(message)s")
logger = logging.getLogger("worker-gpu")

# MAIOR.MENOR. A maior muda quando um campo some ou muda de significado — e
# ai quem integra precisa olhar, e `INTEGRACAO.md` ganha uma secao.
# Acrescentar campo nao quebra ninguem e nao sobe nada: todo cliente deve
# ignorar o que nao conhece.
CONTRATO_VERSAO = "2.3"

app = FastAPI(title="worker-gpu", version=CONTRATO_VERSAO)

# O texto sempre entra: e ele que faz o arbitro valer. Sem ele os clientes
# voltariam a falar com o Ollama direto, e nada seria arbitrado.
app.include_router(texto.router)

if IMAGEM_ATIVA:
    app.include_router(imagem.router)

if CONVERSAO_ATIVA:
    # Conferido na subida, e nao na primeira conversao: um OCR pedido e
    # ausente so apareceria no PDF digitalizado, que e o caso mais raro.
    conversao.conferir_ocr()
    app.include_router(conversao.router)


def _bloco(nome: str, coletar):
    """Um bloco do `/health/`, ou a descricao do erro no lugar dele.

    O `/health/` NUNCA pode devolver 500. Ele e o endpoint que o instalador, o
    systemd e quem esta diagnosticando consultam, e uma excecao ali transforma
    "alguma coisa esta errada" em "tudo esta errado e nao sei o que". Um bloco
    que falha vira `{"erro": "<tipo>: <mensagem>"}` DENTRO do corpo, com 200 —
    o processo esta de pe, e e isso que o 200 afirma.

    Pelo mesmo motivo que ele dispensa credencial: um diagnostico que precisa
    de segredo nao serve para descobrir por que o segredo nao funciona, e um
    diagnostico que estoura nao serve para descobrir o que estourou.
    """
    try:
        return coletar()
    except Exception as erro:
        logger.exception("Falha ao montar o bloco %r do /health/", nome)
        return {"erro": f"{type(erro).__name__}: {erro}"}


def _bloco_do_ollama() -> dict:
    """Estado do Ollama, numa ida so ao `/api/ps`.

    `carregados` continua sendo a lista de NOMES: e o campo publicado, e ha
    cliente lendo. `carregados_detalhe` e o registro inteiro, e existe por um
    motivo: e nele que vem o `context_length` com que cada modelo foi
    carregado. Janela errada e o defeito que nao aparece — 200, JSON valido,
    schema respeitado, e o modelo nao viu o inicio do prompt de sistema.

    O campo e uma dica de MAQUINA ("esta configurada errada"), e nao uma
    confirmacao de PEDIDO: ele e lido por outra viagem, depois, e entre a
    inferencia e a leitura outro cliente pode ter recarregado o modelo com
    outra janela.
    """
    detalhe = ollama.modelos_carregados_detalhe()

    return {
        "de_pe": ollama.esta_de_pe(),
        "carregados": ollama.modelos_carregados(detalhe),
        "carregados_detalhe": detalhe,
        # Downloads em curso. Sem isto, um lote inteiro levando
        # `baixando_modelo` nao teria como ser distinguido, de fora, de um
        # worker travado — e o download nao aparece em `ocupada`, porque ele
        # nao usa a placa.
        "baixando": modelos.estado(),
    }


@app.get("/health/")
def health():
    """Estado, sem credencial.

    Sem credencial de proposito: e o endpoint que o instalador, o systemd e
    quem esta diagnosticando consultam, e exigir segredo ai transforma "o
    servico caiu" em "o servico caiu ou o segredo esta errado". Ele nao
    revela nada que ja nao se saiba abrindo a porta.

    `def` e nao `async def`: ele toca disco, e no event loop uma leitura
    lenta atrasaria todo o resto. (Foi um `async def` num handler bloqueante
    que fez este servico parar de responder enquanto gerava.)

    A forma desta resposta esta em `contrato/saude-resposta.json`, e um teste
    confere que as duas nao divergem. Sem isso o endpoint que todo mundo usa
    para diagnosticar era o unico sem exemplo publicado — e ja mudou de forma
    uma vez sem quebrar nada ate doer.
    """
    corpo = {
        "status": "ok",
        "service": "worker-gpu",
        # Quem integra compara isto com o `_contrato_versao` dos exemplos que
        # copiou para os testes dele. Sem um numero publicado, a copia velha
        # do cliente nao tem como se denunciar.
        "contrato_versao": CONTRATO_VERSAO,
    }

    try:
        ocupacao = ARBITRO.ocupacao
    except Exception as erro:
        logger.exception("Falha ao ler a ocupacao para o /health/")
        ocupacao = None
        corpo["arbitro"] = {"erro": f"{type(erro).__name__}: {erro}"}

    corpo["ocupada"] = ocupacao is not None
    corpo["ocupante"] = ocupacao.tarefa if ocupacao else None
    # Qual modelo esta em uso agora. Quem planeja um lote le isto junto com
    # `ollama.carregados` para escolher pedidos que nao paguem troca.
    corpo["modelo"] = ocupacao.modelo if ocupacao else None
    corpo["ha_segundos"] = ocupacao.ha_quantos_segundos if ocupacao else 0
    corpo["rotas"] = {
        "texto": True,
        "imagem": IMAGEM_ATIVA,
        "conversao": CONVERSAO_ATIVA,
    }
    corpo["ollama"] = _bloco("ollama", _bloco_do_ollama)

    if IMAGEM_ATIVA:
        corpo["imagem"] = _bloco("imagem", imagem.estado)
    if CONVERSAO_ATIVA:
        corpo["conversao"] = _bloco("conversao", conversao.estado)

    return corpo
