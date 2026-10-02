"""A janela de contexto do modelo de texto, decidida a cada pedido.

O Ollama carrega o modelo com 4096 tokens de contexto por padrao. Um pedido
maior — planejar um artigo manda ~5 mil — volta 400, ou, pior, e truncado pelo
comeco e o modelo nao ve o prompt de sistema. Subir a janela a mao resolve ate
a proxima reinstalacao; este modulo faz o worker se ajustar sozinho.

## As tres regras

1. **O pedido diz quanto precisa**, no cabecalho `X-PubliBot-Contexto` (ou em
   `options.num_ctx`). Sem nenhum dos dois, vale o `CONTEXTO_PADRAO`.
2. **A janela nunca baixa.** Cada troca de `num_ctx` faz o Ollama RECARREGAR
   o modelo. Por isso vai sempre `max(pedido, maior_ja_usado)`, por modelo: um
   pedido pequeno depois de um grande reaproveita a carga que ja esta la.
3. **A janela nunca passa do `CONTEXTO_MAXIMO`**, que e o que esta maquina
   aguenta. Acima da VRAM o Ollama nao recusa — poe camadas na CPU, e a
   geracao fica muitas vezes mais lenta. Se o pedido nao couber nem no maximo,
   o Ollama recusa, e o erro sai na forma do llama.cpp
   (`exceed_context_size_error`, com `n_prompt_tokens` e `n_ctx`).

Aplicado a TODO pedido de texto, e nao so aos do PubliBot: dois clientes no
mesmo modelo com janelas diferentes fariam o Ollama recarregar a cada vez que
um sucede o outro.
"""

from __future__ import annotations

import logging
import re
import threading

from config import CONTEXTO_MAXIMO, CONTEXTO_PADRAO

logger = logging.getLogger("worker-gpu.contexto")

CABECALHO = "X-PubliBot-Contexto"

_maior_por_modelo: dict[str, int] = {}
_trava = threading.Lock()

# "request (4835 tokens) exceeds the available context size (4096 tokens)" e
# variacoes. O texto e do llama.cpp, por dentro do Ollama; os dois numeros sao
# o que o cliente precisa para dizer "o pedido tem X, cabem Y".
_EXCEDE = re.compile(
    r"\(?(\d+)\s*tokens?\)?\s*exceeds?\s+the\s+available\s+context\s+size\s*\(?(\d+)",
    re.IGNORECASE,
)


def _do_cabecalho(valor: str | None) -> int | None:
    if valor is None or not valor.strip():
        return None
    try:
        numero = int(valor.strip())
    except ValueError:
        logger.warning("%s=%r nao e um numero; usando o padrao.", CABECALHO, valor)
        return None
    if numero <= 0:
        logger.warning("%s=%s nao e positivo; usando o padrao.", CABECALHO, numero)
        return None
    return numero


def decidir(modelo: str, cabecalho: str | None, corpo: dict) -> int:
    """O `num_ctx` deste pedido, ou 0 para nao mandar nenhum."""
    explicito = (corpo.get("options") or {}).get("num_ctx")
    explicito = explicito if isinstance(explicito, int) and explicito > 0 else None
    do_cabecalho = _do_cabecalho(cabecalho)

    pedido = max(do_cabecalho or CONTEXTO_PADRAO, explicito or 0)
    if pedido <= 0:
        return 0

    with _trava:
        janela = max(pedido, _maior_por_modelo.get(modelo, 0))
        if CONTEXTO_MAXIMO > 0 and janela > CONTEXTO_MAXIMO:
            if pedido > CONTEXTO_MAXIMO:
                logger.info(
                    "O pedido quer %s tokens de contexto; o teto desta maquina e %s "
                    "(CONTEXTO_MAXIMO). Indo com o teto.",
                    pedido,
                    CONTEXTO_MAXIMO,
                )
            janela = CONTEXTO_MAXIMO
        _maior_por_modelo[modelo] = max(janela, _maior_por_modelo.get(modelo, 0))

    return janela


def aplicar(corpo: dict, cabecalho: str | None) -> tuple[dict, int]:
    """O corpo com `options.num_ctx` decidido, e a janela (0 se nenhuma).

    Com `options`, o pedido vai pela API nativa (`/api/chat`): a camada
    compativel com a OpenAI descarta `options` em silencio.
    """
    janela = decidir(str(corpo.get("model") or ""), cabecalho, corpo)
    if not janela:
        return corpo, 0

    novo = dict(corpo)
    opcoes = dict(novo.get("options") or {})
    opcoes["num_ctx"] = janela
    novo["options"] = opcoes
    return novo, janela


def erro_de_contexto(status: int, dados: dict, janela: int) -> dict | None:
    """O 400 de "nao cabe" na forma do llama.cpp, ou None se nao for esse.

    O cliente le `error.type == "exceed_context_size_error"` e os dois numeros
    para dizer a pessoa quanto o pedido tem e quanto cabe. A mensagem original
    vai junto, intacta.
    """
    if status != 400 or not isinstance(dados, dict):
        return None

    erro = dados.get("error")
    mensagem = erro.get("message") if isinstance(erro, dict) else erro
    if not isinstance(mensagem, str):
        return None

    achado = _EXCEDE.search(mensagem)
    baixo = mensagem.lower()
    if not achado and not ("context" in baixo and "exceed" in baixo):
        return None

    corpo = {"code": 400, "type": "exceed_context_size_error", "message": mensagem}
    if achado:
        corpo["n_prompt_tokens"] = int(achado.group(1))
        corpo["n_ctx"] = int(achado.group(2))
    elif janela:
        corpo["n_ctx"] = janela
    return {"error": corpo}


def avisar_se_encheu(dados: dict, janela: int) -> None:
    """Uma resposta 200 que usou a janela inteira provavelmente foi truncada.

    O Ollama, sem espaco, pode descartar o comeco da conversa em vez de
    recusar. Nao da para afirmar daqui (o cache de prompt reduz a contagem),
    mas o journal e o lugar de desconfiar.
    """
    if not janela or not isinstance(dados, dict):
        return
    uso = dados.get("usage") or {}
    total = int(uso.get("prompt_tokens") or 0) + int(uso.get("completion_tokens") or 0)
    if total >= janela:
        logger.warning(
            "O pedido usou %s tokens numa janela de %s: o comeco pode ter sido "
            "descartado. Suba CONTEXTO_MAXIMO, se a placa aguentar.",
            total,
            janela,
        )


def estado(carregados: list[dict]) -> dict:
    """O bloco `ollama.contexto` do `/health/`.

    `atual` vem do `/api/ps` (o `context_length` com que o modelo esta
    carregado), e nao da memoria daqui: depois de um restart do worker, o
    Ollama pode continuar com o modelo carregado.
    """
    atual = 0
    for item in carregados:
        valor = item.get("context_length")
        if isinstance(valor, int) and valor > atual:
            atual = valor

    return {
        "padrao": CONTEXTO_PADRAO,
        "atual": atual,
        "maximo": CONTEXTO_MAXIMO,
        "segue_cabecalho": True,
    }
