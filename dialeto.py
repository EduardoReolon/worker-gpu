"""Traducao entre o dialeto da OpenAI e o nativo do Ollama.

Este modulo existe por causa de UM campo: `num_ctx`.

O worker publica `/v1/chat/completions` porque e o que os clientes falam, e
repassava para o `/v1/chat/completions` do Ollama, que e a camada compativel
com a OpenAI. Ela desserializa o corpo num struct tipado, e **chave que ela nao
conhece some na desserializacao** — sem erro, sem log, com 200 na resposta.
`options` inteiro cai nessa, e com ele a janela de contexto.

O efeito e o pior tipo de defeito: o modelo roda com a janela padrao dele
(2048 ou 4096), o prompt e truncado PELO COMECO, e o que morre primeiro e o
prompt de sistema. A resposta volta 200, com JSON valido e schema respeitado, e
o modelo nao viu as instrucoes. Passa por revisao automatizada, passa por
contagem de caracteres, e envenena tudo o que for construido em cima.

## Por que traduzir, e nao embutir num Modelfile

Um `PARAMETER num_ctx` num Modelfile resolve sem tocar no worker, e era a
recomendacao. Ela tem tres custos que a traducao nao tem:

- alguem precisa rodar `ollama create` a cada maquina nova, e esquecer nao da
  erro nenhum — volta a janela padrao, em silencio;
- com `OLLAMA_MAX_LOADED_MODELS=1`, cada derivado e um modelo carregado
  distinto: dois inquilinos com a mesma base e janelas diferentes pagam troca
  entre si;
- o `ollama list` da maquina, que e compartilhada com outro projeto, enche de
  modelos derivados que ninguem sabe de quem sao.

## Quando o nativo entra

**So quando o pedido traz algo que a camada compativel descartaria** — hoje
`options` ou `keep_alive`. Quem nao manda nenhum dos dois segue pelo caminho
antigo, byte por byte, e nao corre risco nenhum desta traducao.

E deliberado: ha dois clientes em producao, e um deles nao precisa disto. A
troca e ter dois caminhos para manter em vez de um; o ganho e que o cliente que
nao pediu nada nao paga por uma traducao que eu poderia ter escrito errado.
Quando os dois estiverem exercitados, unificar e mudar uma linha.

## O que a traducao NAO cobre

`content` como lista de partes (o multimodal do dialeto da OpenAI) vai
inalterado para o nativo, que espera `content` string e `images` separado. Se
um cliente mandar imagem por aqui, o Ollama recusa — e recusa com erro, que e o
comportamento aceitavel. Nenhum dos dois clientes manda imagem nesta rota.

Campo que nao tem para onde ir vira **AVISO no journal**, e nunca silencio.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

logger = logging.getLogger("worker-gpu.dialeto")

# Campos do dialeto da OpenAI que viram `options` do Ollama.
#
# `max_tokens` -> `num_predict` e o unico que a camada compativel ja fazia por
# conta propria; os outros estao aqui para o caminho nativo nao perder nada que
# o caminho antigo entregava.
PARA_OPTIONS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
    "max_tokens": "num_predict",
    "max_completion_tokens": "num_predict",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
}

# Campos que existem igual nos dois dialetos, na raiz do corpo.
PARA_RAIZ = ("model", "messages", "tools", "keep_alive", "think")

# Aceitos e sem efeito, com o motivo. Estao nomeados para NAO virarem aviso: o
# cliente pode manda-los, e avisar sobre eles a cada pedido enterraria os
# avisos que importam.
IGNORADOS = {
    "stream": "o worker nao faz streaming",
    "stream_options": "sem streaming, nada a configurar",
    "user": "nao ha usuarios neste servico",
    "tool_choice": "o Ollama nao expoe escolha forcada de ferramenta",
    "logit_bias": "sem equivalente nativo",
    "logprobs": "sem equivalente nativo",
    "top_logprobs": "sem equivalente nativo",
}

_TRATADOS = (
    set(PARA_RAIZ) | set(PARA_OPTIONS) | set(IGNORADOS) | {"options", "response_format", "n"}
)


def precisa_do_nativo(corpo: dict) -> bool:
    """Se este pedido perderia alguma coisa pela camada compativel.

    Um `options` vazio nao conta: `{"options": {}}` nao pede nada, e mandar o
    pedido pelo caminho novo sem necessidade seria trocar risco por nada.
    """
    return bool(corpo.get("options")) or "keep_alive" in corpo


def para_nativo(corpo: dict) -> dict:
    """Corpo do dialeto da OpenAI -> corpo do `/api/chat`."""
    opcoes = dict(corpo.get("options") or {})
    nativo: dict = {"stream": False}

    for chave in PARA_RAIZ:
        if chave in corpo:
            nativo[chave] = corpo[chave]

    for da_openai, do_ollama in PARA_OPTIONS.items():
        if corpo.get(da_openai) is not None:
            # `setdefault`: o `options` explicito do cliente GANHA do campo
            # equivalente da OpenAI. Foi ele que trouxe o pedido para este
            # caminho, e ele e a intencao mais especifica — um cliente que
            # manda `max_tokens: 100` e `options: {"num_predict": 500}` quis
            # 500, ou nao teria escrito o segundo.
            opcoes.setdefault(do_ollama, corpo[da_openai])

    formato = corpo.get("response_format")
    if isinstance(formato, dict):
        if formato.get("type") == "json_schema":
            esquema = (formato.get("json_schema") or {}).get("schema")
            if esquema is not None:
                nativo["format"] = esquema
        elif formato.get("type") == "json_object":
            nativo["format"] = "json"

    if opcoes:
        nativo["options"] = opcoes

    _avisar_do_que_ficou_de_fora(corpo)

    return nativo


def _avisar_do_que_ficou_de_fora(corpo: dict) -> None:
    """Campo que nao tem para onde ir vira aviso, e nunca silencio.

    E a regra desta rota inteira: o defeito que custa caro aqui e o que
    responde 200 e entrega menos do que foi pedido.
    """
    if corpo.get("n") not in (None, 1):
        logger.warning(
            "`n`=%s: o Ollama gera UMA resposta por pedido. As outras nao vao existir.",
            corpo["n"],
        )

    de_fora = sorted(chave for chave in corpo if chave not in _TRATADOS)
    if de_fora:
        logger.warning(
            "Campos sem equivalente no dialeto nativo do Ollama, ignorados: %s",
            ", ".join(de_fora),
        )


def para_openai(nativo: dict, modelo_pedido: str | None = None) -> dict:
    """Resposta do `/api/chat` -> resposta do dialeto da OpenAI.

    A forma de saida e a de `contrato/texto-resposta.json`, e precisa ser
    indistinguivel da que o caminho antigo devolvia: quem integra copiou aquele
    exemplo para os testes dele, e nao tem como saber por qual caminho o pedido
    foi.
    """
    mensagem = dict(nativo.get("message") or {})
    # O `message` vai INTEIRO, e nao campo por campo: se uma versao nova do
    # Ollama acrescentar algo ali (foi o caso do `thinking`), campo por campo
    # perderia em silencio. Cliente do dialeto da OpenAI ignora o que nao
    # conhece; perder o dado nao tem volta.
    mensagem.setdefault("role", "assistant")
    mensagem.setdefault("content", "")

    chamadas = mensagem.get("tool_calls")
    if chamadas:
        mensagem["tool_calls"] = [_chamada_para_openai(chamada) for chamada in chamadas]

    do_prompt = int(nativo.get("prompt_eval_count") or 0)
    da_saida = int(nativo.get("eval_count") or 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": nativo.get("model") or modelo_pedido or "",
        "choices": [
            {
                "index": 0,
                "message": mensagem,
                "finish_reason": _motivo(nativo.get("done_reason"), bool(chamadas)),
            }
        ],
        # `prompt_tokens` e o detector de truncamento do lado do cliente: ele
        # compara com o que mandou. Perde-lo aqui tiraria o unico sinal barato
        # que existe para o defeito que este modulo veio consertar.
        "usage": {
            "prompt_tokens": do_prompt,
            "completion_tokens": da_saida,
            "total_tokens": do_prompt + da_saida,
        },
    }


def _motivo(done_reason: str | None, teve_chamada: bool) -> str:
    if teve_chamada:
        return "tool_calls"
    # `length` e o que denuncia geracao cortada pelo teto. Ele tem o mesmo nome
    # nos dois dialetos; o resto ("stop", "load", ausente) vira "stop".
    return "length" if done_reason == "length" else "stop"


def _chamada_para_openai(chamada: dict) -> dict:
    funcao = dict((chamada or {}).get("function") or {})

    if not isinstance(funcao.get("arguments"), str):
        # No nativo os argumentos vem como OBJETO; no dialeto da OpenAI eles
        # sao uma STRING de JSON. Um cliente que faz `json.loads` nos
        # argumentos — o que a doc da OpenAI manda fazer — quebraria com o
        # objeto, e quebraria no repositorio dele.
        funcao["arguments"] = json.dumps(funcao.get("arguments") or {}, ensure_ascii=False)

    return {
        "id": (chamada or {}).get("id") or f"call_{uuid.uuid4().hex[:20]}",
        "type": "function",
        "function": funcao,
    }


def erro_para_openai(nativo: dict) -> dict:
    """Erro do `/api/chat` -> erro na forma da OpenAI.

    O nativo devolve `{"error": "texto"}`; a camada compativel devolve
    `{"error": {"message": "texto"}}`. Sem esta normalizacao, o MESMO erro
    chegaria ao cliente em duas formas diferentes conforme o caminho que o
    pedido tomou — e ele nao tem como saber qual foi.
    """
    erro = nativo.get("error") if isinstance(nativo, dict) else None

    if isinstance(erro, dict):
        return {"error": erro}
    if isinstance(erro, str) and erro:
        return {"error": {"message": erro}}

    return {"error": {"message": str(nativo)[:500]}}
