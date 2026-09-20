"""O Ollama, visto pelo worker: proxy de texto e controle de memoria.

Duas responsabilidades que so fazem sentido juntas.

**Proxy.** O worker publica `POST /v1/chat/completions` e repassa ao Ollama,
que ja fala esse dialeto. O cliente nao percebe diferenca — e e justamente por
isso que ele passa a ser arbitrado sem ninguem reescrever nada.

**Memoria.** O Ollama decide sozinho quando carregar e descarregar modelo, e
nao participa de reserva nenhuma. Num placa de 8 GB com um modelo grande, isso
significa que a difusao nunca acha VRAM livre. O worker resolve porque agora
ele sabe o que esta acontecendo: detendo o lock da GPU, ninguem esta gerando
texto, entao mandar o Ollama soltar o modelo e seguro.

O jeito de soltar e o proprio protocolo: um pedido com `keep_alive: 0` e sem
prompt descarrega o modelo. Nao ha comando especial nem processo a matar.

## O que NAO atravessa este proxy

O corpo vai intacto, mas "intacto" nao e "inteiro": `/v1/chat/completions` do
Ollama e a camada compativel com a OpenAI, e ela desserializa o corpo num
struct tipado. **Chave que ela nao conhece some na desserializacao, sem erro e
sem log** — `options` inteiro, e com ele `num_ctx` e `num_predict`.

O que funciona e o que e campo do dialeto da OpenAI: `max_tokens` (que ela
mesma traduz para `num_predict`), `temperature`, `top_p`, `seed`, `stop`,
`response_format`. Quem precisa de janela de contexto propria embute a janela
no modelo (`PARAMETER num_ctx` num Modelfile) ou ajusta o
`OLLAMA_CONTEXT_LENGTH` da maquina.

Havia aqui uma injecao de `keep_alive` no corpo, vinda de um `OLLAMA_KEEP_ALIVE`
no `.env`. Ela foi REMOVIDA: `keep_alive` tambem nao e campo do dialeto da
OpenAI, entao a variavel parecia ativa e nao tinha efeito nenhum — o pior tipo
de configuracao, a que mente. Enquanto o proxy falar o dialeto da OpenAI, a
politica de memoria desta maquina se ajusta no Ollama, e nao aqui.
"""

from __future__ import annotations

import logging

import httpx

from config import OLLAMA_TIMEOUT, OLLAMA_URL

logger = logging.getLogger("worker-gpu.ollama")


class OllamaIndisponivel(RuntimeError):
    """Nao foi possivel falar com o Ollama. Transitorio: repetir faz sentido."""


class OllamaDemorouDemais(RuntimeError):
    """O trabalho passou do orcamento de tempo. Repetir IGUAL nao faz sentido.

    Irma de `OllamaIndisponivel`, e nao subclasse. Com heranca, um
    `except OllamaIndisponivel` posto antes engoliria este caso em silencio, e
    a ordem dos `except` em `texto.py` passaria a decidir qual `error.code` o
    cliente recebe — um defeito que so aparece quando alguem reordena o arquivo
    meses depois. Irmas tornam a ordem irrelevante.
    """


# Um cliente para o processo, e nao um por chamada. Cada `httpx.Client()` monta
# um contexto SSL do zero, o que e trabalho inutil para falar HTTP com o
# loopback — e era feito em TODA ida ao Ollama, inclusive nas duas que o
# `/health/` faz.
#
# O timeout padrao e o CURTO, de proposito. Quem precisa de mais passa
# `timeout=` na chamada; quem esquecer herda 5s e falha rapido. O inverso —
# padrao longo, curto por chamada — faria a primeira funcao nova escrita sem o
# parametro pendurar o `/health/` por dez minutos, que e exatamente quando ele
# mais importa.
#
# Construido na importacao: se falhar, o servico nao sobe, e o journal diz por
# que. E melhor que subir e devolver 500 em cada pedido, pelo mesmo motivo que
# `conversao.conferir_ocr()` recusa a subida em vez de falhar na primeira
# conversao.
_cliente = httpx.Client(base_url=OLLAMA_URL, timeout=5.0)


def modelos_carregados_detalhe() -> list[dict]:
    """Os modelos na memoria AGORA, como o `/api/ps` os descreve.

    Diferente de `/api/tags`, que lista o que existe no disco. A diferenca e o
    ponto: o que ocupa VRAM e o que esta carregado.

    Devolve o registro inteiro, e nao so o nome, porque e nele que vem o
    `context_length` com que o modelo foi carregado. Janela errada e o defeito
    que nao aparece — 200, JSON valido, e o modelo nao viu o inicio do prompt
    de sistema, porque o truncamento come pela cabeca. Sem este campo nenhum
    cliente tem como desconfiar.
    """
    try:
        resposta = _cliente.get("/api/ps", timeout=10.0)
        resposta.raise_for_status()
        modelos = resposta.json().get("models") or []
    except Exception as erro:
        # `Exception` e nao `httpx.HTTPError`: um `/api/ps` que responde algo
        # que nao e JSON levanta `ValueError`, que fica de fora daquela
        # hierarquia — e derrubava o `/health/` com 500.
        logger.warning("Nao consegui listar os modelos carregados: %s", erro)
        return []

    return [item for item in modelos if isinstance(item, dict)]


def modelos_carregados(detalhe: list[dict] | None = None) -> list[str]:
    """So os nomes. Continua `list[str]`, e isso e contrato.

    `ollama.carregados` no `/health/` e publicado como lista de nomes em
    `INTEGRACAO.md` e no `README.md`, e ha cliente lendo. Enriquecer ESTE
    retorno mudaria o significado de um campo existente — MAIOR, pela regra do
    `app.py` — e quebraria `descarregar_tudo()` aqui embaixo, que monta o
    pedido com o nome. Quem quer o resto chama `modelos_carregados_detalhe()`.

    `detalhe` evita a segunda ida ao `/api/ps` para quem ja tem a lista na mao
    — o `/health/`, que publica os dois campos.
    """
    if detalhe is None:
        detalhe = modelos_carregados_detalhe()

    return [nome for item in detalhe if (nome := item.get("name"))]


def descarregar_tudo() -> list[str]:
    """Solta da memoria todos os modelos carregados. Devolve quais eram.

    Chamado com o lock da GPU na mao, antes de um trabalho de imagem. Sem ele,
    a difusao encontra a placa cheia e o pedido falha por falta de VRAM — o
    defeito que motivou o arbitro.

    Nao levanta: falhar em descarregar nao e motivo para recusar o trabalho. A
    geracao tentara e, se nao couber, o erro dira isso com o nome certo.
    """
    carregados = modelos_carregados()
    if not carregados:
        return []

    logger.info("Descarregando do Ollama para liberar a placa: %s", ", ".join(carregados))
    for modelo in carregados:
        try:
            # Sem `prompt` e com `keep_alive: 0`: o Ollama entende como
            # "carregue nada e esqueca este modelo". Aqui e o dialeto NATIVO,
            # onde `keep_alive` existe de verdade.
            _cliente.post("/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=30.0)
        except Exception as erro:
            logger.warning("Falha ao descarregar %s: %s", modelo, erro)

    return carregados


def conversar(corpo: dict, cabecalhos: dict[str, str]) -> tuple[int, dict]:
    """Repassa um `/v1/chat/completions` ao Ollama e devolve (status, json).

    O corpo vai intacto: o que o cliente pediu e o que o modelo recebe. A unica
    intromissao e o `stream`. Veja a docstring do modulo para o que a camada
    compativel do Ollama descarta por conta propria.
    """
    corpo = dict(corpo)

    # Streaming nao passa por aqui: o arbitro precisa saber quando o trabalho
    # termina para soltar a placa, e uma resposta em streaming so termina
    # quando o cliente termina de ler. Recusar e melhor que entregar pela
    # metade.
    corpo["stream"] = False

    try:
        resposta = _cliente.post(
            "/v1/chat/completions",
            json=corpo,
            headers={"Content-Type": "application/json"},
            timeout=OLLAMA_TIMEOUT,
        )
    except httpx.ReadTimeout as erro:
        # SO `ReadTimeout`, e nunca `TimeoutException`. O pai cobre tambem
        # `ConnectTimeout` (o Ollama nao atende) e `PoolTimeout` (disputa
        # interna daqui) — os dois transitorios, os dois resolvem sozinhos.
        # Rotula-los "o pedido estourou o orcamento" faria o cliente DESISTIR
        # de um trabalho que ia dar certo, que e a perda sem volta.
        #
        # `WriteTimeout` fica de fora pelo mesmo criterio: com o Ollama em
        # loopback, nao terminar de escrever o corpo significa receptor travado,
        # e nao corpo grande demais.
        raise OllamaDemorouDemais(
            f"o trabalho passou dos {OLLAMA_TIMEOUT:.0f}s de orcamento do worker: {erro}"
        ) from erro
    except httpx.HTTPError as erro:
        # O Ollama fora do ar e rotina, e nao defeito: sem traceback.
        raise OllamaIndisponivel(f"nao foi possivel falar com {OLLAMA_URL}: {erro}") from erro
    except Exception as erro:
        # Aqui nao ha rotina nenhuma: o que cai neste ramo e erro de programacao
        # — `KeyError`, `AttributeError`, um `None` onde nao devia. Ele vira 503
        # para o cliente nao ficar sem resposta, mas o `exception` e obrigatorio:
        # sem o traceback, um defeito do worker some no journal vestido de "o
        # Ollama nao responde", e o cliente reagenda por horas.
        logger.exception("Falha inesperada ao falar com o Ollama")
        raise OllamaIndisponivel(f"nao foi possivel falar com {OLLAMA_URL}: {erro}") from erro

    try:
        return resposta.status_code, resposta.json()
    except ValueError:
        return resposta.status_code, {"error": {"message": resposta.text[:500]}}


def modelos_no_disco() -> list[str]:
    """O que o `/v1/models` do worker responde: o catalogo do Ollama."""
    try:
        resposta = _cliente.get("/api/tags", timeout=10.0)
        resposta.raise_for_status()
        modelos = resposta.json().get("models") or []
    except Exception as erro:
        logger.warning("Nao consegui listar o catalogo do Ollama: %s", erro)
        return []

    return [item.get("name", "") for item in modelos if item]


def esta_de_pe() -> bool:
    try:
        return _cliente.get("/api/version").is_success
    except Exception:
        return False
