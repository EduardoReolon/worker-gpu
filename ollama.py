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

## Dois caminhos, e como se escolhe

O Ollama fala dois dialetos, e o worker usa os dois:

    sem `options`     ->  /v1/chat/completions   (compativel, repasse cru)
    com `options`     ->  /api/chat              (nativo, traduzido)

A camada compativel desserializa o corpo num struct tipado e **descarta chave
que nao conhece, sem erro e sem log** — `options` inteiro cai nessa, e com ele
`num_ctx`. Um pedido com janela de 16k voltava 200 rodando com 4k, e o
truncamento come o prompt PELO COMECO: morre o prompt de sistema primeiro.

Por isso um pedido que traz `options` (ou `keep_alive`) vai pelo nativo, onde
esses campos existem de verdade, e `dialeto.py` traduz ida e volta. Quem nao
traz nenhum dos dois segue pelo caminho antigo, byte por byte — ha dois
clientes em producao e um deles nao precisa disto, e nao deve pagar pelo risco
de uma traducao.

`OLLAMA_KEEP_ALIVE` continua FORA do `.env` daqui, e a ausencia e deliberada:
uma variavel que vale num caminho e nao no outro e pior que variavel nenhuma. A
politica de memoria da maquina se ajusta no Ollama; um `keep_alive` mandado
pelo cliente, por vir no pedido, e honrado.
"""

from __future__ import annotations

import logging
import threading

import httpx

import dialeto
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


_trava_do_cliente = threading.Lock()
_cliente: httpx.Client | None = None


def _obter_cliente() -> httpx.Client:
    """O cliente HTTP do processo, construido uma vez, na primeira ida.

    Um por processo, e nao um por chamada: cada `httpx.Client()` monta um
    contexto SSL do zero, trabalho inutil para falar HTTP com o loopback — e
    era feito em TODA ida ao Ollama, inclusive nas duas que o `/health/` faz.

    **Preguicoso, e nao na importacao**, porque essa construcao JA FALHOU numa
    maquina de producao: `create_ssl_context` levantou `FileNotFoundError`
    procurando o pacote de certificados, e o `/health/` passou a responder 500.

    Construido na importacao, a MESMA falha viraria "o servico nao sobe" — e
    derrubaria junto a imagem e a conversao, que nao precisam do Ollama para
    nada. Preguicoso, ela vira 503 na rota de texto e `de_pe: false` no
    `/health/`: o diagnostico certo, pela parte certa, com o resto de pe.

    O timeout padrao e o CURTO, de proposito. Quem precisa de mais passa
    `timeout=` na chamada; quem esquecer herda 5s e falha rapido. O inverso —
    padrao longo, curto por chamada — faria a primeira funcao nova escrita sem
    o parametro pendurar o `/health/` por dez minutos, que e exatamente quando
    ele mais importa.
    """
    global _cliente

    if _cliente is None:
        with _trava_do_cliente:
            if _cliente is None:
                _cliente = httpx.Client(base_url=OLLAMA_URL, timeout=5.0)

    return _cliente


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
        resposta = _obter_cliente().get("/api/ps", timeout=10.0)
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
            _obter_cliente().post(
                "/api/generate", json={"model": modelo, "keep_alive": 0}, timeout=30.0
            )
        except Exception as erro:
            logger.warning("Falha ao descarregar %s: %s", modelo, erro)

    return carregados


def conversar(corpo: dict, cabecalhos: dict[str, str]) -> tuple[int, dict]:
    """Repassa um `/v1/chat/completions` ao Ollama e devolve (status, json).

    Pelo caminho compativel o corpo vai intacto e a unica intromissao e o
    `stream`. Pelo nativo ele e traduzido por `dialeto.py`, ida e volta, e a
    resposta sai na mesma forma nos dois casos — quem integra nao tem como
    saber qual caminho o pedido tomou, e nao deve precisar saber.
    """
    corpo = dict(corpo)

    # Streaming nao passa por aqui: o arbitro precisa saber quando o trabalho
    # termina para soltar a placa, e uma resposta em streaming so termina
    # quando o cliente termina de ler. Recusar e melhor que entregar pela
    # metade.
    corpo["stream"] = False

    pelo_nativo = dialeto.precisa_do_nativo(corpo)
    if pelo_nativo:
        rota, envio = "/api/chat", dialeto.para_nativo(corpo)
        logger.info(
            "Pelo dialeto nativo (options=%s): a camada compativel descartaria isso.",
            envio.get("options"),
        )
    else:
        rota, envio = "/v1/chat/completions", corpo

    try:
        resposta = _obter_cliente().post(
            rota,
            json=envio,
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
        dados = resposta.json()
    except ValueError:
        return resposta.status_code, {"error": {"message": resposta.text[:500]}}

    if not pelo_nativo:
        return resposta.status_code, dados
    if resposta.is_success:
        return resposta.status_code, dialeto.para_openai(dados, corpo.get("model"))

    # Erro do nativo normalizado para a forma da OpenAI: o mesmo erro nao pode
    # chegar ao cliente em duas formas conforme o caminho que ele nem escolheu.
    return resposta.status_code, dialeto.erro_para_openai(dados)


def modelos_no_disco() -> list[str]:
    """O que o `/v1/models` do worker responde: o catalogo do Ollama."""
    try:
        resposta = _obter_cliente().get("/api/tags", timeout=10.0)
        resposta.raise_for_status()
        modelos = resposta.json().get("models") or []
    except Exception as erro:
        logger.warning("Nao consegui listar o catalogo do Ollama: %s", erro)
        return []

    return [item.get("name", "") for item in modelos if item]


def esta_de_pe() -> bool:
    try:
        return _obter_cliente().get("/api/version").is_success
    except Exception:
        return False
