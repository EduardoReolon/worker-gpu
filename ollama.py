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
"""

from __future__ import annotations

import logging

import httpx

from config import OLLAMA_KEEP_ALIVE, OLLAMA_TIMEOUT, OLLAMA_URL

logger = logging.getLogger("worker-gpu.ollama")


class OllamaIndisponivel(RuntimeError):
    """Nao foi possivel falar com o Ollama."""


def modelos_carregados() -> list[str]:
    """Os modelos que estao na memoria AGORA (`/api/ps`).

    Diferente de `/api/tags`, que lista o que existe no disco. A diferenca e o
    ponto: o que ocupa VRAM e o que esta carregado.
    """
    try:
        with httpx.Client(timeout=10.0) as cliente:
            resposta = cliente.get(f"{OLLAMA_URL}/api/ps")
        resposta.raise_for_status()
    except httpx.HTTPError as erro:
        logger.warning("Nao consegui listar os modelos carregados: %s", erro)
        return []

    return [item.get("name", "") for item in (resposta.json().get("models") or []) if item]


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
            with httpx.Client(timeout=30.0) as cliente:
                # Sem `prompt` e com `keep_alive: 0`: o Ollama entende como
                # "carregue nada e esqueca este modelo".
                cliente.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": modelo, "keep_alive": 0},
                )
        except httpx.HTTPError as erro:
            logger.warning("Falha ao descarregar %s: %s", modelo, erro)

    return carregados


def conversar(corpo: dict, cabecalhos: dict[str, str]) -> tuple[int, dict]:
    """Repassa um `/v1/chat/completions` ao Ollama e devolve (status, json).

    O corpo vai quase intacto: o que o cliente pediu e o que o modelo recebe.
    A unica intromissao e o `keep_alive`, quando configurado — e mesmo essa e
    opcional, porque ela pertence a politica de memoria desta maquina e nao ao
    pedido de quem chamou.
    """
    corpo = dict(corpo)
    if OLLAMA_KEEP_ALIVE:
        corpo.setdefault("keep_alive", OLLAMA_KEEP_ALIVE)

    # Streaming nao passa por aqui: o arbitro precisa saber quando o trabalho
    # termina para soltar a placa, e uma resposta em streaming so termina
    # quando o cliente termina de ler. Recusar e melhor que entregar pela
    # metade.
    corpo["stream"] = False

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as cliente:
            resposta = cliente.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=corpo,
                headers={"Content-Type": "application/json"},
            )
    except httpx.HTTPError as erro:
        raise OllamaIndisponivel(f"nao foi possivel falar com {OLLAMA_URL}: {erro}") from erro

    try:
        return resposta.status_code, resposta.json()
    except ValueError:
        return resposta.status_code, {"error": {"message": resposta.text[:500]}}


def modelos_no_disco() -> list[str]:
    """O que o `/v1/models` do worker responde: o catalogo do Ollama."""
    try:
        with httpx.Client(timeout=10.0) as cliente:
            resposta = cliente.get(f"{OLLAMA_URL}/api/tags")
        resposta.raise_for_status()
    except httpx.HTTPError:
        return []

    return [item.get("name", "") for item in (resposta.json().get("models") or []) if item]


def esta_de_pe() -> bool:
    try:
        with httpx.Client(timeout=5.0) as cliente:
            return cliente.get(f"{OLLAMA_URL}/api/version").is_success
    except httpx.HTTPError:
        return False
