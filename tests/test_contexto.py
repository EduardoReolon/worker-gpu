"""A janela de contexto: o cabecalho, a regra de nunca baixar, o teto, e o
400 na forma do llama.cpp."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

MENSAGENS = [{"role": "user", "content": "oi"}]


@pytest.fixture
def contexto_mod(worker):
    import contexto

    return contexto


@pytest.fixture
def ollama_falso(worker, monkeypatch):
    """O `/api/chat` do Ollama, gravando o que recebe."""
    import ollama

    vistos = []
    respostas = []

    def post(url, **kwargs):
        vistos.append({"url": url, "json": kwargs["json"]})
        if respostas:
            return respostas.pop(0)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop",
                "prompt_eval_count": 10,
                "eval_count": 1,
            },
        )

    monkeypatch.setattr(ollama._obter_cliente(), "post", post)
    return vistos, respostas


def _pedir(worker, cabecalhos, contexto=None, **corpo):
    extras = {} if contexto is None else {"X-PubliBot-Contexto": contexto}
    return TestClient(worker.app).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": MENSAGENS, **corpo},
        headers={**cabecalhos, **extras},
    )


def test_o_cabecalho_vira_num_ctx_pela_api_nativa(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso

    resposta = _pedir(worker, cabecalhos, contexto="24576", max_tokens=8, temperature=0.2)

    assert resposta.status_code == 200
    assert resposta.json()["choices"][0]["message"]["content"] == "ok"
    assert vistos[0]["url"] == "/api/chat"
    assert vistos[0]["json"]["options"] == {"num_ctx": 24576, "num_predict": 8, "temperature": 0.2}


def test_sem_cabecalho_vale_o_padrao(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos)

    assert vistos[0]["json"]["options"]["num_ctx"] == 16384


@pytest.mark.parametrize("ruim", ["", "muito", "-5", "0"])
def test_cabecalho_invalido_vale_o_padrao(worker, cabecalhos, ollama_falso, ruim):
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos, contexto=ruim)

    assert vistos[0]["json"]["options"]["num_ctx"] == 16384


def test_a_janela_nunca_baixa_entre_pedidos(worker, cabecalhos, ollama_falso):
    """Cada troca de `num_ctx` recarrega o modelo: um pedido pequeno depois de
    um grande reaproveita a carga."""
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos, contexto="28672")
    _pedir(worker, cabecalhos, contexto="16384")

    assert [v["json"]["options"]["num_ctx"] for v in vistos] == [28672, 28672]


def test_a_janela_e_por_modelo(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos, contexto="28672")
    _pedir(worker, cabecalhos, contexto="16384", model="outro")

    assert vistos[1]["json"]["options"]["num_ctx"] == 16384


def test_a_janela_nunca_passa_do_teto(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos, contexto="131072")

    assert vistos[0]["json"]["options"]["num_ctx"] == 32768


def test_o_num_ctx_do_cliente_tambem_conta(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso

    _pedir(worker, cabecalhos, options={"num_ctx": 20480})

    assert vistos[0]["json"]["options"]["num_ctx"] == 20480


def test_response_format_vira_format(worker, cabecalhos, ollama_falso):
    vistos, _ = ollama_falso
    esquema = {"type": "object", "properties": {"a": {"type": "string"}}}

    _pedir(
        worker,
        cabecalhos,
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": esquema}},
    )

    assert vistos[0]["json"]["format"] == esquema


def test_nao_coube_nem_no_teto_e_400_exceed_context_size_error(worker, cabecalhos, ollama_falso):
    _, respostas = ollama_falso
    respostas.append(
        httpx.Response(
            400,
            json={
                "error": "request (40000 tokens) exceeds the available context size (32768 tokens)"
            },
        )
    )

    resposta = _pedir(worker, cabecalhos, contexto="65536")

    assert resposta.status_code == 400
    erro = resposta.json()["error"]
    assert erro["type"] == "exceed_context_size_error"
    assert (erro["n_prompt_tokens"], erro["n_ctx"]) == (40000, 32768)
    assert "40000 tokens" in erro["message"]


def test_outro_400_vai_como_veio(worker, cabecalhos, ollama_falso):
    _, respostas = ollama_falso
    respostas.append(httpx.Response(400, json={"error": "invalid json schema"}))

    resposta = _pedir(worker, cabecalhos)

    assert resposta.status_code == 400
    assert resposta.json() == {"error": {"message": "invalid json schema"}}


def test_o_health_declara_o_contexto(worker, monkeypatch):
    import ollama

    monkeypatch.setattr(
        ollama, "modelos_carregados_detalhe", lambda: [{"name": "m", "context_length": 24576}]
    )
    monkeypatch.setattr(ollama, "esta_de_pe", lambda: True)

    bloco = TestClient(worker.app).get("/health/").json()["ollama"]["contexto"]

    assert bloco == {"padrao": 16384, "atual": 24576, "maximo": 32768, "segue_cabecalho": True}


def test_sem_modelo_carregado_o_atual_e_zero(contexto_mod):
    assert contexto_mod.estado([])["atual"] == 0
