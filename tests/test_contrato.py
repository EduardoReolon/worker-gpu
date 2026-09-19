"""O que os clientes dependem, conferido contra os exemplos publicados.

Este arquivo existe por causa da separacao dos repositorios. Antes, o teste
de contrato rodava o cliente de VERDADE do PubliBot contra este app, no mesmo
processo — e era ele que impedia os dois lados de divergirem num nome de
campo. Com os repositorios separados, esse teste nao pode mais existir, e a
divergencia voltaria a ser possivel com as duas suites verdes.

A substituicao sao os exemplos em `contrato/`, conferidos dos dois lados:
aqui, contra a resposta real; no cliente, contra o adaptador dele.

O que se compara e a FORMA — as chaves e os tipos —, nao os valores. Comparar
valores obrigaria a atualizar o exemplo a cada mudanca de conteudo, e um
exemplo que se atualiza sozinho nao prova nada.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CONTRATO = Path(__file__).resolve().parent.parent / "contrato"


def _exemplo(nome: str) -> dict:
    dados = json.loads((CONTRATO / nome).read_text(encoding="utf-8"))
    # As chaves com `_` sao anotacao para quem le, e nao parte da resposta.
    return {chave: valor for chave, valor in dados.items() if not chave.startswith("_")}


def _forma(valor, caminho: str = "") -> set[str]:
    """As chaves e os tipos, achatados. `{"a": {"b": 1}}` -> `{"a.b:int"}`.

    Listas sao representadas pelo PRIMEIRO item: o contrato diz o formato de
    um elemento, e uma lista vazia no exemplo nao descreveria nada.
    """
    if isinstance(valor, dict):
        marcas = set()
        for chave, dentro in valor.items():
            marcas |= _forma(dentro, f"{caminho}.{chave}" if caminho else chave)
        return marcas
    if isinstance(valor, list):
        return _forma(valor[0], f"{caminho}[]") if valor else {f"{caminho}[]:vazio"}
    return {f"{caminho}:{type(valor).__name__}"}


@pytest.fixture
def cliente(worker, monkeypatch):
    import conversao
    import imagem
    import ollama

    monkeypatch.setattr(
        imagem, "gerar_imagens", lambda dispositivo, pedido, quantas, largura, altura: [b"\x89PNG"]
    )
    monkeypatch.setattr(
        ollama,
        "conversar",
        lambda corpo, cab: (
            200,
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "qwen2.5:7b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Ola."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    monkeypatch.setattr(conversao, "obter_conversor", lambda: _Conversor())
    monkeypatch.setattr(ollama, "modelos_carregados", list)
    monkeypatch.setattr(ollama, "esta_de_pe", lambda: True)
    return TestClient(worker.app)


class _Conversor:
    def convert(self, caminho):
        class D:
            @staticmethod
            def export_to_markdown():
                return "# T\n\np."

        class R:
            document = D()

        return R()


def test_a_resposta_de_texto_tem_a_forma_publicada(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert _forma(_exemplo("texto-resposta.json")) <= _forma(resposta.json())


def test_a_resposta_de_imagem_tem_a_forma_publicada(cliente, cabecalhos):
    resposta = cliente.post("/v1/images/generations", json={"prompt": "x"}, headers=cabecalhos)

    assert _forma(_exemplo("imagem-resposta.json")) <= _forma(resposta.json())


def test_a_resposta_de_conversao_tem_a_forma_publicada(cliente, cabecalhos):
    resposta = cliente.post(
        "/parse/", files={"file": ("a.pdf", b"%PDF", "application/pdf")}, headers=cabecalhos
    )

    assert _forma(_exemplo("conversao-resposta.json")) <= _forma(resposta.json())


def test_o_503_tem_a_forma_publicada(worker, cabecalhos):
    """O corpo do 503 e contrato como qualquer outro: `error.code` e o campo
    que o cliente le para decidir entre esperar e desistir.

    A ocupacao aqui carrega `modelo` porque as rotas de verdade carregam: o
    texto passa o que o cliente pediu, a imagem passa o `IMAGEM_MODELO`.
    """
    from arbitro import ARBITRO

    with ARBITRO.usar("texto", modelo="qwen2.5:7b-instruct"):
        resposta = TestClient(worker.app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "oi"}]},
            headers=cabecalhos,
        )

    assert resposta.status_code == 503
    assert _forma(_exemplo("ocupada-resposta.json")) <= _forma(resposta.json())
    assert int(resposta.headers["Retry-After"]) > 0


def test_o_503_diz_qual_modelo_esta_na_placa(worker, cabecalhos):
    """E o campo que decide o que o cliente manda EM SEGUIDA. Sem ele, quem
    leva a recusa so sabe "volte em 18s" — e pode voltar com um pedido de
    outro modelo, pagando uma troca que daria para evitar."""
    from arbitro import ARBITRO

    with ARBITRO.usar("texto", modelo="qwen2.5:7b-instruct"):
        resposta = TestClient(worker.app).post(
            "/v1/chat/completions",
            json={"model": "outro", "messages": [{"role": "user", "content": "oi"}]},
            headers=cabecalhos,
        )

    assert resposta.json()["error"]["modelo"] == "qwen2.5:7b-instruct"


def test_sem_modelo_conhecido_o_campo_nao_aparece(worker, cabecalhos):
    """`"modelo": null` seria lido como "a placa esta limpa", que e o
    contrario do que um 503 diz. Ausente e a unica forma honesta de dizer
    "nao sei"."""
    from arbitro import ARBITRO

    with ARBITRO.usar("conversao"):
        resposta = TestClient(worker.app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "oi"}]},
            headers=cabecalhos,
        )

    assert "modelo" not in resposta.json()["error"]


def test_o_modelo_pedido_pelo_cliente_chega_ao_ollama_e_ao_arbitro(worker, cabecalhos, monkeypatch):
    """Cada cliente escolhe o seu — o CRM tem um modelo por tenant. O worker
    nao impoe nem substitui: ele arbitra a placa, nao a escolha.

    E o mesmo nome precisa chegar aos DOIS lugares. Se ele so chegasse ao
    Ollama, o 503 de quem tentou ao mesmo tempo nao saberia dizer o que esta
    na placa — que e a unica razao de o arbitro guardar isso.
    """
    import ollama
    from arbitro import ARBITRO

    visto = {}

    def conversar(corpo, cabecalhos):
        visto["pedido"] = corpo.get("model")
        ocupacao = ARBITRO.ocupacao
        visto["no_arbitro"] = ocupacao.modelo if ocupacao else None
        return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(ollama, "conversar", conversar)

    TestClient(worker.app).post(
        "/v1/chat/completions",
        json={"model": "llama3.1:70b", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert visto["pedido"] == "llama3.1:70b"
    assert visto["no_arbitro"] == "llama3.1:70b"


def test_o_health_diz_qual_modelo_esta_em_uso(worker, cabecalhos):
    """Para quem planeja um lote: uma consulta, e a escolha dos proximos
    pedidos sai dela."""
    from arbitro import ARBITRO

    cliente = TestClient(worker.app)

    assert cliente.get("/health/").json()["modelo"] is None

    with ARBITRO.usar("texto", modelo="qwen2.5:7b-instruct"):
        assert cliente.get("/health/").json()["modelo"] == "qwen2.5:7b-instruct"


def test_a_versao_do_contrato_aparece_no_health(cliente):
    """Sem um numero publicado, a copia velha do exemplo no cliente nao tem
    como se denunciar."""
    import json as _json

    publicada = cliente.get("/health/").json()["contrato_versao"]
    nos_exemplos = {
        _json.loads(arquivo.read_text(encoding="utf-8"))["_contrato_versao"]
        for arquivo in CONTRATO.glob("*.json")
    }

    assert nos_exemplos == {publicada}
