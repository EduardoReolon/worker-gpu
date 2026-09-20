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
    monkeypatch.setattr(ollama, "modelos_carregados_detalhe", list)
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


def test_a_resposta_do_health_tem_a_forma_publicada(worker, monkeypatch):
    """O `/health/` e o endpoint que todo cliente consulta para diagnosticar,
    e era o unico sem exemplo em `contrato/`.

    A consequencia apareceu em uso: quando as tres rotas viraram um servico
    so, o estado passou a vir aninhado e `busy` virou `ocupada` na raiz. Um
    cliente continuou lendo as chaves antigas, `dict.get` devolveu `None`, e
    os avisos que justificavam o diagnostico dele sumiram — sem erro nenhum,
    por semanas.

    Com o exemplo publicado, uma mudanca de forma quebra AQUI, na suite do
    worker, em vez de quebrar um cliente em producao. Em especial: trocar
    `ollama.carregados` de lista de nomes para lista de objetos passa a ser
    uma falha visivel, e nao uma surpresa no outro repositorio.
    """
    import ollama

    monkeypatch.setattr(
        ollama,
        "modelos_carregados_detalhe",
        lambda: [{"name": "qwen2.5:7b-instruct", "context_length": 4096}],
    )
    monkeypatch.setattr(ollama, "esta_de_pe", lambda: True)

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    assert _forma(_exemplo("saude-resposta.json")) <= _forma(resposta.json())


def test_o_health_nunca_responde_500(worker, monkeypatch):
    """A promessa que o exemplo de `saude-resposta.json` faz por escrito.

    Vale para todos os blocos ao mesmo tempo: e o cenario de maquina doente,
    que e exatamente quando alguem abre o `/health/`.
    """
    import conversao
    import imagem
    import ollama

    def explodir(*a, **k):
        raise RuntimeError("a maquina esta ruim")

    monkeypatch.setattr(ollama, "modelos_carregados_detalhe", explodir)
    monkeypatch.setattr(imagem, "estado", explodir)
    monkeypatch.setattr(conversao, "estado", explodir)

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ok"
    assert all("erro" in corpo[bloco] for bloco in ("ollama", "imagem", "conversao"))


def test_o_timeout_e_o_unico_503_sem_retry_after(worker, cabecalhos, monkeypatch):
    """A regra que `ocupada-resposta.json` publica: todo 503 tem
    `error.code`, e todos trazem `Retry-After` menos o `timeout`."""
    import ollama

    def estourar(*a, **k):
        raise ollama.OllamaDemorouDemais("passou do orcamento")

    monkeypatch.setattr(ollama, "conversar", estourar)
    corpo = {"messages": [{"role": "user", "content": "oi"}]}
    cliente_http = TestClient(worker.app)

    demorou = cliente_http.post("/v1/chat/completions", json=corpo, headers=cabecalhos)
    assert demorou.json()["error"]["code"] == "timeout"
    assert "Retry-After" not in demorou.headers

    monkeypatch.setattr(
        ollama, "conversar", lambda c, cab: (_ for _ in ()).throw(ollama.OllamaIndisponivel("fora"))
    )
    caiu = cliente_http.post("/v1/chat/completions", json=corpo, headers=cabecalhos)
    assert caiu.json()["error"]["code"] == "ollama_indisponivel"
    assert "Retry-After" in caiu.headers


def test_a_forma_publicada_e_a_mesma_pelos_dois_caminhos(worker, cabecalhos, monkeypatch):
    """A garantia central da traducao: quem integra copiou UM exemplo e nao
    tem como saber por qual dialeto o pedido dele foi.

    Um pedido com `options` vai pelo `/api/chat`, cuja resposta tem outra forma
    (`message` na raiz, `prompt_eval_count` em vez de `usage`). Se a traducao
    de volta divergir em um campo, o cliente quebra sem nada no worker acusar —
    e quebra so nos pedidos que pedem janela de contexto.
    """
    import httpx

    import ollama

    nativa = {
        "model": "qwen2.5:7b-instruct",
        "message": {"role": "assistant", "content": "O texto gerado."},
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 340,
    }
    compativel = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "qwen2.5:7b-instruct",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def post(url, **kwargs):
        return httpx.Response(200, json=nativa if url == "/api/chat" else compativel)

    monkeypatch.setattr(ollama._cliente, "post", post)
    cliente_http = TestClient(worker.app)
    esperada = _forma(_exemplo("texto-resposta.json"))

    pelo_antigo = cliente_http.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:7b-instruct", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )
    pelo_nativo = cliente_http.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5:7b-instruct",
            "messages": [{"role": "user", "content": "oi"}],
            "options": {"num_ctx": 16384},
        },
        headers=cabecalhos,
    )

    assert esperada <= _forma(pelo_antigo.json())
    assert esperada <= _forma(pelo_nativo.json())
