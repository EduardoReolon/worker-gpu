"""As rotas, pelo HTTP — que e a unica forma como alguem as usa.

Os modelos pesados nao entram: `gerar_imagens`, `obter_conversor` e o Ollama
sao substituidos. O que sobra e o contrato, que e onde os erros deste
repositorio doem: do outro lado ha clientes que so veem JSON.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from fastapi.testclient import TestClient

PDF = b"%PDF-1.4 fingindo ser um artigo"


@pytest.fixture
def cliente(worker, monkeypatch):
    import conversao
    import imagem
    import ollama

    monkeypatch.setattr(
        imagem,
        "gerar_imagens",
        lambda dispositivo, pedido, quantas, largura, altura: [
            f"png-{i}-{largura}x{altura}".encode() for i in range(quantas)
        ],
    )
    monkeypatch.setattr(conversao, "obter_conversor", lambda: _ConversorFalso())
    monkeypatch.setattr(ollama, "conversar", lambda corpo, cab: (200, _RESPOSTA_DE_TEXTO))
    monkeypatch.setattr(ollama, "modelos_no_disco", lambda: ["qwen2.5:7b-instruct"])
    monkeypatch.setattr(ollama, "modelos_carregados", list)
    monkeypatch.setattr(ollama, "esta_de_pe", lambda: True)
    return TestClient(worker.app)


_RESPOSTA_DE_TEXTO = {
    "id": "chatcmpl-1",
    "model": "qwen2.5:7b-instruct",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ola."}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}


class _ConversorFalso:
    def convert(self, caminho):
        class Documento:
            @staticmethod
            def export_to_markdown():
                return "# Titulo\n\nUm paragrafo."

        class Resultado:
            document = Documento()

        return Resultado()


# ---------------------------------------------------------------------------
# Credencial
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rota,metodo,corpo",
    [
        ("/v1/chat/completions", "post", {"messages": [{"role": "user", "content": "oi"}]}),
        ("/v1/images/generations", "post", {"prompt": "um gato"}),
        ("/v1/models", "get", None),
    ],
)
def test_toda_rota_exige_credencial(cliente, rota, metodo, corpo):
    """Um endpoint que roda modelo na sua placa, aberto, e placa de graca para
    quem achar a porta."""
    resposta = getattr(cliente, metodo)(rota, **({"json": corpo} if corpo else {}))

    assert resposta.status_code == 401


def test_parse_tambem_exige(cliente):
    resposta = cliente.post("/parse/", files={"file": ("a.pdf", PDF, "application/pdf")})

    assert resposta.status_code == 401


def test_aceita_bearer_e_cabecalho_proprio(cliente, cabecalhos, segredo):
    """O `Bearer` porque e o que os clientes do dialeto OpenAI mandam
    sozinhos; o proprio porque encurta um `curl` de diagnostico. Duas rotas do
    mesmo processo exigindo cabecalhos diferentes seria armadilha."""
    corpo = {"messages": [{"role": "user", "content": "oi"}]}

    assert cliente.post("/v1/chat/completions", json=corpo, headers=cabecalhos).status_code == 200
    assert (
        cliente.post(
            "/v1/chat/completions", json=corpo, headers={"X-Worker-Secret": segredo}
        ).status_code
        == 200
    )


def test_health_nao_exige_credencial(cliente):
    """E o endpoint que o instalador, o systemd e quem diagnostica consultam.
    Exigir segredo ai transforma "o servico caiu" em "caiu, ou o segredo esta
    errado"."""
    assert cliente.get("/health/").status_code == 200


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
def test_texto_repassa_e_devolve_o_formato_da_openai(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/chat/completions",
        json={"model": "qwen2.5:7b-instruct", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.json()["choices"][0]["message"]["content"] == "Ola."


def test_texto_sem_mensagens_e_recusado(cliente, cabecalhos):
    assert cliente.post("/v1/chat/completions", json={}, headers=cabecalhos).status_code == 422


def test_streaming_e_recusado_com_explicacao(cliente, cabecalhos):
    """Ignorar em silencio seria pior: quem pediu streaming espera pedacos, e
    receber tudo de uma vez quebra a leitura dele de um jeito dificil de
    diagnosticar."""
    resposta = cliente.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}], "stream": True},
        headers=cabecalhos,
    )

    assert resposta.status_code == 422
    assert "stream" in resposta.json()["detail"]


def test_ollama_fora_do_ar_vira_503(cliente, cabecalhos, monkeypatch):
    """503 e nao 500: e transitorio, e os clientes ja sabem esperar."""
    import ollama

    def cair(*a, **k):
        raise ollama.OllamaIndisponivel("connection refused")

    monkeypatch.setattr(ollama, "conversar", cair)

    resposta = cliente.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "ollama_indisponivel"


# ---------------------------------------------------------------------------
# Imagem
# ---------------------------------------------------------------------------
def test_imagem_devolve_b64_no_formato_da_openai(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/images/generations",
        json={"prompt": "um gato", "n": 3, "size": "1024x576"},
        headers=cabecalhos,
    )

    corpo = resposta.json()
    assert len(corpo["data"]) == 3
    assert base64.b64decode(corpo["data"][0]["b64_json"]) == b"png-0-1024x576"


def test_lado_que_nao_e_multiplo_de_oito_e_recusado(cliente, cabecalhos):
    """Nao falharia: o modelo arredonda por dentro e devolve uma imagem de
    tamanho diferente do pedido, sem avisar."""
    resposta = cliente.post(
        "/v1/images/generations",
        json={"prompt": "x", "size": "1000x1001"},
        headers=cabecalhos,
    )

    assert resposta.status_code == 422
    assert "multiplo de 8" in resposta.json()["detail"]


def test_pedido_maior_que_o_teto_e_aparado(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/images/generations", json={"prompt": "x", "n": 50}, headers=cabecalhos
    )

    assert len(resposta.json()["data"]) == 4


def test_o_ollama_e_descarregado_antes_de_gerar(worker, cabecalhos, monkeypatch):
    """O motivo de o arbitro existir. Com o lock na mao ninguem esta gerando
    texto, entao mandar o Ollama soltar a VRAM e seguro — e sem isso a
    difusao encontra a placa cheia."""
    import imagem
    import ollama

    monkeypatch.setattr(imagem, "OLLAMA_DESCARREGAR_PARA_IMAGEM", True)
    monkeypatch.setattr(
        imagem, "gerar_imagens", lambda dispositivo, pedido, quantas, largura, altura: [b"png"]
    )

    descarregou = []
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: descarregou.append(True) or [])

    TestClient(worker.app).post("/v1/images/generations", json={"prompt": "x"}, headers=cabecalhos)

    assert descarregou == [True]


# ---------------------------------------------------------------------------
# Conversao
# ---------------------------------------------------------------------------
def test_conversao_devolve_markdown_e_digest(cliente, cabecalhos):
    resposta = cliente.post(
        "/parse/", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=cabecalhos
    )

    corpo = resposta.json()
    assert corpo["markdown"].startswith("# Titulo")
    assert corpo["sha256"] == hashlib.sha256(PDF).hexdigest()


def test_digest_divergente_e_recusado(cliente, cabecalhos):
    """Um arquivo truncado no caminho converteria em silencio, e o Markdown de
    um documento que ninguem pediu entraria no acervo."""
    resposta = cliente.post(
        "/parse/",
        files={"file": ("a.pdf", PDF, "application/pdf")},
        headers={**cabecalhos, "X-Expected-Sha256": "0" * 64},
    )

    assert resposta.status_code == 422


def test_arquivo_grande_demais_e_recusado(worker, cabecalhos, monkeypatch):
    import conversao

    monkeypatch.setattr(conversao, "MAX_PDF_BYTES", 10)

    resposta = TestClient(worker.app).post(
        "/parse/", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=cabecalhos
    )

    assert resposta.status_code == 413
