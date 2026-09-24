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
    monkeypatch.setattr(ollama, "modelos_carregados_detalhe", list)
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


def test_lado_que_nao_e_multiplo_de_dezesseis_e_recusado(cliente, cabecalhos):
    """16 e nao 8: os modelos de transformer agrupam o latente em blocos de
    2x2, e 1352 (multiplo de 8) passaria aqui para o pipeline recusar la
    dentro, com um 500."""
    resposta = cliente.post(
        "/v1/images/generations",
        json={"prompt": "x", "size": "1352x760"},
        headers=cabecalhos,
    )

    assert resposta.status_code == 422
    assert "multiplo de 16" in resposta.json()["detail"]


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


# ---------------------------------------------------------------------------
# Adiar ou desistir: o que o 503 diz ao cliente
# ---------------------------------------------------------------------------
def test_o_estouro_de_orcamento_vira_timeout_sem_retry_after(cliente, cabecalhos, monkeypatch):
    """Antes isto chegava como `ollama_indisponivel` — "o Ollama caiu", que e
    transitorio — e o cliente reagendava um pedido que nunca ia caber, para
    sempre. O codigo `timeout` ja estava documentado; era so inalcancavel.

    E vai SEM `Retry-After`: para esse codigo a orientacao e reduzir o pedido.
    Mandar o cabecalho junto seria o contrato se contradizendo dentro da mesma
    resposta, e cabecalho ausente e mais dificil de obedecer por engano do que
    cabecalho presente que a doc pede para ignorar.
    """
    import ollama

    def estourar(*a, **k):
        raise ollama.OllamaDemorouDemais("passou dos 540s")

    monkeypatch.setattr(ollama, "conversar", estourar)

    resposta = cliente.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "timeout"
    assert "Retry-After" not in resposta.headers


@pytest.mark.parametrize("status", [502, 503, 504])
def test_o_5xx_do_ollama_ganha_codigo_e_retry_after(cliente, cabecalhos, monkeypatch, status):
    """Repassado cru, um 503 do upstream chegava sem `error.code` e sem
    `Retry-After` — a unica forma de 503 que saia daqui sem nada para decidir.
    Um cliente que ramifica por codigo lia "adiavel, desconhecido" e
    reagendava para sempre.

    502 e 504 entram junto porque `OLLAMA_URL` aceita qualquer endereco:
    apontando para um proxy reverso, sao esses que aparecem no lugar do 503.
    """
    import ollama

    monkeypatch.setattr(
        ollama, "conversar", lambda corpo, cab: (status, {"error": {"message": "server busy"}})
    )

    resposta = cliente.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "ollama_indisponivel"
    assert int(resposta.headers["Retry-After"]) > 0
    # O envelope troca a FORMA, nunca o diagnostico.
    assert "server busy" in resposta.json()["error"]["message"]


@pytest.mark.parametrize(
    "status,corpo",
    [
        (404, {"error": {"message": 'model "x" not found, try pulling it first'}}),
        (400, {"error": {"message": "invalid json schema"}}),
    ],
)
def test_o_4xx_do_ollama_passa_verbatim(cliente, cabecalhos, monkeypatch, status, corpo):
    """ "Seu pedido esta errado" e informacao, e o cliente precisa dela para
    desistir em vez de reagendar. Traduzir para 503 apagaria o que o Ollama
    tinha a dizer e transformaria um erro permanente em fila eterna."""
    import ollama

    monkeypatch.setattr(ollama, "conversar", lambda c, cab: (status, corpo))

    resposta = cliente.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.status_code == status
    assert resposta.json() == corpo


# ---------------------------------------------------------------------------
# O `/health/` nunca devolve 500
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "modulo,funcao,bloco", [("imagem", "estado", "imagem"), ("conversao", "estado", "conversao")]
)
def test_um_bloco_que_estoura_vira_campo_e_nao_500(worker, monkeypatch, modulo, funcao, bloco):
    """E o endpoint de diagnostico: uma excecao nele transforma "alguma coisa
    esta errada" em "tudo esta errado e nao sei o que". O processo esta de pe,
    e e isso que o 200 afirma."""
    import importlib

    alvo = importlib.import_module(modulo)

    def explodir():
        raise RuntimeError("faltou alguma coisa")

    monkeypatch.setattr(alvo, funcao, explodir)

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    assert resposta.json()[bloco]["erro"] == "RuntimeError: faltou alguma coisa"


def test_o_bloco_do_ollama_que_estoura_tambem(worker, monkeypatch):
    """`resposta.json()` levanta `ValueError`, que fica de fora de
    `httpx.HTTPError` — era por aqui que o `/health/` caia."""
    import ollama

    def explodir():
        raise ValueError("json invalido do /api/ps")

    monkeypatch.setattr(ollama, "modelos_carregados_detalhe", explodir)

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    assert resposta.json()["ollama"]["erro"] == "ValueError: json invalido do /api/ps"


def test_mesmo_com_um_bloco_quebrado_o_resto_continua_legivel(worker, monkeypatch):
    """Um diagnostico so serve se os campos que AINDA funcionam aparecerem.
    Em especial `ha_segundos`, que e o alarme de lock preso."""
    import conversao

    monkeypatch.setattr(conversao, "estado", lambda: (_ for _ in ()).throw(RuntimeError("x")))

    corpo = TestClient(worker.app).get("/health/").json()

    assert corpo["status"] == "ok"
    assert corpo["ocupada"] is False
    assert corpo["ha_segundos"] == 0
    assert corpo["contrato_versao"]


def test_o_health_nao_paga_duas_viagens_ao_api_ps(worker, monkeypatch):
    """`carregados` e `carregados_detalhe` saem da MESMA consulta. Duas idas
    ao Ollama por `/health/` seriam trabalho dobrado num endpoint que o
    systemd consulta em laco."""
    import ollama

    idas = []
    monkeypatch.setattr(
        ollama,
        "modelos_carregados_detalhe",
        lambda: idas.append(1) or [{"name": "qwen2.5:7b-instruct", "context_length": 4096}],
    )

    corpo = TestClient(worker.app).get("/health/").json()

    assert len(idas) == 1
    assert corpo["ollama"]["carregados"] == ["qwen2.5:7b-instruct"]
    assert corpo["ollama"]["carregados_detalhe"][0]["context_length"] == 4096


# ---------------------------------------------------------------------------
# Janela de contexto: qual dialeto o pedido toma
# ---------------------------------------------------------------------------
def test_o_num_ctx_do_cliente_chega_ao_ollama(worker, cabecalhos, monkeypatch):
    """O defeito que a traducao existe para consertar: pela camada compativel
    isto respondia 200 tendo rodado com a janela padrao, e o truncamento come o
    prompt de sistema pelo comeco."""
    import httpx

    import ollama

    visto = {}

    def post(url, **kwargs):
        visto["url"] = url
        visto["json"] = kwargs["json"]
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "ok"},
                "prompt_eval_count": 9000,
                "eval_count": 5,
            },
        )

    monkeypatch.setattr(ollama._obter_cliente(), "post", post)

    resposta = TestClient(worker.app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "oi"}],
            "options": {"num_ctx": 16384},
            "max_tokens": 512,
        },
        headers=cabecalhos,
    )

    assert visto["url"] == "/api/chat"
    assert visto["json"]["options"]["num_ctx"] == 16384
    assert visto["json"]["options"]["num_predict"] == 512
    # E o `prompt_tokens` volta, que e como o cliente detecta truncamento.
    assert resposta.json()["usage"]["prompt_tokens"] == 9000


def test_sem_options_o_caminho_antigo_nao_muda(worker, cabecalhos, monkeypatch):
    """Ha dois clientes em producao e um deles nao manda `options`. Ele nao
    deve pagar pelo risco de uma traducao que nao pediu."""
    import httpx

    import ollama

    visto = {}

    def post(url, **kwargs):
        visto["url"] = url
        visto["json"] = kwargs["json"]
        return httpx.Response(200, json={"choices": [], "usage": {}})

    monkeypatch.setattr(ollama._obter_cliente(), "post", post)

    TestClient(worker.app).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "oi"}], "temperature": 0.2},
        headers=cabecalhos,
    )

    assert visto["url"] == "/v1/chat/completions"
    assert visto["json"] == {
        "model": "m",
        "messages": [{"role": "user", "content": "oi"}],
        "temperature": 0.2,
        "stream": False,
    }


def test_um_erro_do_nativo_sai_na_forma_da_openai(worker, cabecalhos, monkeypatch):
    """O nativo devolve `{"error": "texto"}`. Sem normalizar, o mesmo modelo
    inexistente chegaria ao cliente em duas formas conforme ele ter mandado
    `options` ou nao."""
    import httpx

    import ollama

    monkeypatch.setattr(
        ollama._obter_cliente(),
        "post",
        lambda url, **k: httpx.Response(404, json={"error": 'model "x" not found'}),
    )

    resposta = TestClient(worker.app).post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "oi"}],
            "options": {"num_ctx": 8192},
        },
        headers=cabecalhos,
    )

    assert resposta.status_code == 404
    assert resposta.json()["error"]["message"] == 'model "x" not found'


# ---------------------------------------------------------------------------
# Memoria: o que o worker devolve quando para de trabalhar
# ---------------------------------------------------------------------------
def test_o_docling_e_descarregado_depois_de_ocioso(worker, cabecalhos, monkeypatch):
    """Ele ficava residente PARA SEMPRE depois da primeira conversao: a rota
    de imagem tinha o seu temporizador de descarga e esta nao tinha nada.

    Sao 1 a 2 GB de memoria anonima parada — e e a anonima parada que o kernel
    escreve no swap quando precisa de paginas, porque ela nao pode ser
    descartada de graca como um arquivo mapeado.
    """
    import conversao

    agendados = []
    monkeypatch.setattr(conversao, "CONVERSAO_OCIOSO_SEGUNDOS", 900)
    monkeypatch.setattr(
        conversao.threading,
        "Timer",
        lambda segundos, funcao: (
            agendados.append((segundos, funcao))
            or type(
                "Falso",
                (),
                {"start": lambda self: None, "cancel": lambda self: None, "daemon": True},
            )()
        ),
    )

    TestClient(worker.app).post(
        "/parse/", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=cabecalhos
    )

    assert agendados and agendados[0][0] == 900
    assert agendados[0][1] is conversao.descarregar


def test_a_descarga_e_agendada_mesmo_quando_a_conversao_falha(worker, cabecalhos, monkeypatch):
    """Um PDF que falhou deixa o conversor carregado do mesmo jeito, e a
    memoria dele precisa ser devolvida igual. Por isso o `finally`."""
    import conversao

    monkeypatch.setattr(
        conversao, "obter_conversor", lambda: (_ for _ in ()).throw(RuntimeError("pdf ruim"))
    )
    agendados = []
    monkeypatch.setattr(conversao, "_agendar_descarga", lambda: agendados.append(True))

    resposta = TestClient(worker.app).post(
        "/parse/", files={"file": ("a.pdf", PDF, "application/pdf")}, headers=cabecalhos
    )

    assert resposta.status_code == 500
    assert agendados == [True]


def test_descarregar_o_docling_e_idempotente(worker):
    """Chamado pelo temporizador, e o temporizador pode disparar depois de uma
    descarga manual."""
    import conversao

    conversao.descarregar()
    conversao.descarregar()

    assert conversao.estado()["carregado"] is False


def test_com_ocioso_zero_o_docling_fica_residente(worker, monkeypatch):
    """Era o comportamento anterior, e continua alcancavel: num acervo grande,
    pagando dezenas de segundos de recarga, manter residente pode compensar."""
    import conversao

    monkeypatch.setattr(conversao, "CONVERSAO_OCIOSO_SEGUNDOS", 0)
    criados = []
    monkeypatch.setattr(conversao.threading, "Timer", lambda *a, **k: criados.append(True))

    conversao._agendar_descarga()

    assert criados == []
