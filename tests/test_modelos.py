"""O download automatico, e as tres defesas contra o laco infinito.

O modo de este recurso dar muito errado e sempre o mesmo: o cliente pede X, o
worker acha que X nao existe, baixa, e continua achando que X nao existe. Sao
downloads em laco ate o disco acabar — e num servico que os clientes sabem
reagendar, isso roda por dias sem ninguem olhar.

Cada defesa tem teste proprio aqui, e e por isso que este arquivo existe.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def baixando(worker, monkeypatch):
    """O modulo com o download LIGADO e rodando sincrono.

    Sincrono porque o que se testa e a decisao, nao a concorrencia: uma thread
    de verdade tornaria o teste uma corrida.
    """
    import modelos

    monkeypatch.setattr(modelos, "BAIXAR_MODELO_AUTOMATICO", True)
    monkeypatch.setattr(modelos, "_disparar", modelos._baixar_ate_o_fim)
    modelos.esquecer_tudo()

    return modelos


def _catalogo(modelos, nomes):
    modelos.ollama.modelos_no_disco = lambda: list(nomes)


# ---------------------------------------------------------------------------
# O caminho feliz
# ---------------------------------------------------------------------------
def test_um_modelo_que_ja_existe_passa_direto(baixando, monkeypatch):
    _catalogo(baixando, ["qwen2.5:7b-instruct"])
    baixou = []
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: baixou.append(n))

    baixando.conferir("qwen2.5:7b-instruct")

    assert baixou == []


def test_o_catalogo_so_e_consultado_uma_vez_por_modelo(baixando):
    """Pesos nao se desbaixam. Um `/api/tags` por pedido seria uma viagem a
    mais em todo o lote para saber o que ja sabemos."""
    idas = []
    baixando.ollama.modelos_no_disco = lambda: idas.append(1) or ["m:latest"]

    for _ in range(5):
        baixando.conferir("m")

    assert len(idas) == 1


def test_o_modelo_ausente_dispara_o_download_e_levanta(baixando, monkeypatch):
    catalogo = ["outro:latest"]
    baixando.ollama.modelos_no_disco = lambda: list(catalogo)
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: catalogo.append("novo:latest"))

    with pytest.raises(baixando.Baixando):
        baixando.conferir("novo")

    # Terminado o download (sincrono aqui), o proximo pedido passa.
    baixando.conferir("novo")


# ---------------------------------------------------------------------------
# Defesa 1: a tag implicita
# ---------------------------------------------------------------------------
def test_um_pedido_sem_tag_casa_com_o_latest_do_catalogo(baixando, monkeypatch):
    """O `/api/tags` lista `qwen2.5:latest`; o cliente pede `qwen2.5`.
    Comparando cru isso nunca casaria, e o worker baixaria o MESMO modelo a
    cada pedido, para sempre."""
    _catalogo(baixando, ["qwen2.5:latest"])
    baixou = []
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: baixou.append(n))

    baixando.conferir("qwen2.5")

    assert baixou == []


def test_um_modelo_com_barra_e_porta_nao_ganha_tag_errada(baixando, monkeypatch):
    """`registro:5000/qwen` tem `:` no HOST, nao na tag. Normalizar olhando a
    string inteira produziria `registro:5000/qwen` sem `:latest` e o modelo
    nunca casaria."""
    _catalogo(baixando, ["registro:5000/qwen:latest"])
    baixou = []
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: baixou.append(n))

    baixando.conferir("registro:5000/qwen")

    assert baixou == []


# ---------------------------------------------------------------------------
# Defesa 2: conferir DEPOIS do download
# ---------------------------------------------------------------------------
def test_um_download_que_nao_produz_o_modelo_vira_falha(baixando, monkeypatch):
    """Se o Ollama termina sem erro e o modelo nao aparece no catalogo, chamar
    isso de sucesso faria o proximo pedido disparar outro download, e o
    seguinte tambem."""
    _catalogo(baixando, [])
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: None)

    with pytest.raises(baixando.Baixando):
        baixando.conferir("fantasma")

    with pytest.raises(baixando.NaoDisponivel, match="nao aparece no catalogo"):
        baixando.conferir("fantasma")


# ---------------------------------------------------------------------------
# Defesa 3: memoria das falhas
# ---------------------------------------------------------------------------
def test_um_download_que_falha_nao_e_tentado_de_novo(baixando, monkeypatch):
    _catalogo(baixando, [])
    tentativas = []

    def falhar(nome, progresso=None):
        tentativas.append(nome)
        raise RuntimeError("nao existe no registro")

    monkeypatch.setattr(baixando.ollama, "baixar", falhar)

    with pytest.raises(baixando.Baixando):
        baixando.conferir("inexistente")
    for _ in range(3):
        with pytest.raises(baixando.NaoDisponivel):
            baixando.conferir("inexistente")

    assert len(tentativas) == 1


def test_a_falha_e_esquecida_depois_do_prazo(baixando, monkeypatch):
    """Lembrar para sempre exigiria reiniciar o servico depois de uma queda de
    rede. O prazo e o que deixa a maquina se curar sozinha."""
    _catalogo(baixando, [])
    monkeypatch.setattr(
        baixando.ollama, "baixar", lambda n, p=None: (_ for _ in ()).throw(RuntimeError("rede"))
    )
    monkeypatch.setattr(baixando, "FALHA_DE_DOWNLOAD_LEMBRADA", 0)

    with pytest.raises(baixando.Baixando):
        baixando.conferir("instavel")

    # Prazo zero: a proxima ida ja tenta de novo, em vez de devolver 404.
    with pytest.raises(baixando.Baixando):
        baixando.conferir("instavel")


# ---------------------------------------------------------------------------
# Concorrencia e desligamento
# ---------------------------------------------------------------------------
def test_dois_pedidos_do_mesmo_modelo_dao_um_download_so(baixando, monkeypatch):
    """Dois pulls do mesmo modelo nao quebram o Ollama, mas gastam banda em
    dobro e poluem o `/health/`."""
    _catalogo(baixando, [])
    chamadas = []
    monkeypatch.setattr(baixando, "_disparar", lambda alvo: chamadas.append(alvo))

    for _ in range(4):
        with pytest.raises(baixando.Baixando):
            baixando.conferir("devagar")

    assert chamadas == ["devagar:latest"]


def test_desligado_nao_baixa_e_deixa_o_ollama_responder(baixando, monkeypatch):
    """Comportamento de antes: o Ollama devolve 404 e a mensagem dele chega ao
    cliente intacta. E o que se quer num disco apertado."""
    monkeypatch.setattr(baixando, "BAIXAR_MODELO_AUTOMATICO", False)
    _catalogo(baixando, [])
    baixou = []
    monkeypatch.setattr(baixando.ollama, "baixar", lambda n, p=None: baixou.append(n))

    baixando.conferir("ausente")

    assert baixou == []


def test_sem_model_no_corpo_quem_decide_e_o_ollama(baixando):
    """A mensagem de erro dele e melhor que qualquer palpite daqui."""
    baixando.conferir(None)
    baixando.conferir("")


# ---------------------------------------------------------------------------
# O Retry-After sai do progresso
# ---------------------------------------------------------------------------
def test_o_retry_after_encolhe_conforme_o_download_avanca(baixando, monkeypatch):
    """Um valor fixo diria "volte em 60s" com 5% baixados, e seria um 503
    garantido."""
    import time

    item = baixando._Download(nome="m", comeco=time.monotonic() - 100)

    item.porcento = 0.5
    assert item.falta_estimado == 60

    item.porcento = 50.0
    assert item.falta_estimado == 100

    item.porcento = 99.0
    assert item.falta_estimado == 15


def test_o_retry_after_do_download_tem_teto(baixando):
    """Uma estimativa absurda poria a tarefa do cliente dormindo por horas."""
    import time

    item = baixando._Download(nome="m", comeco=time.monotonic() - 10)
    item.porcento = 0.6

    assert item.falta_estimado <= 600


# ---------------------------------------------------------------------------
# Pela rota, que e como o cliente ve
# ---------------------------------------------------------------------------
def test_o_pedido_de_modelo_ausente_vira_503_baixando(worker, cabecalhos, monkeypatch):
    import modelos

    monkeypatch.setattr(modelos, "BAIXAR_MODELO_AUTOMATICO", True)
    monkeypatch.setattr(modelos, "_disparar", lambda alvo: None)
    monkeypatch.setattr(modelos.ollama, "modelos_no_disco", list)
    modelos.esquecer_tudo()

    resposta = TestClient(worker.app).post(
        "/v1/chat/completions",
        json={"model": "novo:8b", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "baixando_modelo"
    assert int(resposta.headers["Retry-After"]) > 0


def test_um_download_falhado_vira_404_e_nao_503(worker, cabecalhos, monkeypatch):
    """A diferenca que decide tudo do lado do cliente: 503 ele reagenda, 404
    ele desiste. Um modelo que nao existe no registro nunca vai existir, e um
    503 aqui o poria reagendando para sempre."""
    import modelos

    monkeypatch.setattr(modelos, "BAIXAR_MODELO_AUTOMATICO", True)
    monkeypatch.setattr(modelos, "_disparar", modelos._baixar_ate_o_fim)
    monkeypatch.setattr(modelos.ollama, "modelos_no_disco", list)
    monkeypatch.setattr(
        modelos.ollama, "baixar", lambda n, p=None: (_ for _ in ()).throw(RuntimeError("404"))
    )
    modelos.esquecer_tudo()

    cliente_http = TestClient(worker.app)
    corpo = {"model": "nao-existe", "messages": [{"role": "user", "content": "oi"}]}

    primeira = cliente_http.post("/v1/chat/completions", json=corpo, headers=cabecalhos)
    segunda = cliente_http.post("/v1/chat/completions", json=corpo, headers=cabecalhos)

    assert primeira.status_code == 503
    assert segunda.status_code == 404
    assert "nao-existe" in segunda.json()["error"]["message"]


def test_o_download_nao_toma_o_lock_da_gpu(worker, cabecalhos, monkeypatch):
    """A razao de a conferencia vir ANTES de `ARBITRO.usar()`. Segurando o
    lock, a imagem e a conversao ficariam em 503 por dezenas de minutos por
    causa de um modelo de texto que nem carregou ainda."""
    import modelos
    from arbitro import ARBITRO

    monkeypatch.setattr(modelos, "BAIXAR_MODELO_AUTOMATICO", True)
    monkeypatch.setattr(modelos, "_disparar", lambda alvo: None)
    monkeypatch.setattr(modelos.ollama, "modelos_no_disco", list)
    modelos.esquecer_tudo()

    TestClient(worker.app).post(
        "/v1/chat/completions",
        json={"model": "gigante:70b", "messages": [{"role": "user", "content": "oi"}]},
        headers=cabecalhos,
    )

    assert ARBITRO.ocupacao is None


def test_o_health_mostra_o_download_em_curso(worker, monkeypatch):
    """Sem isto, um lote inteiro levando `baixando_modelo` nao teria como ser
    distinguido, de fora, de um worker travado."""
    import modelos

    monkeypatch.setattr(modelos, "BAIXAR_MODELO_AUTOMATICO", True)
    monkeypatch.setattr(modelos, "_disparar", lambda alvo: None)
    monkeypatch.setattr(modelos.ollama, "modelos_no_disco", list)
    modelos.esquecer_tudo()

    with pytest.raises(modelos.Baixando):
        modelos.conferir("llama3.1:8b")

    corpo = TestClient(worker.app).get("/health/").json()

    assert corpo["ollama"]["baixando"][0]["modelo"] == "llama3.1:8b"
    # E NAO aparece em `ocupada`: um download nao usa a placa.
    assert corpo["ocupada"] is False
