"""A rota de vetorizacao, pelo HTTP, com o modelo substituido.

O que se exercita e o contrato que o PubliBot espera: o nome do modelo
conferido, o 503 `modelo_carregando` com download unico em segundo plano, a
politica de descarregar o Ollama, e o mean pooling com mascara.
"""

from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

MODELO = "intfloat/multilingual-e5-large"


@pytest.fixture
def vetorizacao_mod(worker):
    import vetorizacao

    return vetorizacao


@pytest.fixture
def cliente(worker, vetorizacao_mod, monkeypatch):
    import ollama

    descargas = []
    monkeypatch.setattr(vetorizacao_mod, "modelo_esta_no_disco", lambda: True)
    monkeypatch.setattr(vetorizacao_mod, "resolver_dispositivo", lambda: "cuda")
    monkeypatch.setattr(
        vetorizacao_mod,
        "vetorizar",
        lambda dispositivo, textos: ([[float(i)] * 1024 for i in range(len(textos))], 7),
    )
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: descargas.append(True))
    cliente = TestClient(worker.app)
    cliente.descargas = descargas
    return cliente


def _pedir(cliente, cabecalhos, **corpo):
    return cliente.post(
        "/v1/embeddings",
        json={"model": MODELO, "input": ["passage: a", "passage: b"], **corpo},
        headers=cabecalhos,
    )


# ---------------------------------------------------------------------------
# O caminho feliz
# ---------------------------------------------------------------------------
def test_exige_credencial(cliente):
    assert cliente.post("/v1/embeddings", json={"model": MODELO, "input": ["x"]}).status_code == 401


def test_devolve_um_vetor_de_1024_por_texto_na_ordem(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["object"] == "list"
    assert corpo["model"] == MODELO
    assert [item["index"] for item in corpo["data"]] == [0, 1]
    assert corpo["data"][1]["embedding"][:2] == [1.0, 1.0]
    assert len(corpo["data"][0]["embedding"]) == 1024
    assert corpo["usage"]["total_tokens"] == 7


def test_o_texto_chega_ao_modelo_sem_nada_acrescentado(
    cliente, cabecalhos, vetorizacao_mod, monkeypatch
):
    """O prefixo `passage: ` e do cliente. Acrescentar outro daria vetor
    diferente do servidor, no mesmo indice."""
    vistos = []
    monkeypatch.setattr(
        vetorizacao_mod,
        "vetorizar",
        lambda dispositivo, textos: (vistos.extend(textos) or [[0.0]] * len(textos), 1),
    )

    _pedir(cliente, cabecalhos, input=["passage: A curadoria garante."])

    assert vistos == ["passage: A curadoria garante."]


def test_input_como_texto_unico_e_aceito(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, input="passage: um so")

    assert len(resposta.json()["data"]) == 1


# ---------------------------------------------------------------------------
# Pedido errado
# ---------------------------------------------------------------------------
def test_outro_modelo_e_recusado_com_422(cliente, cabecalhos):
    """Vetor de outro modelo no mesmo indice estraga a busca sem aviso."""
    resposta = _pedir(cliente, cabecalhos, model="text-embedding-3-small")

    assert resposta.status_code == 422
    assert resposta.json()["error"]["code"] == "modelo_errado"
    assert MODELO in resposta.json()["error"]["message"]


def test_input_vazio_e_422(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, input=[])

    assert resposta.status_code == 422
    assert resposta.json()["error"]["message"] == "input vazio."


def test_textos_demais_e_422(cliente, cabecalhos, vetorizacao_mod, monkeypatch):
    monkeypatch.setattr(vetorizacao_mod, "VETORIZACAO_MAXIMO_TEXTOS", 1)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 422


# ---------------------------------------------------------------------------
# Modelo ausente: 503 modelo_carregando, e um download so
# ---------------------------------------------------------------------------
def test_modelo_ausente_e_503_modelo_carregando_e_baixa_uma_vez(
    cliente, cabecalhos, vetorizacao_mod, monkeypatch
):
    solta = threading.Event()
    downloads = []

    def baixar():
        downloads.append(True)
        solta.wait(5)
        return "/cache/e5"

    monkeypatch.setattr(vetorizacao_mod, "modelo_esta_no_disco", lambda: False)
    monkeypatch.setattr(vetorizacao_mod, "baixar", baixar)

    try:
        primeira = _pedir(cliente, cabecalhos)
        segunda = _pedir(cliente, cabecalhos)
    finally:
        solta.set()
        vetorizacao_mod._download_thread.join(5)

    for resposta in (primeira, segunda):
        assert resposta.status_code == 503
        assert resposta.json()["error"]["code"] == "modelo_carregando"
        assert resposta.headers["Retry-After"] == "60"
    assert downloads == [True]


def test_uma_falha_de_download_e_lembrada(cliente, cabecalhos, vetorizacao_mod, monkeypatch):
    """Sem memoria, cada retentativa dispararia um download novo de algo que
    nao vai vir."""

    def falha():
        raise OSError("sem rede")

    monkeypatch.setattr(vetorizacao_mod, "modelo_esta_no_disco", lambda: False)
    monkeypatch.setattr(vetorizacao_mod, "baixar", falha)

    _pedir(cliente, cabecalhos)
    vetorizacao_mod._download_thread.join(5)
    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert "sem rede" in resposta.json()["error"]["message"]
    assert int(resposta.headers["Retry-After"]) > 60


# ---------------------------------------------------------------------------
# O Ollama e a VRAM
# ---------------------------------------------------------------------------
def test_se_faltar_tenta_ao_lado_do_ollama_primeiro(cliente, cabecalhos):
    """Vetorizar leva segundos; recarregar um 30B leva dezenas deles."""
    _pedir(cliente, cabecalhos)

    assert cliente.descargas == []


def test_se_faltar_descarrega_o_ollama_e_repete_quando_nao_cabe(
    cliente, cabecalhos, vetorizacao_mod, monkeypatch
):
    tentativas = []

    def vetorizar(dispositivo, textos):
        tentativas.append(True)
        if len(tentativas) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return [[0.0] * 1024] * len(textos), 3

    monkeypatch.setattr(vetorizacao_mod, "vetorizar", vetorizar)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 200
    assert cliente.descargas == [True]
    assert len(tentativas) == 2


def test_sem_vram_mesmo_sem_o_ollama_e_503(cliente, cabecalhos, vetorizacao_mod, monkeypatch):
    def sem_vram(*argumentos):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(vetorizacao_mod, "vetorizar", sem_vram)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "sem_vram"


def test_sim_descarrega_o_ollama_antes(cliente, cabecalhos, vetorizacao_mod, monkeypatch):
    monkeypatch.setattr(vetorizacao_mod, "OLLAMA_DESCARREGAR_PARA_VETORIZACAO", "sim")

    _pedir(cliente, cabecalhos)

    assert cliente.descargas == [True]


def test_placa_ocupada_e_503(cliente, cabecalhos):
    import arbitro

    with arbitro.ARBITRO.usar("imagem"):
        resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "gpu_ocupada"


def test_travado_e_500_e_o_processo_se_encerra(cliente, cabecalhos, vetorizacao_mod, monkeypatch):
    solta = threading.Event()
    mortes = []
    monkeypatch.setattr(vetorizacao_mod, "VETORIZACAO_TEMPO_TRAVADO", 0.2)
    monkeypatch.setattr(vetorizacao_mod, "vetorizar", lambda *argumentos: solta.wait(5))
    monkeypatch.setattr(vetorizacao_mod.prazo, "morrer_em_seguida", lambda: mortes.append(True))

    try:
        resposta = _pedir(cliente, cabecalhos)
    finally:
        solta.set()

    assert resposta.status_code == 500
    assert resposta.json()["error"]["code"] == "worker_travado"
    assert mortes == [True]


# ---------------------------------------------------------------------------
# Subida e /health/
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("chave", "valor"),
    [
        ("VETORIZACAO_DEVICE", "gpu"),
        ("VETORIZACAO_DTYPE", "fp32"),
        ("OLLAMA_DESCARREGAR_PARA_VETORIZACAO", "talvez"),
    ],
)
def test_valores_invalidos_recusam_subir(vetorizacao_mod, monkeypatch, chave, valor):
    monkeypatch.setattr(vetorizacao_mod, chave, valor)

    with pytest.raises(RuntimeError, match=chave):
        vetorizacao_mod.conferir_configuracao()


def test_o_health_publica_o_bloco_e_a_rota(cliente):
    corpo = cliente.get("/health/").json()

    assert corpo["rotas"]["vetorizacao"] is True
    assert corpo["vetorizacao"]["modelo"] == MODELO
    assert corpo["vetorizacao"]["precisao"] == "float32"


# ---------------------------------------------------------------------------
# A conta: mean pooling com mascara, sem normalizar (o do fastembed 0.8)
# ---------------------------------------------------------------------------
def test_o_mean_pooling_ignora_o_preenchimento_e_nao_normaliza(vetorizacao_mod, monkeypatch):
    torch = pytest.importorskip("torch")

    class _Entrada(dict):
        def to(self, dispositivo):
            return self

    class _Tokenizador:
        model_max_length = 512

        def __call__(self, lote, **argumentos):
            assert argumentos["truncation"] is True and argumentos["max_length"] == 512
            # "curto" tem 1 token de verdade e 2 de preenchimento.
            return _Entrada(
                input_ids=torch.zeros(2, 3, dtype=torch.long),
                attention_mask=torch.tensor([[1, 1, 1], [1, 0, 0]]),
            )

    class _Modelo:
        def __call__(self, input_ids, attention_mask):
            estados = torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[9.0, 9.0], [100.0, 100.0], [100.0, 100.0]],
                ]
            )
            return type("Saida", (), {"last_hidden_state": estados})()

    monkeypatch.setattr(
        vetorizacao_mod, "obter_modelo", lambda dispositivo: (_Modelo(), _Tokenizador())
    )

    vetores, tokens = vetorizacao_mod.vetorizar("cpu", ["longo", "curto"])

    assert vetores == [[3.0, 4.0], [9.0, 9.0]]
    assert tokens == 4
