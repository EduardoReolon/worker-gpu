"""A medicao de memoria do proprio processo.

Existe por um defeito de diagnostico que custou dias. Nesta maquina, com 32 GB
e um modelo de texto de 30B ao lado, quem foi para o swap foi o WORKER — e nada
no `/health/` dizia isso, entao a suspeita caiu no Ollama.

O worker retem RAM por desenho: `enable_model_cpu_offload()` mantem os pesos do
modelo de imagem na RAM para o pico de VRAM caber ao lado do Ollama. Sao 5 a 6
GB de memoria anonima parada, que e o primeiro alvo do kernel quando ele precisa
de paginas — ao contrario dos pesos do Ollama, que sao arquivo mapeado e podem
ser descartados sem escrever nada.
"""

from __future__ import annotations

import logging

import pytest

import recursos

ESTADO_FALSO = """Name:\tpython3
VmPeak:\t 8123456 kB
VmSize:\t 7654321 kB
VmRSS:\t 6291456 kB
RssAnon:\t 6000000 kB
VmSwap:\t 1048576 kB
Threads:\t8
"""


@pytest.fixture(autouse=True)
def sem_aviso_vazado():
    """O aviso e uma vez por processo, e o estado vaza entre testes."""
    recursos.esquecer_o_aviso()
    yield
    recursos.esquecer_o_aviso()


@pytest.fixture
def estado(tmp_path, monkeypatch):
    def montar(conteudo: str):
        arquivo = tmp_path / "status"
        arquivo.write_text(conteudo, encoding="utf-8")
        monkeypatch.setattr(recursos, "_ARQUIVO_DE_ESTADO", str(arquivo))

    return montar


def test_le_rss_e_swap_em_megabytes(estado):
    estado(ESTADO_FALSO)

    assert recursos.memoria() == {"rss_mb": 6144.0, "swap_mb": 1024.0}


def test_o_swap_acima_de_zero_avisa_uma_vez_so(estado, caplog):
    """Uma vez por subida, e nao por leitura: o systemd consulta o `/health/`
    em laco, e um aviso por consulta enterraria tudo o que importa."""
    estado(ESTADO_FALSO)

    with caplog.at_level(logging.WARNING, logger="worker-gpu.recursos"):
        for _ in range(5):
            recursos.memoria()

    assert caplog.text.count("no swap") == 1
    assert "MemorySwapMax=0" in caplog.text


def test_sem_swap_nao_avisa(estado, caplog):
    estado(ESTADO_FALSO.replace("VmSwap:\t 1048576 kB", "VmSwap:\t       0 kB"))

    with caplog.at_level(logging.WARNING, logger="worker-gpu.recursos"):
        recursos.memoria()

    assert caplog.text == ""
    assert recursos.memoria()["swap_mb"] == 0.0


def test_um_proc_sem_vmswap_nao_derruba_a_leitura(estado):
    """Kernel sem swap compilado nao publica `VmSwap`. Um campo a menos nao
    pode custar o outro."""
    estado("Name:\tpython3\nVmRSS:\t 1024 kB\n")

    assert recursos.memoria() == {"rss_mb": 1.0}


def test_sem_proc_devolve_vazio_em_vez_de_levantar(monkeypatch):
    """A regra do `/health/`: ele nunca devolve 500. Um campo que nao da para
    saber e um campo ausente, e nao uma excecao."""
    monkeypatch.setattr(recursos, "_ARQUIVO_DE_ESTADO", "/nao/existe/status")

    assert recursos.memoria() == {}


def test_um_valor_ilegivel_e_ignorado_sem_levantar(estado):
    estado("VmRSS:\tmuita kB\nVmSwap:\t 2048 kB\n")

    assert recursos.memoria() == {"swap_mb": 2.0}


def test_o_health_publica_a_memoria_do_processo(worker):
    """E o campo que faltava. Sem ele, um worker no swap e indistinguivel de um
    worker lento, e quem investiga vai olhar a GPU."""
    from fastapi.testclient import TestClient

    corpo = TestClient(worker.app).get("/health/").json()

    assert "rss_mb" in corpo["memoria"]


def test_o_health_sobrevive_a_uma_leitura_que_estoura(worker, monkeypatch):
    import recursos as modulo

    monkeypatch.setattr(
        modulo, "memoria", lambda: (_ for _ in ()).throw(RuntimeError("proc quebrado"))
    )

    from fastapi.testclient import TestClient

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    assert "erro" in resposta.json()["memoria"]
