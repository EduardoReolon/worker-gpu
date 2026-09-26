"""A rota de transcricao, pelo HTTP, com o Whisper substituido.

O que se exercita e o contrato que o PubliBot espera: a forma do
`verbose_json`, os codigos de erro, a arbitragem, e o laco sobre os segmentos
(com um modelo de mentira) — que e onde o teto macio e conferido.
"""

from __future__ import annotations

import threading
import types

import pytest
from fastapi.testclient import TestClient

AUDIO = b"ID3 fingindo ser um mp3"

_RESULTADO = {
    "task": "transcribe",
    "language": "pt",
    "duration": 9.8,
    "text": "Ola. Hoje vamos falar de BDI.",
    "segments": [
        {"id": 0, "start": 0.0, "end": 4.2, "text": "Ola."},
        {"id": 1, "start": 4.2, "end": 9.8, "text": "Hoje vamos falar de BDI."},
    ],
}


@pytest.fixture
def transcricao_mod(worker):
    import transcricao

    return transcricao


@pytest.fixture
def cliente(worker, transcricao_mod, monkeypatch):
    import ollama

    vistos = []

    def transcrever(dispositivo, caminho, idioma, contexto):
        with open(caminho, "rb") as arquivo:
            vistos.append({"conteudo": arquivo.read(), "idioma": idioma, "contexto": contexto})
        return dict(_RESULTADO)

    monkeypatch.setattr(transcricao_mod, "transcrever", transcrever)
    monkeypatch.setattr(transcricao_mod, "resolver_dispositivo", lambda: "cuda")
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: None)
    cliente = TestClient(worker.app)
    cliente.vistos = vistos
    return cliente


def _pedir(cliente, cabecalhos, **campos):
    dados = {"model": "whisper", "language": "pt", "response_format": "verbose_json", **campos}
    return cliente.post(
        "/v1/audio/transcriptions",
        files={"file": ("aula.mp3", AUDIO, "audio/mpeg")},
        data=dados,
        headers=cabecalhos,
    )


# ---------------------------------------------------------------------------
# O caminho feliz
# ---------------------------------------------------------------------------
def test_exige_credencial(cliente):
    resposta = cliente.post(
        "/v1/audio/transcriptions", files={"file": ("a.mp3", AUDIO, "audio/mpeg")}
    )

    assert resposta.status_code == 401


def test_verbose_json_traz_texto_idioma_duracao_e_segmentos(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["text"] == _RESULTADO["text"]
    assert corpo["language"] == "pt"
    assert corpo["duration"] == 9.8
    assert corpo["segments"][1] == {
        "id": 1,
        "start": 4.2,
        "end": 9.8,
        "text": "Hoje vamos falar de BDI.",
    }


def test_o_arquivo_e_o_idioma_chegam_ao_whisper(cliente, cabecalhos):
    _pedir(cliente, cabecalhos, language="PT", prompt="BDI, SINAPI")

    assert cliente.vistos == [{"conteudo": AUDIO, "idioma": "pt", "contexto": "BDI, SINAPI"}]


def test_sem_idioma_o_whisper_detecta(cliente, cabecalhos):
    _pedir(cliente, cabecalhos, language="")

    assert cliente.vistos[0]["idioma"] is None


def test_json_traz_so_o_texto(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, response_format="json")

    assert resposta.json() == {"text": _RESULTADO["text"]}


def test_text_devolve_texto_puro(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, response_format="text")

    assert resposta.headers["content-type"].startswith("text/plain")
    assert resposta.text == _RESULTADO["text"]


def test_descarrega_o_ollama_antes_de_transcrever(cliente, cabecalhos, monkeypatch):
    """O Whisper disputa a VRAM com o modelo de texto, como a difusao."""
    import ollama

    chamadas = []
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: chamadas.append(True))

    _pedir(cliente, cabecalhos)

    assert chamadas == [True]


# ---------------------------------------------------------------------------
# Pedido errado: 4xx com error.message
# ---------------------------------------------------------------------------
def test_formato_de_resposta_desconhecido_e_422(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, response_format="srt")

    assert resposta.status_code == 422
    assert "verbose_json" in resposta.json()["error"]["message"]


def test_arquivo_vazio_e_422(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.mp3", b"", "audio/mpeg")},
        headers=cabecalhos,
    )

    assert resposta.status_code == 422
    assert resposta.json()["error"]["code"] == "arquivo_invalido"


def test_arquivo_grande_demais_e_413(cliente, cabecalhos, transcricao_mod, monkeypatch):
    monkeypatch.setattr(transcricao_mod, "MAX_AUDIO_BYTES", 5)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 413
    assert resposta.json()["error"]["code"] == "arquivo_grande"


def test_arquivo_que_nao_e_audio_e_422_com_a_mensagem(
    cliente, cabecalhos, transcricao_mod, monkeypatch
):
    def recusa(*argumentos):
        raise transcricao_mod.AudioInvalido("Invalid data found when processing input")

    monkeypatch.setattr(transcricao_mod, "transcrever", recusa)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 422
    assert "Invalid data" in resposta.json()["error"]["message"]


# ---------------------------------------------------------------------------
# O contrato de 503 e 500
# ---------------------------------------------------------------------------
def test_placa_ocupada_e_503_com_retry_after(cliente, cabecalhos):
    import arbitro

    with arbitro.ARBITRO.usar("imagem"):
        resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "gpu_ocupada"
    assert "Retry-After" in resposta.headers


def test_teto_macio_e_503_timeout_sem_retry_after(
    cliente, cabecalhos, transcricao_mod, monkeypatch
):
    def demora(*argumentos):
        raise transcricao_mod.TempoEsgotado("passou de 1200s")

    monkeypatch.setattr(transcricao_mod, "transcrever", demora)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "timeout"
    assert "Retry-After" not in resposta.headers


def test_falta_de_vram_e_503_sem_vram(cliente, cabecalhos, transcricao_mod, monkeypatch):
    def sem_vram(*argumentos):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr(transcricao_mod, "transcrever", sem_vram)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "sem_vram"


def test_travado_e_500_e_o_processo_se_encerra(cliente, cabecalhos, transcricao_mod, monkeypatch):
    import arbitro

    solta = threading.Event()
    mortes = []
    monkeypatch.setattr(transcricao_mod, "TRANSCRICAO_TEMPO_TRAVADO", 0.2)
    monkeypatch.setattr(transcricao_mod, "transcrever", lambda *argumentos: solta.wait(5))
    monkeypatch.setattr(transcricao_mod.prazo, "morrer_em_seguida", lambda: mortes.append(True))

    try:
        resposta = _pedir(cliente, cabecalhos)
        assert arbitro.ARBITRO.ocupacao is None
    finally:
        solta.set()

    assert resposta.status_code == 500
    assert resposta.json()["error"]["code"] == "worker_travado"
    assert mortes == [True]


def test_outra_falha_e_500_com_codigo(cliente, cabecalhos, transcricao_mod, monkeypatch):
    def quebra(*argumentos):
        raise RuntimeError("algo inesperado")

    monkeypatch.setattr(transcricao_mod, "transcrever", quebra)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 500
    assert resposta.json()["error"]["code"] == "falha_na_transcricao"


# ---------------------------------------------------------------------------
# O laco sobre os segmentos, com um Whisper de mentira
# ---------------------------------------------------------------------------
def _segmento(inicio, fim, texto):
    return types.SimpleNamespace(start=inicio, end=fim, text=texto)


class _WhisperFalso:
    def __init__(self, segmentos=(), erro=None):
        self.segmentos = segmentos
        self.erro = erro
        self.argumentos = {}

    def transcribe(self, caminho, **argumentos):
        if self.erro:
            raise self.erro
        self.argumentos = argumentos
        info = types.SimpleNamespace(language="pt", duration=9.8321)
        return iter(self.segmentos), info


def test_os_segmentos_viram_texto_e_lista(transcricao_mod, monkeypatch):
    falso = _WhisperFalso([_segmento(0.0, 4.2049, " Ola."), _segmento(4.2049, 9.8321, " Hoje.")])
    monkeypatch.setattr(transcricao_mod, "obter_modelo", lambda dispositivo: falso)

    resultado = transcricao_mod.transcrever("cuda", "audio.mp3", "pt", None)

    assert resultado["text"] == "Ola. Hoje."
    assert resultado["duration"] == 9.83
    assert resultado["segments"][1] == {"id": 1, "start": 4.2, "end": 9.83, "text": "Hoje."}
    assert falso.argumentos["vad_filter"] is True
    assert falso.argumentos["language"] == "pt"


def test_o_teto_macio_e_conferido_entre_segmentos(transcricao_mod, monkeypatch):
    falso = _WhisperFalso([_segmento(0.0, 60.0, "a"), _segmento(60.0, 120.0, "b")])
    monkeypatch.setattr(transcricao_mod, "obter_modelo", lambda dispositivo: falso)
    monkeypatch.setattr(transcricao_mod, "TRANSCRICAO_TEMPO_MAXIMO", 1e-9)

    with pytest.raises(transcricao_mod.TempoEsgotado):
        transcricao_mod.transcrever("cuda", "audio.mp3", None, None)


def test_erro_do_pyav_vira_audio_invalido(transcricao_mod, monkeypatch):
    """O PyAV recusa o que nao e audio com uma excecao do modulo `av`. E o
    pedido que esta errado, e nao o worker: 422, e nao 500."""
    InvalidDataError = type("InvalidDataError", (Exception,), {"__module__": "av.error"})
    falso = _WhisperFalso(erro=InvalidDataError("Invalid data found when processing input"))
    monkeypatch.setattr(transcricao_mod, "obter_modelo", lambda dispositivo: falso)

    with pytest.raises(transcricao_mod.AudioInvalido):
        transcricao_mod.transcrever("cuda", "audio.mp3", None, None)


# ---------------------------------------------------------------------------
# Subida e /health/
# ---------------------------------------------------------------------------
def test_o_prazo_duro_precisa_ser_maior_que_o_macio(transcricao_mod, monkeypatch):
    monkeypatch.setattr(transcricao_mod, "TRANSCRICAO_TEMPO_MAXIMO", 1500)
    monkeypatch.setattr(transcricao_mod, "TRANSCRICAO_TEMPO_TRAVADO", 1500)

    with pytest.raises(RuntimeError, match="precisa ser maior"):
        transcricao_mod.conferir_configuracao()


def test_sem_faster_whisper_recusa_subir(transcricao_mod, monkeypatch):
    monkeypatch.setattr(transcricao_mod.importlib.util, "find_spec", lambda nome: None)

    with pytest.raises(RuntimeError, match="faster-whisper"):
        transcricao_mod.conferir_configuracao()


def test_o_padrao_fica_abaixo_do_timeout_do_publibot(transcricao_mod):
    """O PubliBot espera 1800 s. O 500 legivel tem que chegar antes."""
    assert (
        transcricao_mod.TRANSCRICAO_TEMPO_MAXIMO < transcricao_mod.TRANSCRICAO_TEMPO_TRAVADO < 1800
    )


def test_o_health_publica_o_bloco_e_a_rota(cliente):
    corpo = cliente.get("/health/").json()

    assert corpo["rotas"]["transcricao"] is True
    assert corpo["transcricao"]["modelo"] == "large-v3"
    assert corpo["transcricao"]["carregado"] is False
    assert corpo["transcricao"]["tempo_travado"] == 1500


def test_o_catalogo_lista_o_whisper(cliente, cabecalhos, monkeypatch):
    import ollama

    monkeypatch.setattr(ollama, "modelos_no_disco", lambda: [])

    nomes = [m["id"] for m in cliente.get("/v1/models", headers=cabecalhos).json()["data"]]

    assert "large-v3" in nomes
