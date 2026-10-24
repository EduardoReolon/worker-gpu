"""A rota de legenda do YouTube, sem ir ao YouTube.

As excecoes sao as da biblioteca de verdade: e o mapeamento delas para os
codigos do contrato que o PubliBot le.
"""

from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

VIDEO = "dQw4w9WgXcQ"


@pytest.fixture
def youtube_mod(worker):
    import youtube

    return youtube


@pytest.fixture
def cliente(worker):
    return TestClient(worker.app)


def _legenda_falsa(codigo="pt", gerada=True):
    trechos = [
        types.SimpleNamespace(start=0.0, duration=4.2, text="Ola."),
        types.SimpleNamespace(start=4.2, duration=5.6, text="Hoje vamos falar de BDI."),
    ]
    return types.SimpleNamespace(language_code=codigo, is_generated=gerada), trechos


def _pedir(cliente, cabecalhos, **corpo):
    return cliente.post(
        "/v1/youtube/legenda",
        json={"video_id": VIDEO, "idiomas": ["pt", "pt-BR", "en"], **corpo},
        headers=cabecalhos,
    )


def test_exige_credencial(cliente):
    assert cliente.post("/v1/youtube/legenda", json={"video_id": VIDEO}).status_code == 401


def test_devolve_idioma_origem_e_segmentos(cliente, cabecalhos, youtube_mod, monkeypatch):
    monkeypatch.setattr(youtube_mod, "buscar_legenda", lambda video, idiomas: _legenda_falsa())

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["idioma"] == "pt"
    assert corpo["gerada_automaticamente"] is True
    assert corpo["segmentos"][1] == {
        "start": 4.2,
        "duration": 5.6,
        "text": "Hoje vamos falar de BDI.",
    }


def test_nao_entra_no_lock_da_placa(cliente, cabecalhos, youtube_mod, monkeypatch):
    """Legenda e rede, nao GPU: com a placa ocupada, ela sai do mesmo jeito."""
    import arbitro

    monkeypatch.setattr(youtube_mod, "buscar_legenda", lambda video, idiomas: _legenda_falsa())

    with arbitro.ARBITRO.usar("imagem"):
        resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 200


def test_video_id_invalido_e_422(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, video_id="https://youtu.be/x")

    assert resposta.status_code == 422
    assert resposta.json()["error"]["code"] == "video_id_invalido"


@pytest.mark.parametrize(
    ("excecao", "status", "codigo"),
    [
        (lambda m: m.NoTranscriptFound(VIDEO, ["pt"], "nenhuma"), 404, "sem_legenda"),
        (lambda m: m.TranscriptsDisabled(VIDEO), 404, "sem_legenda"),
        (lambda m: m.VideoUnavailable(VIDEO), 404, "video_indisponivel"),
        (lambda m: m.RequestBlocked(VIDEO), 503, "bloqueado"),
        (lambda m: m.IpBlocked(VIDEO), 503, "bloqueado"),
        (lambda m: ConnectionError("rede caiu"), 503, "youtube_indisponivel"),
    ],
)
def test_as_falhas_viram_os_codigos_do_contrato(
    cliente, cabecalhos, youtube_mod, monkeypatch, excecao, status, codigo
):
    import youtube_transcript_api

    erro = excecao(youtube_transcript_api)

    def falha(video, idiomas):
        raise erro

    monkeypatch.setattr(youtube_mod, "buscar_legenda", falha)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == status
    assert resposta.json()["error"]["code"] == codigo
    if status == 503:
        assert "Retry-After" in resposta.headers


# ---------------------------------------------------------------------------
# A escolha da legenda
# ---------------------------------------------------------------------------
class _ListaFalsa:
    """Legendas disponiveis: {idioma: {"manual": bool, "automatica": bool}}."""

    def __init__(self, disponiveis):
        self.disponiveis = disponiveis

    def _achar(self, idiomas, tipo):
        from youtube_transcript_api import NoTranscriptFound

        for idioma in idiomas:
            if self.disponiveis.get(idioma, {}).get(tipo):
                legenda = types.SimpleNamespace(
                    language_code=idioma, is_generated=tipo == "automatica"
                )
                legenda.fetch = lambda: []
                return legenda
        raise NoTranscriptFound(VIDEO, idiomas, "lista")

    def find_manually_created_transcript(self, idiomas):
        return self._achar(idiomas, "manual")

    def find_generated_transcript(self, idiomas):
        return self._achar(idiomas, "automatica")


def _com_lista(monkeypatch, disponiveis):
    import youtube_transcript_api

    class _Api:
        def __init__(self, http_client=None):
            pass

        def list(self, video_id):
            return _ListaFalsa(disponiveis)

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", _Api)


def test_manual_antes_de_automatica_no_mesmo_idioma(youtube_mod, monkeypatch):
    _com_lista(monkeypatch, {"pt": {"manual": True, "automatica": True}})

    legenda, _ = youtube_mod.buscar_legenda(VIDEO, ["pt"])

    assert legenda.is_generated is False


def test_o_idioma_preferido_vence_mesmo_so_com_automatica(youtube_mod, monkeypatch):
    """A ordem de `idiomas` e do cliente: `pt` automatica antes de `en` manual."""
    _com_lista(monkeypatch, {"pt": {"automatica": True}, "en": {"manual": True}})

    legenda, _ = youtube_mod.buscar_legenda(VIDEO, ["pt", "en"])

    assert (legenda.language_code, legenda.is_generated) == ("pt", True)


def test_nenhum_idioma_disponivel_e_sem_legenda(youtube_mod, monkeypatch):
    from youtube_transcript_api import NoTranscriptFound

    _com_lista(monkeypatch, {"de": {"manual": True}})

    with pytest.raises(NoTranscriptFound):
        youtube_mod.buscar_legenda(VIDEO, ["pt", "en"])


def test_toda_ida_ao_youtube_tem_teto_de_tempo(youtube_mod, monkeypatch):
    """A biblioteca nao passa `timeout` ao requests: sem isto, uma conexao
    muda prenderia a requisicao para sempre."""
    import requests

    vistos = {}
    monkeypatch.setattr(requests.Session, "request", lambda self, *a, **k: vistos.update(k))

    youtube_mod._sessao().request("GET", "https://www.youtube.com/watch")

    assert vistos["timeout"] == youtube_mod.YOUTUBE_TIMEOUT


def test_o_health_lista_a_rota(cliente):
    assert cliente.get("/health/").json()["rotas"]["youtube"] is True
