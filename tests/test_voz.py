"""A rota de voz (/v1/audio/speech), com o Kokoro substituido por um script de
mentira: a voz escolhida, o audio devolvido, os erros."""

from __future__ import annotations

import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

FALSO = """
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--voz"); p.add_argument("--saida"); p.add_argument("--velocidade")
a = p.parse_args()
texto = sys.stdin.read()
if "quebre" in texto:
    print("o modelo nao carregou", file=sys.stderr); sys.exit(3)
open(a.saida, "wb").write(f"ID3 {a.voz} {a.velocidade} {texto}".encode())
print("FORMATO mp3")
"""


@pytest.fixture
def cliente(worker, tmp_path, monkeypatch):
    import voz

    script = tmp_path / "falso.py"
    script.write_text(FALSO)
    monkeypatch.setattr(voz, "SCRIPT", script)
    monkeypatch.setattr(voz, "VOZ_PYTHON", sys.executable)
    app = FastAPI()
    app.include_router(voz.router)
    return TestClient(app)


def test_fala_com_a_voz_do_kokoro_ou_a_padrao(cliente, cabecalhos):
    resposta = cliente.post(
        "/v1/audio/speech",
        headers=cabecalhos,
        json={"model": "kokoro", "voice": "pm_alex", "input": "Ola!", "response_format": "mp3"},
    )
    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == "audio/mpeg"
    assert resposta.content == b"ID3 pm_alex 1.0 Ola!"
    da_openai = cliente.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "alloy", "input": "Oi"}
    )
    assert da_openai.content.startswith(b"ID3 pf_dora")  # vira a padrao


def test_erros(cliente, cabecalhos):
    vazio = cliente.post("/v1/audio/speech", headers=cabecalhos, json={"input": "  "})
    assert vazio.status_code == 422
    quebrou = cliente.post("/v1/audio/speech", headers=cabecalhos, json={"input": "quebre"})
    assert quebrou.status_code == 500
    assert quebrou.json()["error"]["code"] == "falha_na_voz"
    assert "nao carregou" in quebrou.json()["error"]["message"]
    assert cliente.post("/v1/audio/speech", json={"input": "x"}).status_code == 401


def test_conferir_configuracao(worker, monkeypatch, tmp_path):
    import voz

    monkeypatch.setattr(voz, "VOZ_PYTHON", str(tmp_path / "nao_existe"))
    with pytest.raises(RuntimeError, match="instalar_voz"):
        voz.conferir_configuracao()
