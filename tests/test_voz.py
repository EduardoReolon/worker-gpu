"""A rota de voz (/v1/audio/speech), com o Kokoro e o Qwen substituidos por
scripts de mentira: a voz escolhida, o audio devolvido, os erros, a placa."""

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
    com_prefixo = cliente.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "kokoro:pm_santa", "input": "Oi"}
    )
    assert com_prefixo.content.startswith(b"ID3 pm_santa")


@pytest.mark.parametrize("voz", ["JBFqnCBsd6RMkjVDRZzb", "pt-BR-Neural2-B", "qwen:carla"])
def test_voz_de_outro_servico_e_recusada(cliente, cabecalhos, voz):
    """Em vez de narrar com a padrao: o cliente com mais de um servico passa
    para o proximo, e quem so tem o worker fica sabendo."""
    resposta = cliente.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": voz, "input": "Oi"}
    )
    assert resposta.status_code == 422
    assert resposta.json()["error"]["code"] == "voz_desconhecida"


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


FALSO_QWEN = """
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("modo"); p.add_argument("--voz-dir"); p.add_argument("--modelo")
p.add_argument("--dispositivo"); p.add_argument("--idioma"); p.add_argument("--saida")
p.add_argument("--descricao"); p.add_argument("--gravacao"); p.add_argument("--texto")
a = p.parse_args()
pasta = pathlib.Path(a.voz_dir)
if a.modo == "falar":
    texto = sys.stdin.read()
    if "quebre" in texto:
        print("CUDA out of memory", file=sys.stderr); sys.exit(1)
    ref = (pasta / "referencia.txt").read_text().strip()
    open(a.saida, "wb").write(f"ID3 {pasta.name} {a.modelo} {ref} {texto}".encode())
    print("FORMATO mp3")
elif a.modo == "desenhar":
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "referencia.wav").write_bytes(b"RIFF")
    (pasta / "referencia.txt").write_text("Oi! Que bom.")
    (pasta / "descricao.txt").write_text(a.descricao)
else:
    if open(a.gravacao, "rb").read() == b"curta":
        print("a gravacao tem menos de 3 s", file=sys.stderr); sys.exit(2)
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "referencia.wav").write_bytes(b"RIFF")
    if a.texto:
        (pasta / "referencia.txt").write_text(a.texto)
print("PRONTO")
"""


@pytest.fixture
def qwen(cliente, tmp_path, monkeypatch):
    import recursos
    import voz

    script = tmp_path / "falso_qwen.py"
    script.write_text(FALSO_QWEN)
    soltou = []
    monkeypatch.setattr(voz, "VOZ_QWEN", True)
    monkeypatch.setattr(voz, "SCRIPT_QWEN", script)
    monkeypatch.setattr(voz, "VOZ_QWEN_PYTHON", sys.executable)
    monkeypatch.setattr(voz, "VOZ_VOZES_DIR", str(tmp_path / "vozes"))
    monkeypatch.setattr(recursos, "soltar_a_placa", soltou.append)
    cliente.soltou = soltou
    return cliente


def test_cria_voz_por_descricao_e_fala_nela(qwen, cabecalhos):
    criada = qwen.post(
        "/v1/audio/voices", headers=cabecalhos,
        data={"nome": "Carla", "descricao": "mulher de 30 anos, voz calorosa"},
    )  # fmt: skip
    assert criada.status_code == 201
    assert criada.json() == {
        "id": "qwen:carla", "motor": "qwen",
        "descricao": "mulher de 30 anos, voz calorosa", "transcrita": True,
    }  # fmt: skip
    assert qwen.soltou == ["criar a voz"]  # a placa foi tomada e esvaziada

    de_novo = qwen.post(
        "/v1/audio/voices", headers=cabecalhos, data={"nome": "carla", "descricao": "outra"}
    )
    assert de_novo.status_code == 409

    fala = qwen.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "qwen:carla", "input": "Ola!"}
    )
    assert fala.status_code == 200
    assert fala.content == b"ID3 carla Qwen/Qwen3-TTS-12Hz-1.7B-Base Oi! Que bom. Ola!"

    vozes = qwen.get("/v1/audio/voices", headers=cabecalhos).json()
    assert [v["id"] for v in vozes["data"]] == ["pf_dora", "pm_alex", "pm_santa", "qwen:carla"]


def test_cria_voz_de_gravacao(qwen, cabecalhos):
    criada = qwen.post(
        "/v1/audio/voices", headers=cabecalhos, data={"nome": "eu"},
        files={"gravacao": ("eu.wav", b"RIFF....", "audio/wav")},
    )  # fmt: skip
    assert criada.status_code == 201
    assert criada.json()["transcrita"] is False  # sem o texto: so a impressao da voz
    assert qwen.soltou == []  # importar nao usa a placa
    curta = qwen.post(
        "/v1/audio/voices", headers=cabecalhos, data={"nome": "curta"},
        files={"gravacao": ("c.wav", b"curta", "audio/wav")},
    )  # fmt: skip
    assert curta.status_code == 422
    assert "menos de 3 s" in curta.json()["error"]["message"]
    sem_nada = qwen.post("/v1/audio/voices", headers=cabecalhos, data={"nome": "x"})
    assert sem_nada.status_code == 422


def test_voz_do_qwen_espera_a_placa(qwen, cabecalhos):
    import arbitro

    qwen.post("/v1/audio/voices", headers=cabecalhos, data={"nome": "bia", "descricao": "jovem"})
    with arbitro.ARBITRO.usar("animacao"):
        resposta = qwen.post(
            "/v1/audio/speech", headers=cabecalhos, json={"voice": "qwen:bia", "input": "Oi"}
        )
    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "gpu_ocupada"
    assert "Retry-After" in resposta.headers

    quebrou = qwen.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "qwen:bia", "input": "quebre"}
    )
    assert quebrou.status_code == 500
    assert "out of memory" in quebrou.json()["error"]["message"]

    sumiu = qwen.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "qwen:ze", "input": "Oi"}
    )
    assert sumiu.status_code == 422
    assert "qwen:bia" in sumiu.json()["error"]["message"]  # diz as que existem


def test_voz_padrao_do_qwen(qwen, cabecalhos, monkeypatch):
    import voz

    qwen.post("/v1/audio/voices", headers=cabecalhos, data={"nome": "bia", "descricao": "jovem"})
    monkeypatch.setattr(voz, "VOZ_PADRAO", "qwen:bia")
    resposta = qwen.post("/v1/audio/speech", headers=cabecalhos, json={"input": "Oi"})
    assert resposta.content.startswith(b"ID3 bia")
    assert voz.estado()["qwen"]["vozes"] == ["qwen:bia"]


def test_trechos_do_qwen():
    from falar_qwen import trechos

    texto = "Primeira frase. " * 30 + "Uma frase enorme, " * 30
    partes = trechos(texto, limite=100)
    assert all(len(p) <= 100 for p in partes)
    assert " ".join(partes).split() == texto.split()  # nada some nem se repete
    assert trechos("Oi.\n\nTudo bem?") == ["Oi. Tudo bem?"]


def test_a_lista_de_vozes_tem_a_forma_publicada(qwen, cabecalhos):
    import json
    from pathlib import Path

    exemplo = json.loads(
        (Path(__file__).resolve().parent.parent / "contrato" / "vozes-resposta.json").read_text()
    )
    qwen.post("/v1/audio/voices", headers=cabecalhos, data={"nome": "bia", "descricao": "jovem"})
    resposta = qwen.get("/v1/audio/voices", headers=cabecalhos).json()

    assert set(resposta) == {k for k in exemplo if not k.startswith("_")}
    assert set(resposta["data"][0]) == set(exemplo["data"][0])
    assert set(resposta["data"][-1]) == set(exemplo["data"][-1])
