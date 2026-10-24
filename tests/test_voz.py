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
    # Sem ffmpeg: o audio sai como o motor gravou, e os testes conferem isso.
    monkeypatch.setattr(voz, "_ffmpeg", lambda: None)
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
p.add_argument("--sotaque")
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
    assert "Brazil" in a.sotaque
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


def test_instrucao_e_falas_vizinhas_sao_aceitas_e_ignoradas(cliente, cabecalhos):
    """O PubliBot manda os campos da OpenAI e da ElevenLabs; o worker nao os
    usa, nao quebra, e diz isso no /health/."""
    import voz

    resposta = cliente.post(
        "/v1/audio/speech", headers=cabecalhos,
        json={
            "voice": "pf_dora", "input": "Oi", "instructions": "como quem conversa",
            "previous_text": "antes", "next_text": "depois",
        },
    )  # fmt: skip
    assert resposta.status_code == 200
    assert resposta.content == b"ID3 pf_dora 1.0 Oi"
    assert voz.estado()["campos"] == {
        "instructions": False, "previous_text": False, "next_text": False,
    }  # fmt: skip


FALSO_CONVERSAO = """
import argparse, pathlib, sys
p = argparse.ArgumentParser()
p.add_argument("--origem"); p.add_argument("--alvo"); p.add_argument("--saida")
p.add_argument("--dispositivo"); p.add_argument("--segundos-maximos")
a = p.parse_args()
origem = open(a.origem, "rb").read()
if origem == b"longo":
    print("o audio tem 300 s; o maximo e 120 s", file=sys.stderr); sys.exit(2)
alvo = pathlib.Path(a.alvo)
open(a.saida, "wb").write(b"ID3 " + alvo.parent.name.encode() + b" " + origem)
print("FORMATO mp3")
"""


@pytest.fixture
def conversao(qwen, tmp_path, monkeypatch):
    import voz

    script = tmp_path / "falso_conversao.py"
    script.write_text(FALSO_CONVERSAO)
    monkeypatch.setattr(voz, "VOZ_CONVERSAO", True)
    monkeypatch.setattr(voz, "SCRIPT_CONVERSAO", script)
    monkeypatch.setattr(voz, "VOZ_CONVERSAO_PYTHON", sys.executable)
    return qwen


def _converter(cliente, cabecalhos, voz="qwen:carla", conteudo=b"minha fala"):
    return cliente.post(
        "/v1/audio/convert", headers=cabecalhos, data={"voice": voz},
        files={"audio": ("fala.mp3", conteudo, "audio/mpeg")},
    )  # fmt: skip


def test_converte_a_gravacao_para_o_timbre_da_voz(conversao, cabecalhos):
    conversao.post(
        "/v1/audio/voices", headers=cabecalhos, data={"nome": "carla", "descricao": "calorosa"}
    )
    conversao.soltou.clear()
    resposta = _converter(conversao, cabecalhos)
    assert resposta.status_code == 200
    assert resposta.headers["content-type"] == "audio/mpeg"
    assert resposta.content == b"ID3 carla minha fala"
    assert conversao.soltou == ["converter a voz"]  # tomou e esvaziou a placa

    import voz

    assert voz.estado()["conversao"]["ultima"]["voz"] == "qwen:carla"


def test_erros_da_conversao(conversao, cabecalhos, monkeypatch):
    import arbitro
    import voz

    assert _converter(conversao, cabecalhos).json()["error"]["code"] == "voz_desconhecida"
    kokoro = _converter(conversao, cabecalhos, voz="pf_dora")
    assert kokoro.status_code == 422
    assert "referencia" in kokoro.json()["error"]["message"]

    conversao.post(
        "/v1/audio/voices", headers=cabecalhos, data={"nome": "carla", "descricao": "calorosa"}
    )
    longo = _converter(conversao, cabecalhos, conteudo=b"longo")
    assert longo.status_code == 422
    assert "maximo" in longo.json()["error"]["message"]
    assert _converter(conversao, cabecalhos, conteudo=b"").status_code == 422

    with arbitro.ARBITRO.usar("animacao"):
        ocupada = _converter(conversao, cabecalhos)
    assert ocupada.status_code == 503
    assert ocupada.json()["error"]["code"] == "gpu_ocupada"

    monkeypatch.setattr(voz, "VOZ_CONVERSAO", False)
    assert _converter(conversao, cabecalhos).status_code == 404


FALSO_FFMPEG = """#!{python}
import sys
dados = sys.stdin.buffer.read()
if dados.startswith(b"ID3 quebre"):
    sys.stderr.write("filtro invalido"); sys.exit(1)
sys.stdout.buffer.write(b"LIMPO " + dados)
"""


@pytest.fixture
def ffmpeg_falso(tmp_path, monkeypatch):
    import voz

    falso = tmp_path / "ffmpeg"
    falso.write_text(FALSO_FFMPEG.format(python=sys.executable))
    falso.chmod(0o755)
    monkeypatch.setattr(voz, "_ffmpeg", lambda: str(falso))


def test_acabamento_em_toda_fala(cliente, cabecalhos, ffmpeg_falso):
    resposta = cliente.post(
        "/v1/audio/speech", headers=cabecalhos, json={"voice": "pf_dora", "input": "Oi"}
    )
    assert resposta.status_code == 200
    assert resposta.content == b"LIMPO ID3 pf_dora 1.0 Oi"
    assert resposta.headers["content-type"] == "audio/mpeg"


def test_acabamento_que_falha_nao_derruba_a_fala(worker, ffmpeg_falso, monkeypatch):
    import voz

    assert voz.acabamento(b"ID3 quebre", "wav") == (b"ID3 quebre", "wav")
    monkeypatch.setattr(voz, "VOZ_ACABAMENTO", False)
    assert voz.acabamento(b"ID3 x", "mp3") == (b"ID3 x", "mp3")
    assert voz.estado()["acabamento"] == {"ligado": False, "ffmpeg": True}
