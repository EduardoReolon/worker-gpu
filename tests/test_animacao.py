"""A rota de animacao (dialeto /v1/videos), com o FramePack substituido por
um script de mentira que imprime o progresso e grava um "video".

O que se exercita e o contrato que o PubliBot espera: o trabalho assincrono,
a placa tomada no POST (503 se ocupada) e solta no fim, o download, os erros."""

from __future__ import annotations

import sys
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

FALSO = """
import argparse, sys
p = argparse.ArgumentParser()
for a in ("--framepack-dir", "--imagem", "--prompt", "--segundos", "--saida", "--resolucao",
          "--passos", "--memoria-preservada"):
    p.add_argument(a)
p.add_argument("--teacache", action="store_true")
args = p.parse_args()
if "quebre" in args.prompt:
    print("torch.OutOfMemoryError: CUDA out of memory", flush=True)
    sys.exit(1)
for n in (10, 50, 90):
    print(f"PROGRESSO {n}", flush=True)
open(args.saida, "wb").write(b"MP4:" + args.prompt.encode())
"""


@pytest.fixture
def animacao_mod(worker, tmp_path, monkeypatch):
    import animacao
    import ollama

    script = tmp_path / "falso.py"
    script.write_text(FALSO)
    monkeypatch.setattr(animacao, "SCRIPT", script)
    monkeypatch.setattr(animacao, "ANIMACAO_PYTHON", sys.executable)
    monkeypatch.setattr(animacao, "FRAMEPACK_DIR", str(tmp_path))
    descarregados = []
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: descarregados.append("ollama"))
    animacao.descarregados = descarregados
    return animacao


@pytest.fixture
def cliente(animacao_mod):
    app = FastAPI()
    app.include_router(animacao_mod.router)
    return TestClient(app)


def _criar(cliente, cabecalhos, prompt="The man waves.", segundos="3"):
    return cliente.post(
        "/v1/videos",
        headers=cabecalhos,
        data={"prompt": prompt, "model": "ltx-video", "seconds": segundos, "size": "720x1280"},
        files={"input_reference": ("lamina.png", b"PNG", "image/png")},
    )


def _esperar(cliente, cabecalhos, identificador):
    for _ in range(100):
        dados = cliente.get(f"/v1/videos/{identificador}", headers=cabecalhos).json()
        if dados["status"] in ("completed", "failed"):
            return dados
        time.sleep(0.05)
    raise AssertionError("o trabalho nao terminou")


def test_cria_espera_e_baixa(cliente, cabecalhos, animacao_mod):
    from arbitro import ARBITRO

    resposta = _criar(cliente, cabecalhos)
    assert resposta.status_code == 200
    trabalho = resposta.json()
    assert trabalho["object"] == "video" and trabalho["model"] == "framepack"
    assert trabalho["seconds"] == "3" and trabalho["size"] == "720x1280"

    pronto = _esperar(cliente, cabecalhos, trabalho["id"])
    assert pronto["status"] == "completed" and pronto["progress"] == 100
    video = cliente.get(f"/v1/videos/{trabalho['id']}/content", headers=cabecalhos)
    assert video.status_code == 200 and video.content == b"MP4:The man waves."
    assert video.headers["content-type"] == "video/mp4"
    assert not ARBITRO.ocupada  # a placa foi devolvida
    assert "ollama" in animacao_mod.descarregados  # e soltaram o Ollama antes


def test_placa_ocupada_da_503_e_id_desconhecido_da_404(cliente, cabecalhos):
    from arbitro import ARBITRO

    with ARBITRO.usar("imagem"):
        resposta = _criar(cliente, cabecalhos)
    assert resposta.status_code == 503 and "Retry-After" in resposta.headers
    assert resposta.json()["error"]["code"] == "gpu_ocupada"
    sumido = cliente.get("/v1/videos/video_nao_existe", headers=cabecalhos)
    assert sumido.status_code == 404 and sumido.json()["error"]["code"] == "nao_encontrado"


def test_falta_de_vram_vira_failed_e_solta_a_placa(cliente, cabecalhos):
    from arbitro import ARBITRO

    trabalho = _criar(cliente, cabecalhos, prompt="quebre por favor").json()
    falho = _esperar(cliente, cabecalhos, trabalho["id"])
    assert falho["status"] == "failed" and falho["error"]["code"] == "sem_vram"
    conteudo = cliente.get(f"/v1/videos/{trabalho['id']}/content", headers=cabecalhos)
    assert conteudo.status_code == 409
    assert not ARBITRO.ocupada


def test_segundos_no_teto_e_pedido_invalido(cliente, cabecalhos, animacao_mod):
    trabalho = _criar(cliente, cabecalhos, segundos="30").json()
    assert trabalho["seconds"] == str(animacao_mod.ANIMACAO_SEGUNDOS_MAXIMOS)
    _esperar(cliente, cabecalhos, trabalho["id"])
    sem_imagem = cliente.post("/v1/videos", headers=cabecalhos, data={"prompt": "x"})
    assert sem_imagem.status_code == 422


def test_exige_credencial(cliente):
    assert cliente.get("/v1/videos/qualquer").status_code == 401


def test_a_resposta_tem_a_forma_publicada(cliente, cabecalhos):
    import json
    from pathlib import Path

    exemplo = json.loads(
        (Path(__file__).resolve().parent.parent / "contrato" / "video-resposta.json").read_text()
    )
    exemplo = {k: v for k, v in exemplo.items() if not k.startswith("_")}
    trabalho = _criar(cliente, cabecalhos).json()
    assert set(exemplo) <= set(trabalho)
    assert all(type(exemplo[k]) is type(trabalho[k]) for k in exemplo)
    _esperar(cliente, cabecalhos, trabalho["id"])


def test_conferir_configuracao_recusa_sem_framepack(animacao_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(animacao_mod, "FRAMEPACK_DIR", str(tmp_path / "nao_existe"))
    with pytest.raises(RuntimeError, match="FRAMEPACK_DIR"):
        animacao_mod.conferir_configuracao()
    (tmp_path / "diffusers_helper").mkdir()
    monkeypatch.setattr(animacao_mod, "FRAMEPACK_DIR", str(tmp_path))
    animacao_mod.conferir_configuracao()
