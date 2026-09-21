"""Os ajustes que decidem a qualidade da imagem.

Este arquivo existe porque nenhum deles quebra nada quando esta errado. Um
prompt negativo em portugues, um VAE que estoura em float16, um amostrador
construido do zero — os tres geram uma imagem, respondem 200, e entregam pior.
O defeito aparece semanas depois, como "as imagens tem cara de IA", e ninguem
liga a causa ao efeito.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def imagem_mod(worker):
    import imagem

    return imagem


@pytest.fixture
def diffusers_falso(monkeypatch):
    """O `diffusers` de verdade arrasta o torch, que nao entra na suite.

    O que se testa aqui e a chamada DAQUI — `from_config` com a configuracao do
    modelo —, e nao o amostrador do diffusers.
    """
    import sys
    import types

    falso = types.ModuleType("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", falso)

    return falso


# ---------------------------------------------------------------------------
# A lingua dos prompts
# ---------------------------------------------------------------------------
def test_o_prompt_negativo_padrao_esta_em_ingles(imagem_mod):
    """Esteve em portugues por muito tempo e nao fazia nada: os codificadores
    de texto do SDXL foram treinados em legendas da web, esmagadoramente
    inglesas. `borrado` nao esta no vocabulario aprendido; `blurry` esta.

    Era o pior tipo de configuracao — aparecia no `.env`, parecia ativa, e o
    efeito era o de nao ter prompt negativo nenhum.
    """
    negativo = imagem_mod.IMAGEM_NEGATIVO.lower()

    assert "blurry" in negativo
    assert "watermark" in negativo
    # As palavras portuguesas que estavam ali antes.
    assert "borrado" not in negativo
    assert "marca d'agua" not in negativo


def test_um_prompt_em_portugues_gera_aviso(imagem_mod, caplog):
    """O worker NAO traduz: traduzir em silencio mudaria o pedido de quem
    chamou. Ele avisa, e quem integra decide."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_prompt_em_portugues(
            "uma fotografia de um laboratorio com luz natural"
        )

    assert "portugues" in caplog.text


def test_um_prompt_em_ingles_nao_gera_aviso(imagem_mod, caplog):
    """Errar para "nao avisa" e o certo: um aviso a mais em todo pedido seria
    ruido no journal, e o aviso que falta e so um aviso que falta."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_prompt_em_portugues(
            "editorial photograph of a laboratory bench, soft daylight, 50mm"
        )

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# O VAE que estoura em float16
# ---------------------------------------------------------------------------
class _VaeFalso:
    def __init__(self, canais: int, escala: float):
        self.config = type("Config", (), {"latent_channels": canais, "scaling_factor": escala})()


class _PipeComVae:
    def __init__(self, canais: int, escala: float):
        self.vae = _VaeFalso(canais, escala)


def _sdxl():
    """Qualquer modelo da familia SDXL, seja `stable-diffusion-xl` ou
    `RealVisXL`: o que os identifica e este `scaling_factor`, nao o nome."""
    return _PipeComVae(4, 0.13025)


def test_um_modelo_sdxl_sem_vae_ajustado_avisa(imagem_mod, caplog):
    """Manchas, faixas de cor e as vezes imagem preta — o defeito que quem olha
    chama de "cara de IA" sem saber apontar o que e."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(_sdxl(), vae="", meia=True)

    assert "sdxl-vae-fp16-fix" in caplog.text


def test_a_familia_e_descoberta_no_objeto_e_nao_no_nome(imagem_mod, caplog):
    """`RealVisXL_V5.0` e SDXL e nao contem "sdxl"; procurar "xl" solto
    acertaria ele e erraria em qualquer nome com "xl" no meio de outra palavra.
    O `scaling_factor` do VAE e um fato do modelo."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(_sdxl(), vae="", meia=True)

    assert "0.13025" in caplog.text


def test_com_vae_ajustado_nao_avisa(imagem_mod, caplog):
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(_sdxl(), vae="madebyollin/sdxl-vae-fp16-fix", meia=True)

    assert caplog.text == ""


def test_em_float32_nao_avisa(imagem_mod, caplog):
    """O estouro e do float16. Em CPU, com float32, o VAE original esta certo —
    avisar ali mandaria trocar o que esta bom."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(_sdxl(), vae="", meia=False)

    assert caplog.text == ""


@pytest.mark.parametrize(
    "canais,escala,familia",
    [(4, 0.18215, "SD 1.5 / 2.x"), (16, 1.5305, "SD 3.5 / FLUX")],
)
def test_outras_familias_de_vae_nao_avisam(imagem_mod, caplog, canais, escala, familia):
    """Um VAE de SDXL nao encaixa nelas, e as de 16 canais nao tem o problema.
    Mandar trocar ali seria mandar quebrar."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(_PipeComVae(canais, escala), vae="", meia=True)

    assert caplog.text == "", familia


def test_um_pipeline_sem_vae_legivel_nao_derruba_a_carga(imagem_mod, caplog):
    """O aviso e um extra. Uma versao de diffusers que mude o nome do campo
    nao pode impedir o modelo de carregar."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_do_vae(object(), vae="", meia=True)

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# O amostrador
# ---------------------------------------------------------------------------
class _SchedulerFalso:
    def __init__(self):
        self.config = {"beta_start": 0.00085, "beta_end": 0.012, "num_train_timesteps": 1000}


class _PipeFalso:
    def __init__(self):
        self.scheduler = _SchedulerFalso()


def test_o_amostrador_herda_a_configuracao_do_modelo(imagem_mod, diffusers_falso):
    """`from_config` e nao um construtor novo: o amostrador precisa herdar os
    betas e os passos de treino DO MODELO. Construido do zero com os padroes da
    classe, ele gera — e gera errado, sem erro nenhum."""
    diffusers = diffusers_falso
    visto = {}

    class _Novo:
        @staticmethod
        def from_config(config, **extras):
            visto["config"] = config
            visto["extras"] = extras
            return "amostrador-novo"

    diffusers.DPMSolverMultistepScheduler = _Novo
    pipe = _PipeFalso()

    imagem_mod.aplicar_scheduler(pipe, "dpm++2m_karras")

    assert pipe.scheduler == "amostrador-novo"
    assert visto["config"]["num_train_timesteps"] == 1000
    assert visto["extras"] == {"use_karras_sigmas": True}


def test_um_amostrador_vazio_mantem_o_do_modelo(imagem_mod):
    pipe = _PipeFalso()
    original = pipe.scheduler

    imagem_mod.aplicar_scheduler(pipe, "")

    assert pipe.scheduler is original


def test_um_amostrador_inexistente_recusa_com_a_lista(imagem_mod):
    """E nao cai no padrao em silencio: um `IMAGEM_SCHEDULER` com erro de
    digitacao valeria como "o do modelo" e ninguem saberia."""
    with pytest.raises(RuntimeError, match="dpm"):
        imagem_mod.aplicar_scheduler(_PipeFalso(), "dpm2m-karras")


# ---------------------------------------------------------------------------
# Tamanho, e a area de treino do SDXL
# ---------------------------------------------------------------------------
def test_o_teto_de_lado_permite_a_proporcao_de_treino(imagem_mod):
    """O SDXL foi treinado em cerca de 1024x1024 pixels de AREA, em proporcoes
    fixas — e a de 16:9 que ele conhece e 1344x768. Com o teto em 1024 nao
    havia como nem pedir a proporcao certa."""
    assert imagem_mod.IMAGEM_LADO_MAXIMO >= 1344
    assert imagem_mod._medidas("1344x768") == (1344, 768)


def test_o_tamanho_padrao_do_pedido_e_um_bucket_de_treino(imagem_mod):
    assert imagem_mod.PedidoDeImagem(prompt="x").size == "1344x768"


def test_o_health_publica_o_que_decide_qualidade(worker):
    """`vae` vazio num modelo SDXL e um defeito silencioso. Publicado, da para
    conferir de fora, sem ler o journal."""
    from fastapi.testclient import TestClient

    bloco = TestClient(worker.app).get("/health/").json()["imagem"]

    assert "vae" in bloco
    assert "amostrador" in bloco
    assert "guidance" in bloco
    assert "lado_maximo" in bloco
