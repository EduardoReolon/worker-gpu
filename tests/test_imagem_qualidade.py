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
def test_o_worker_nao_tem_prompt_negativo_proprio(imagem_mod):
    """Houve um padrao ("text, letters, words, ...") aplicado a todo pedido, e
    ele brigava com quem pedia texto na imagem: uma planilha com linhas
    rotuladas saia como sopa de letras. O worker executa; o negativo e do
    cliente."""
    assert not hasattr(imagem_mod, "IMAGEM_NEGATIVO")
    assert imagem_mod.PedidoDeImagem(prompt="x").negative_prompt is None


def test_o_negativo_do_cliente_chega_ao_pipeline_como_veio(imagem_mod, monkeypatch):
    import sys
    import types

    visto = {}

    class _Resultado:
        images = ()

    def pipe(**argumentos):
        visto.update(argumentos)
        return _Resultado()

    torch_falso = types.ModuleType("torch")
    torch_falso.Generator = lambda device: None
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    monkeypatch.setattr(imagem_mod, "_obter_pipeline", lambda dispositivo: pipe)

    pedido = imagem_mod.PedidoDeImagem(prompt="x", negative_prompt="blurry, watermark")
    imagem_mod.gerar_imagens("cpu", pedido, 1, 1024, 1024)
    assert visto["negative_prompt"] == "blurry, watermark"

    imagem_mod.gerar_imagens("cpu", imagem_mod.PedidoDeImagem(prompt="x"), 1, 1024, 1024)
    assert visto["negative_prompt"] is None


def test_um_prompt_em_portugues_gera_aviso(imagem_mod, caplog):
    """O worker NAO traduz: traduzir em silencio mudaria o pedido de quem
    chamou. Ele avisa, e quem integra decide."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_prompt_em_portugues(
            "uma fotografia de um laboratorio com luz natural"
        )

    assert "portugues" in caplog.text


def test_o_aviso_de_portugues_e_so_do_sdxl(imagem_mod, caplog, monkeypatch):
    """O aviso e sobre os CLIP do SDXL. O Z-Image le o prompt com um modelo de
    linguagem que entende portugues; avisar ali mandaria mudar o que esta
    certo."""
    monkeypatch.setattr(imagem_mod, "_familia_do_pipeline", "outra")

    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_prompt_em_portugues(
            "uma fotografia de um laboratorio com luz natural"
        )

    assert caplog.text == ""


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


def test_um_amostrador_classico_num_modelo_de_flow_matching_recusa(imagem_mod):
    """Z-Image, FLUX e SD 3.5 usam flow matching. O `from_config` de um DPM++
    ali nao reclama — gera ruido. O `.env` herdado do SDXL traz
    `IMAGEM_SCHEDULER=dpm++2m_karras`, entao e o primeiro erro de quem troca de
    modelo."""

    class FlowMatchEulerDiscreteScheduler:
        pass

    pipe = _PipeFalso()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()

    with pytest.raises(RuntimeError, match="IMAGEM_SCHEDULER vazio"):
        imagem_mod.aplicar_scheduler(pipe, "dpm++2m_karras")


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


# ---------------------------------------------------------------------------
# A grade de proporcoes do SDXL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tamanho", ["1024x1024", "1344x768", "1536x640", "768x1344"])
def test_a_grade_de_treino_inteira_passa(imagem_mod, tamanho):
    """Com o teto por lado em 1024 as proporcoes largas voltavam 422 — dava
    para pedir so a quadrada, que e a pior escolha para uma capa."""
    largura, altura = imagem_mod._medidas(tamanho)

    assert (largura, altura) in imagem_mod.GRADE_DO_SDXL


def test_a_grade_fica_toda_perto_de_um_megapixel(imagem_mod):
    """E uma grade de PROPORCOES a area constante, e nao de tamanhos livres.
    E por isso que o teto de area protege a qualidade junto com a placa."""
    areas = [largura * altura / 1e6 for largura, altura in imagem_mod.GRADE_DO_SDXL]

    assert 0.94 <= min(areas) and max(areas) <= 1.06


def test_uma_area_grande_demais_e_recusada_com_a_alternativa(imagem_mod):
    """O teto por lado sozinho deixaria passar 1536x1536: dentro do lado
    maximo, e o dobro da area de treino. Nao daria erro — daria assunto
    duplicado e VRAM que esta placa nao tem."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as capturado:
        imagem_mod._medidas("1536x1536")

    detalhe = capturado.value.detail
    assert "megapixels" in detalhe
    # E diz o que pedir no lugar, em vez de so recusar.
    assert "1024x1024" in detalhe


@pytest.mark.parametrize(
    "pedido,esperado",
    [
        ("1200x630", "1344x704"),
        ("1024x576", "1344x768"),
        ("800x800", "1024x1024"),
        ("600x900", "832x1216"),
    ],
)
def test_a_alternativa_e_escolhida_pela_proporcao(imagem_mod, pedido, esperado):
    """Pela proporcao e nao pela area: a area de toda a grade e praticamente a
    mesma, entao comparar por area mandaria todo mundo para 1024x1024. Quem
    pede 1200x630 quer aquele FORMATO."""
    largura, altura = (int(parte) for parte in pedido.split("x"))

    assert imagem_mod._perto_na_grade(largura, altura) == esperado


def test_um_tamanho_fora_da_grade_avisa_mas_nao_recusa(imagem_mod, caplog, monkeypatch):
    """Recusar quebraria quem tem motivo para pedir outro formato. Quem pede
    assim aceita o custo; o que nao se aceita e pagar sem saber que existe."""
    monkeypatch.setattr(imagem_mod, "_familia_do_pipeline", "sdxl")

    # Nao levanta.
    assert imagem_mod._medidas("1200x640") == (1200, 640)

    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_tamanho_fora_da_grade(1200, 640)

    assert "fora da grade" in caplog.text
    assert "1344x704" in caplog.text


def test_o_og_image_classico_nem_chega_na_grade(imagem_mod):
    """1200x630, o tamanho que as redes sociais pedem para `og:image`, e
    recusado ANTES da grade: 630 nao e multiplo de 16. A recusa e certa — o
    modelo arredondaria por dentro e devolveria outro tamanho, sem avisar —,
    mas quem integra precisa saber que esse valor exato nao passa."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="multiplo de 16"):
        imagem_mod._medidas("1200x630")


def test_um_tamanho_na_grade_nao_avisa(imagem_mod, caplog, monkeypatch):
    monkeypatch.setattr(imagem_mod, "_familia_do_pipeline", "sdxl")

    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_tamanho_fora_da_grade(1344, 768)

    assert caplog.text == ""


def test_a_grade_nao_e_cobrada_de_outra_familia(imagem_mod, caplog, monkeypatch):
    """A grade e do SDXL. Avisar sobre ela num SD 3.5 ou num FLUX mandaria
    mudar o que esta certo."""
    monkeypatch.setattr(imagem_mod, "_familia_do_pipeline", "outra")

    with caplog.at_level(logging.WARNING, logger="worker-gpu.imagem"):
        imagem_mod._avisar_de_tamanho_fora_da_grade(1200, 630)

    assert caplog.text == ""


def test_o_health_publica_a_grade(worker):
    """Publicada para o cliente validar contra ELA, e nao contra uma copia
    propria que envelhece — foi assim que o `busy` virou `ocupada` sem ninguem
    perceber."""
    from fastapi.testclient import TestClient

    bloco = TestClient(worker.app).get("/health/").json()["imagem"]

    assert "1344x768" in bloco["grade"]
    assert "1536x640" in bloco["grade"]
    assert bloco["area_maxima_mp"] >= 1.05


# ---------------------------------------------------------------------------
# Precisao e quantizacao
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("chave", "valor", "trecho"),
    [
        ("IMAGEM_DTYPE", "fp16", "float16 ou bfloat16"),
        ("IMAGEM_QUANTIZAR", "4bits", "nao, 4bit ou 8bit"),
    ],
)
def test_valores_invalidos_recusam_na_subida(imagem_mod, monkeypatch, chave, valor, trecho):
    monkeypatch.setattr(imagem_mod, chave, valor)

    with pytest.raises(RuntimeError, match=trecho):
        imagem_mod.conferir_configuracao()


def test_quantizar_sem_bitsandbytes_recusa_na_subida(imagem_mod, monkeypatch):
    """O modelo carrega no primeiro pedido. Sem esta conferencia, o
    `bitsandbytes` ausente so apareceria ali, como 500 numa imagem."""
    monkeypatch.setattr(imagem_mod, "IMAGEM_QUANTIZAR", "4bit")
    monkeypatch.setattr(imagem_mod.importlib.util, "find_spec", lambda nome: None)

    with pytest.raises(RuntimeError, match="pip install bitsandbytes"):
        imagem_mod.conferir_configuracao()


def test_o_padrao_sobe(imagem_mod):
    imagem_mod.conferir_configuracao()
