"""A rota de edicao, pelo HTTP, com a difusao substituida.

O que se exercita e o que o PubliBot espera: a escolha do modelo pelo pedido
(com referencia -> o de IP-Adapter; sem -> o de mais qualidade), a mascara da
OpenAI convertida para a do diffusers, a parte opaca intacta pixel a pixel, e
os erros de pedido.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _png(figura) -> bytes:
    memoria = io.BytesIO()
    figura.save(memoria, format="PNG")
    return memoria.getvalue()


def _base(cor=(10, 200, 30), tamanho=(64, 96)):
    return Image.new("RGB", tamanho, cor)


def _mascara(tamanho=(64, 96), manter_topo=32):
    """Opaca (manter) nas `manter_topo` linhas de cima, transparente no resto."""
    figura = Image.new("RGBA", tamanho, (0, 0, 0, 0))
    for y in range(manter_topo):
        for x in range(tamanho[0]):
            figura.putpixel((x, y), (0, 0, 0, 255))
    return figura


@pytest.fixture
def edicao_mod(worker):
    import edicao

    return edicao


@pytest.fixture
def cliente(worker, edicao_mod, monkeypatch):
    import imagem
    import ollama

    chamadas = []

    def editar_imagens(dispositivo, **argumentos):
        chamadas.append(argumentos)
        return [b"png-editada"] * argumentos["quantas"]

    monkeypatch.setattr(edicao_mod, "editar_imagens", editar_imagens)
    monkeypatch.setattr(imagem, "resolver_dispositivo", lambda: "cuda")
    monkeypatch.setattr(ollama, "descarregar_tudo", lambda: None)
    cliente = TestClient(worker.app)
    cliente.chamadas = chamadas
    return cliente


def _pedir(cliente, cabecalhos, arquivos=(), **dados):
    return cliente.post(
        "/v1/images/edits",
        data={"prompt": "a man at a desk", "size": "720x1280", **dados},
        files=list(arquivos),
        headers=cabecalhos,
    )


def test_exige_credencial(cliente):
    assert cliente.post("/v1/images/edits", data={"prompt": "x"}).status_code == 401


def test_so_referencias_gera_do_zero_com_elas(cliente, cabecalhos):
    refs = [("reference_image", (f"r{i}.png", _png(_base()), "image/png")) for i in range(2)]

    resposta = _pedir(cliente, cabecalhos, refs, reference_strength="0.7")

    assert resposta.status_code == 200
    assert base64.b64decode(resposta.json()["data"][0]["b64_json"]) == b"png-editada"
    chamada = cliente.chamadas[0]
    assert len(chamada["referencias"]) == 2
    assert chamada["forca"] == 0.7
    assert chamada["mascara"] is None
    assert (chamada["largura"], chamada["altura"]) == (720, 1280)


def test_base_e_mascara_vao_juntas_e_redimensionadas_ao_size(cliente, cabecalhos):
    arquivos = [
        ("image", ("base.png", _png(_base()), "image/png")),
        ("mask", ("mascara.png", _png(_mascara()), "image/png")),
    ]

    resposta = _pedir(cliente, cabecalhos, arquivos, size="128x192")

    assert resposta.status_code == 200
    chamada = cliente.chamadas[0]
    assert chamada["base"].size == (128, 192)
    assert chamada["mascara"].size == (128, 192)
    assert chamada["referencias"] == []


def test_mascara_sem_alfa_e_422(cliente, cabecalhos):
    arquivos = [
        ("image", ("base.png", _png(_base()), "image/png")),
        ("mask", ("mascara.png", _png(_base()), "image/png")),
    ]

    resposta = _pedir(cliente, cabecalhos, arquivos)

    assert resposta.status_code == 422
    assert "canal alfa" in resposta.json()["error"]["message"]


def test_mascara_de_outro_tamanho_e_422(cliente, cabecalhos):
    arquivos = [
        ("image", ("base.png", _png(_base()), "image/png")),
        ("mask", ("mascara.png", _png(_mascara(tamanho=(32, 32))), "image/png")),
    ]

    resposta = _pedir(cliente, cabecalhos, arquivos)

    assert resposta.status_code == 422


def test_mascara_sem_image_e_422(cliente, cabecalhos):
    resposta = _pedir(cliente, cabecalhos, [("mask", ("m.png", _png(_mascara()), "image/png"))])

    assert resposta.status_code == 422


def test_referencias_demais_e_422(cliente, cabecalhos):
    refs = [("reference_image", (f"r{i}.png", _png(_base()), "image/png")) for i in range(4)]

    resposta = _pedir(cliente, cabecalhos, refs)

    assert resposta.status_code == 422
    assert "maximo" in resposta.json()["error"]["message"]


def test_arquivo_que_nao_e_imagem_e_422(cliente, cabecalhos):
    refs = [("reference_image", ("r.png", b"isto nao e png", "image/png"))]

    resposta = _pedir(cliente, cabecalhos, refs)

    assert resposta.status_code == 422


def test_forca_fora_de_0_a_1_e_422(cliente, cabecalhos):
    refs = [("reference_image", ("r.png", _png(_base()), "image/png"))]

    assert _pedir(cliente, cabecalhos, refs, reference_strength="1.5").status_code == 422


def test_sem_modelo_de_referencia_as_referencias_viram_aviso(cliente, cabecalhos, monkeypatch):
    import imagem

    monkeypatch.setattr(imagem, "IMAGEM_REFERENCIA_MODELO", "")
    refs = [("reference_image", ("r.png", _png(_base()), "image/png"))]

    resposta = _pedir(cliente, cabecalhos, refs)

    assert resposta.status_code == 200
    assert cliente.chamadas[0]["referencias"] == []
    assert "IMAGEM_REFERENCIA_MODELO" in resposta.json()["avisos"][0]


def test_placa_ocupada_e_503(cliente, cabecalhos):
    import arbitro

    with arbitro.ARBITRO.usar("texto"):
        resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 503
    assert resposta.json()["error"]["code"] == "gpu_ocupada"


def test_falha_e_500_com_codigo(cliente, cabecalhos, edicao_mod, monkeypatch):
    def quebra(dispositivo, **argumentos):
        raise RuntimeError("algo")

    monkeypatch.setattr(edicao_mod, "editar_imagens", quebra)

    resposta = _pedir(cliente, cabecalhos)

    assert resposta.status_code == 500
    assert resposta.json()["error"]["code"] == "falha_na_edicao"


def test_o_health_publica_a_rota_e_o_modelo_de_referencia(cliente):
    corpo = cliente.get("/health/").json()

    assert corpo["rotas"]["imagem_edicao"] is True
    assert corpo["imagem"]["referencia"]["modelo"] == "SG161222/RealVisXL_V5.0"


# ---------------------------------------------------------------------------
# A mascara e a parte que fica
# ---------------------------------------------------------------------------
def test_a_mascara_da_openai_vira_a_do_diffusers(edicao_mod):
    """OpenAI: transparente = pintar. Diffusers: branco = pintar."""
    pintar, manter = edicao_mod.mascara_para_o_diffusers(_mascara())

    assert manter.getpixel((5, 5)) == 255 and pintar.getpixel((5, 5)) == 0  # topo: manter
    assert manter.getpixel((5, 80)) == 0 and pintar.getpixel((5, 80)) == 255  # baixo: pintar


def test_a_parte_opaca_sai_identica_a_base(edicao_mod):
    """Pintar passa tudo pelo VAE, e o que devia ficar volta levemente
    diferente. A foto da janela tem que sair pixel a pixel."""
    base = _base(cor=(10, 200, 30))
    gerada = _base(cor=(250, 0, 0))
    _, manter = edicao_mod.mascara_para_o_diffusers(_mascara())

    final = edicao_mod.colar_o_que_fica(base, gerada, manter)

    assert final.getpixel((10, 10)) == (10, 200, 30)
    assert final.getpixel((10, 90)) == (250, 0, 0)


# ---------------------------------------------------------------------------
# A escolha do modelo
# ---------------------------------------------------------------------------
def test_com_referencia_usa_o_perfil_de_ip_adapter_e_sem_usa_o_principal(edicao_mod, monkeypatch):
    import sys
    import types

    import imagem

    vistos = []

    class _Pipe:
        def set_ip_adapter_scale(self, forca):
            vistos.append(("forca", forca))

        def __call__(self, **argumentos):
            vistos.append(("chamada", set(argumentos)))
            return types.SimpleNamespace(images=[_base()])

    def obter(dispositivo, perfil, modo):
        vistos.append((perfil.nome, modo))
        return _Pipe()

    torch_falso = types.ModuleType("torch")
    torch_falso.Generator = lambda device: types.SimpleNamespace(manual_seed=lambda s: s)
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    monkeypatch.setattr(imagem, "_obter_pipeline", obter)
    comum = {
        "prompt": "x",
        "negativo": None,
        "quantas": 1,
        "largura": 64,
        "altura": 96,
        "semente": 1,
    }

    edicao_mod.editar_imagens("cuda", referencias=[_base()], forca=0.5, **comum)
    edicao_mod.editar_imagens("cuda", base=_base(), mascara=_mascara(), **comum)

    assert ("referencia", "base") in vistos
    assert ("forca", {"up": {"block_0": [0.0, 0.5, 0.0]}}) in vistos
    assert ("principal", "pintura") in vistos
    chamadas = [v[1] for v in vistos if v[0] == "chamada"]
    assert "ip_adapter_image" in chamadas[0] and "mask_image" not in chamadas[0]
    assert "mask_image" in chamadas[1] and "ip_adapter_image" not in chamadas[1]


def test_so_um_pipeline_fica_com_os_ganchos_de_offload(worker, monkeypatch):
    """O de pintura reaproveita os modulos do base. Com os ganchos de offload
    instalados pelos dois nos mesmos modulos, um desfaria o outro."""
    import sys
    import types

    import imagem

    eventos = []

    class _Pipe:
        def __init__(self, nome):
            self.nome = nome

        def remove_all_hooks(self):
            eventos.append(f"{self.nome}:remove")

        def enable_model_cpu_offload(self):
            eventos.append(f"{self.nome}:offload")

        def set_progress_bar_config(self, **_):
            pass

    diffusers = types.ModuleType("diffusers")
    diffusers.AutoPipelineForInpainting = types.SimpleNamespace(
        from_pipe=lambda base: eventos.append("from_pipe") or _Pipe("pintura")
    )
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)

    carregado = imagem._Carregado(_Pipe("base"), imagem.perfil_principal(), "cuda")
    carregado.ativar("base")  # ja vem com offload da carga: nada a fazer
    carregado.ativar("pintura")
    carregado.ativar("pintura")  # mesmo modo: nada a fazer
    carregado.ativar("base")

    assert eventos == ["base:remove", "from_pipe", "pintura:offload", "base:offload"]


def test_pedir_outro_perfil_descarrega_o_atual(worker, monkeypatch):
    """Um so modelo de imagem cabe: o de referencia expulsa o principal."""
    import types

    import imagem

    montados = []
    monkeypatch.setattr(
        imagem,
        "_montar_pipeline",
        lambda dispositivo, perfil: montados.append(perfil.nome) or types.SimpleNamespace(),
    )
    monkeypatch.setattr(imagem.recursos, "devolver_memoria", lambda rotulo: None)

    imagem._obter_pipeline("cpu", imagem.perfil_principal())
    imagem._obter_pipeline("cpu", imagem.perfil_principal())
    imagem._obter_pipeline("cpu", imagem.perfil_referencia())

    assert montados == ["principal", "referencia"]
    assert imagem._pipeline.perfil.nome == "referencia"


# ---------------------------------------------------------------------------
# InstantStyle: a referencia so na aparencia
# ---------------------------------------------------------------------------
def test_por_padrao_a_referencia_so_entra_na_camada_de_aparencia(edicao_mod):
    """`up.block_0` e a aparencia; `down.block_2`, a composicao, fica de fora:
    a cena segue o prompt em vez de copiar a pose da referencia."""
    assert edicao_mod.escala_da_referencia(0.6) == {"up": {"block_0": [0.0, 0.6, 0.0]}}


def test_com_todas_a_referencia_entra_no_modelo_inteiro(edicao_mod, monkeypatch):
    monkeypatch.setattr(edicao_mod, "IMAGEM_REFERENCIA_CAMADAS", "todas")

    assert edicao_mod.escala_da_referencia(0.6) == 0.6


def test_valor_invalido_recusa_subir(worker, monkeypatch):
    import imagem

    monkeypatch.setattr(imagem, "IMAGEM_REFERENCIA_CAMADAS", "metade")

    with pytest.raises(RuntimeError, match="aparencia ou todas"):
        imagem.conferir_configuracao()
