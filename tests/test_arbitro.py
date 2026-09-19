"""O lock que protege a placa.

Estes testes valem por todo o resto: se o arbitro deixa dois trabalhos
entrarem, a VRAM estoura e o sintoma nao e um erro — e o processo caindo para
CPU em silencio, dezenas de vezes mais lento.
"""

from __future__ import annotations

import importlib
import threading
import time

import pytest


@pytest.fixture
def arbitro(ambiente):
    import config

    importlib.reload(config)
    import arbitro as modulo

    return importlib.reload(modulo)


def test_um_trabalho_por_vez(arbitro):
    with arbitro.ARBITRO.usar("imagem"):
        with pytest.raises(arbitro.GpuOcupada):
            with arbitro.ARBITRO.usar("texto"):
                pass


def test_a_placa_volta_a_ficar_livre_depois(arbitro):
    with arbitro.ARBITRO.usar("imagem"):
        pass

    with arbitro.ARBITRO.usar("texto"):
        assert arbitro.ARBITRO.ocupada


def test_a_placa_e_liberada_mesmo_com_excecao(arbitro):
    """Sem isto, um erro deixaria a GPU marcada como ocupada ate o restart — e
    todo cliente levaria 503 para sempre."""
    with pytest.raises(ValueError):
        with arbitro.ARBITRO.usar("imagem"):
            raise ValueError("qualquer coisa")

    assert not arbitro.ARBITRO.ocupada


def test_a_recusa_diz_quem_esta_usando(arbitro):
    """ "Ocupado" e verdadeiro e inutil. "Ocupado gerando imagem, tente em 40s"
    e o que deixa o cliente decidir."""
    with arbitro.ARBITRO.usar("imagem"):
        with pytest.raises(arbitro.GpuOcupada) as erro:
            with arbitro.ARBITRO.usar("texto"):
                pass

    assert erro.value.ocupante == "imagem"
    assert erro.value.falta > 0


def test_o_retry_after_encolhe_conforme_o_trabalho_avanca(arbitro, monkeypatch):
    """Calculado, e nao constante: mandar voltar em 60s quando falta meio
    segundo desperdicia, e mandar voltar em 60s quando falta um minuto e meio
    garante um segundo 503."""
    monkeypatch.setattr(arbitro, "DURACAO_ESTIMADA", {"imagem": 60})

    recente = arbitro.Ocupacao(tarefa="imagem", desde=time.monotonic())
    antiga = arbitro.Ocupacao(tarefa="imagem", desde=time.monotonic() - 50)

    assert recente.falta_estimado > antiga.falta_estimado


def test_o_retry_after_nunca_e_zero(arbitro, monkeypatch):
    """`Retry-After: 0` convida o cliente a voltar imediatamente e levar
    outro 503 — um laco apertado entre dois processos."""
    monkeypatch.setattr(arbitro, "DURACAO_ESTIMADA", {"imagem": 60})
    estourada = arbitro.Ocupacao(tarefa="imagem", desde=time.monotonic() - 9999)

    assert estourada.falta_estimado >= 5


def test_de_fato_serializa_entre_threads(arbitro):
    """O teste que importa: duas threads, e nunca as duas dentro ao mesmo
    tempo. E o cenario real — o uvicorn atende cada pedido numa thread."""
    dentro = []
    maximo = []
    trava = threading.Lock()

    def trabalhar():
        try:
            with arbitro.ARBITRO.usar("imagem", espera=5.0):
                with trava:
                    dentro.append(1)
                    maximo.append(len(dentro))
                time.sleep(0.05)
                with trava:
                    dentro.pop()
        except arbitro.GpuOcupada:
            pass

    threads = [threading.Thread(target=trabalhar) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert maximo, "nenhuma thread conseguiu entrar"
    assert max(maximo) == 1
