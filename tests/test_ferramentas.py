"""As ferramentas de terminal, no ponto onde elas podem mentir.

`bancada.py`, `medir_imagem.py` e `baixar_modelo.py` leem o `.env` para medir
a MESMA configuracao que a unit roda. Se a leitura delas divergir da do
systemd, a ferramenta mede uma coisa, o servico roda outra, e ela ainda jura
que mediu o que esta em uso — que e a pior forma de errar que uma bancada tem.
"""

from __future__ import annotations

import os

import pytest

FERRAMENTAS = ("bancada", "medir_imagem", "baixar_modelo")


@pytest.fixture
def env_falso(tmp_path, monkeypatch):
    """Um `.env` de mentira, com o `RAIZ` das ferramentas apontado para ele."""

    def montar(conteudo: str, modulo: str):
        import importlib

        (tmp_path / ".env").write_text(conteudo, encoding="utf-8")
        ferramenta = importlib.import_module(modulo)

        if hasattr(ferramenta, "RAIZ"):
            monkeypatch.setattr(ferramenta, "RAIZ", tmp_path)
        else:
            # `baixar_modelo` resolve o caminho a partir do proprio arquivo.
            falso = type("Falso", (), {"resolve": lambda self: self, "parent": tmp_path})()
            monkeypatch.setattr(ferramenta, "Path", lambda *a: falso)

        return ferramenta

    return montar


@pytest.mark.parametrize("modulo", FERRAMENTAS)
def test_a_chave_repetida_vale_a_ultima_como_no_systemd(env_falso, monkeypatch, modulo):
    """`cat >> .env` duplica a chave, e e como quase todo mundo edita.

    O systemd, lendo `EnvironmentFile`, faz a ULTIMA valer. Estas ferramentas
    faziam `setdefault` linha a linha, entao a PRIMEIRA valia — e a bancada
    media 25 passos enquanto o servico rodava com 40, sem nada denunciando.
    """
    monkeypatch.delenv("IMAGEM_PASSOS", raising=False)
    ferramenta = env_falso("IMAGEM_PASSOS=25\nIMAGEM_PASSOS=40\n", modulo)

    ferramenta._do_env()

    assert os.environ["IMAGEM_PASSOS"] == "40"


@pytest.mark.parametrize("modulo", FERRAMENTAS)
def test_o_ambiente_de_verdade_manda_mais_que_o_arquivo(env_falso, monkeypatch, modulo):
    """E `setdefault` e nao atribuicao: e assim que se passa
    `IMAGEM_DEVICE=cpu ./venv/bin/python bancada.py` para uma rodada so."""
    monkeypatch.setenv("IMAGEM_PASSOS", "99")
    ferramenta = env_falso("IMAGEM_PASSOS=25\n", modulo)

    ferramenta._do_env()

    assert os.environ["IMAGEM_PASSOS"] == "99"


@pytest.mark.parametrize("modulo", FERRAMENTAS)
def test_comentario_e_linha_vazia_nao_viram_variavel(env_falso, monkeypatch, modulo):
    monkeypatch.delenv("IMAGEM_PASSOS", raising=False)
    monkeypatch.delenv("COMENTADA", raising=False)
    ferramenta = env_falso("\n# COMENTADA=nao-deve-entrar\n\nIMAGEM_PASSOS=30\n  \n", modulo)

    ferramenta._do_env()

    assert os.environ["IMAGEM_PASSOS"] == "30"
    assert "COMENTADA" not in os.environ


@pytest.mark.parametrize("modulo", FERRAMENTAS)
def test_aspas_em_volta_do_valor_saem(env_falso, monkeypatch, modulo):
    monkeypatch.delenv("IMAGEM_MODELO", raising=False)
    ferramenta = env_falso('IMAGEM_MODELO="SG161222/RealVisXL_V5.0"\n', modulo)

    ferramenta._do_env()

    assert os.environ["IMAGEM_MODELO"] == "SG161222/RealVisXL_V5.0"
