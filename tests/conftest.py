"""Ambiente comum dos testes do worker.

O modulo `config` le o ambiente na IMPORTACAO — e deliberado, porque e o que
faz a unit do systemd valer alguma coisa. O preco e aqui: cada teste que
muda ambiente precisa recarregar os modulos, e e isso que estas fixtures
fazem, uma vez, em vez de cada arquivo inventar o seu jeito.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

SEGREDO = "segredo-de-teste"


@pytest.fixture
def ambiente(monkeypatch):
    """As variaveis que todo teste quer iguais."""
    monkeypatch.setenv("WORKER_SHARED_SECRET", SEGREDO)
    monkeypatch.setenv("IMAGEM_DEVICE", "cpu")
    # Sem temporizador de descarga: um `threading.Timer` vivo depois do teste
    # dispararia no meio de outro.
    monkeypatch.setenv("IMAGEM_OCIOSO_SEGUNDOS", "0")
    monkeypatch.setenv("IMAGEM_MAXIMO", "4")
    monkeypatch.setenv("DOCLING_OCR", "false")
    monkeypatch.setenv("DOCLING_DEVICE", "cpu")
    # O Ollama nao existe nos testes; quem precisa dele substitui.
    monkeypatch.setenv("OLLAMA_DESCARREGAR_PARA_IMAGEM", "nao")
    monkeypatch.setenv("ESPERA_PELO_LOCK", "0")
    # Desligado por padrao: nenhum teste quer baixar gigabytes, e um pedido de
    # modelo ausente viraria 503 `baixando_modelo` em vez do que o teste mede.
    # Quem exercita o download liga de volta.
    monkeypatch.setenv("BAIXAR_MODELO_AUTOMATICO", "nao")


def _recarregar(*nomes: str):
    """Recarrega na ordem das dependencias: `config` primeiro, sempre."""
    modulos = {}
    for nome in ("config", "arbitro", "respostas", "seguranca", *nomes):
        modulos[nome] = importlib.reload(importlib.import_module(nome))
    return modulos


@pytest.fixture
def worker(ambiente):
    """O app inteiro, recarregado com o ambiente do teste."""
    # `modelos` depois de `ollama` e antes de `texto`: ele le `ollama` e e lido
    # por `texto`. Recarregar fora de ordem deixaria `texto` apontando para um
    # `modelos` velho, com o cache de modelos no disco de outro teste dentro.
    modulos = _recarregar("ollama", "modelos", "texto", "imagem", "conversao", "app")
    return modulos["app"]


@pytest.fixture
def segredo() -> str:
    """Por fixture, e nao por import.

    `tests/` nao tem `__init__.py` — e um namespace package. Se este
    repositorio for posto ao lado de outro que tambem tenha um `tests/`
    (foi o caso enquanto ele morava dentro do PubliBot), o Python funde os
    dois, e um `from tests.conftest import ...` aqui pode resolver para o
    conftest do vizinho e explodir na coleta. Pela fixture isso nao acontece.
    """
    return SEGREDO


@pytest.fixture
def cabecalhos():
    return {"Authorization": f"Bearer {SEGREDO}"}
