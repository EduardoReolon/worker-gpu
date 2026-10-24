"""O proxy do Ollama: como cada falha e classificada, e o cliente compartilhado.

Este arquivo existe porque a classificacao de erro AQUI decide o que o cliente
do outro lado faz com o trabalho dele. Um erro permanente vestido de
transitorio vira reagendamento eterno; um transitorio vestido de permanente
faz o cliente jogar fora um trabalho que ia dar certo. Nos dois casos o
sintoma aparece longe daqui, dias depois, na fila de outro repositorio.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

import httpx
import pytest


@pytest.fixture
def ollama_mod(worker):
    """O modulo recarregado com o ambiente dos testes."""
    import ollama

    return ollama


def _falhar_com(erro: Exception):
    def post(*a, **k):
        raise erro

    return post


# ---------------------------------------------------------------------------
# Classificacao das falhas de transporte
# ---------------------------------------------------------------------------
def test_o_readtimeout_vira_demorou_demais(ollama_mod, monkeypatch):
    """E o unico caso em que repetir IGUAL nao adianta: o trabalho nao cabe no
    orcamento. O cliente precisa reduzir o pedido, nao reagenda-lo."""
    monkeypatch.setattr(
        ollama_mod._obter_cliente(), "post", _falhar_com(httpx.ReadTimeout("nada veio em 540s"))
    )

    with pytest.raises(ollama_mod.OllamaDemorouDemais):
        ollama_mod.conversar({"messages": []}, {})


@pytest.mark.parametrize(
    "erro",
    [
        httpx.ConnectTimeout("nao atendeu"),
        httpx.ConnectError("connection refused"),
        httpx.PoolTimeout("sem conexao livre no pool"),
        httpx.WriteTimeout("nao terminei de escrever"),
        httpx.RemoteProtocolError("servidor desconectou"),
    ],
)
def test_os_outros_timeouts_continuam_indisponivel(ollama_mod, monkeypatch, erro):
    """`ReadTimeout` e o unico. Capturar o pai `httpx.TimeoutException` pegaria
    tambem estes — todos transitorios, todos resolvem sozinhos — e faria o
    cliente DESISTIR de um trabalho que ia dar certo.

    `PoolTimeout` merece o destaque: ele so passou a ser possivel quando o
    cliente httpx virou um so para o processo. Antes, cada chamada tinha pool
    proprio e ele nao podia acontecer.
    """
    monkeypatch.setattr(ollama_mod._obter_cliente(), "post", _falhar_com(erro))

    with pytest.raises(ollama_mod.OllamaIndisponivel):
        ollama_mod.conversar({"messages": []}, {})


def test_as_duas_excecoes_sao_irmas(ollama_mod):
    """E nao mae e filha. Com heranca, um `except OllamaIndisponivel` posto
    antes engoliria o timeout em silencio, e a ordem dos `except` em
    `texto.py` passaria a decidir o `error.code` que o cliente recebe."""
    assert not issubclass(ollama_mod.OllamaDemorouDemais, ollama_mod.OllamaIndisponivel)
    assert not issubclass(ollama_mod.OllamaIndisponivel, ollama_mod.OllamaDemorouDemais)


def test_erro_de_programacao_vira_503_mas_deixa_traceback(ollama_mod, monkeypatch, caplog):
    """O `except Exception` captura tambem bug do worker — `KeyError`,
    `AttributeError`, um `None` onde nao devia. Ele vira 503 para o cliente
    nao ficar sem resposta, mas SEM o traceback o defeito some no journal
    vestido de "o Ollama nao responde", e o cliente reagenda por horas
    enquanto o operador procura no lugar errado.
    """
    monkeypatch.setattr(
        ollama_mod._obter_cliente(), "post", _falhar_com(AttributeError("bug daqui"))
    )

    with caplog.at_level(logging.ERROR, logger="worker-gpu.ollama"):
        with pytest.raises(ollama_mod.OllamaIndisponivel):
            ollama_mod.conversar({"messages": []}, {})

    assert any(registro.exc_info for registro in caplog.records), (
        "o traceback precisa ir para o journal; sem ele o bug fica invisivel"
    )


def test_o_ollama_fora_do_ar_nao_polui_o_journal(ollama_mod, monkeypatch, caplog):
    """O contrario do teste acima: o Ollama desligado e rotina, nao defeito.
    Um traceback por pedido recusado enterraria os erros de verdade."""
    monkeypatch.setattr(
        ollama_mod._obter_cliente(), "post", _falhar_com(httpx.ConnectError("connection refused"))
    )

    with caplog.at_level(logging.ERROR, logger="worker-gpu.ollama"):
        with pytest.raises(ollama_mod.OllamaIndisponivel):
            ollama_mod.conversar({"messages": []}, {})

    assert not any(registro.exc_info for registro in caplog.records)


# ---------------------------------------------------------------------------
# O corpo repassado
# ---------------------------------------------------------------------------
def test_o_corpo_vai_intacto_menos_o_stream(ollama_mod, monkeypatch):
    visto = {}

    def post(url, **kwargs):
        visto["url"] = url
        visto["json"] = kwargs["json"]
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(ollama_mod._obter_cliente(), "post", post)

    ollama_mod.conversar(
        {"model": "m", "messages": [{"role": "user", "content": "oi"}], "max_tokens": 8}, {}
    )

    assert visto["url"] == "/v1/chat/completions"
    assert visto["json"]["max_tokens"] == 8
    assert visto["json"]["stream"] is False


def test_nenhum_keep_alive_e_injetado(ollama_mod, monkeypatch):
    """Havia uma injecao de `keep_alive` vinda do `.env`. Ela foi removida
    porque `keep_alive` nao e campo do dialeto da OpenAI: a camada compativel
    do Ollama descartava a chave, e a variavel parecia ativa sem ter efeito
    nenhum. Configuracao que mente e pior que configuracao ausente."""
    visto = {}
    monkeypatch.setattr(
        ollama_mod._obter_cliente(),
        "post",
        lambda url, **k: visto.update(k["json"]) or httpx.Response(200, json={}),
    )

    ollama_mod.conversar({"messages": []}, {})

    assert "keep_alive" not in visto


def test_uma_resposta_que_nao_e_json_nao_levanta(ollama_mod, monkeypatch):
    monkeypatch.setattr(
        ollama_mod._obter_cliente(),
        "post",
        lambda url, **k: httpx.Response(502, text="<html>proxy</html>"),
    )

    status, dados = ollama_mod.conversar({"messages": []}, {})

    assert status == 502
    assert "proxy" in dados["error"]["message"]


# ---------------------------------------------------------------------------
# `carregados` continua `list[str]` — e por que isso importa
# ---------------------------------------------------------------------------
def test_carregados_continua_sendo_lista_de_nomes(ollama_mod, monkeypatch):
    """`ollama.carregados` esta publicado como lista de nomes no
    `INTEGRACAO.md` e no `README.md`. Enriquecer ESTE retorno mudaria o
    significado de um campo existente, e um cliente que itera strings
    quebraria em silencio."""
    monkeypatch.setattr(
        ollama_mod,
        "modelos_carregados_detalhe",
        lambda: [{"name": "qwen2.5:7b-instruct", "context_length": 4096}],
    )

    assert ollama_mod.modelos_carregados() == ["qwen2.5:7b-instruct"]


def test_descarregar_tudo_sobrevive_ao_detalhe(ollama_mod, monkeypatch):
    """O `", ".join(carregados)` de `descarregar_tudo()` roda FORA de try e
    DENTRO do lock da GPU. Com dicts no lugar de strings ele levanta
    `TypeError`, a difusao nunca roda, e a rota de imagem devolve 500 com a
    placa ja descarregada — livre, e ninguem usando."""
    monkeypatch.setattr(
        ollama_mod,
        "modelos_carregados_detalhe",
        lambda: [{"name": "qwen2.5:7b-instruct", "size_vram": 5_400_000_000}],
    )
    pedidos = []
    monkeypatch.setattr(
        ollama_mod._obter_cliente(),
        "post",
        lambda url, **k: pedidos.append(k["json"]) or httpx.Response(200, json={}),
    )

    assert ollama_mod.descarregar_tudo() == ["qwen2.5:7b-instruct"]
    assert pedidos == [{"model": "qwen2.5:7b-instruct", "keep_alive": 0}]


def test_um_api_ps_que_nao_e_json_devolve_lista_vazia(ollama_mod, monkeypatch):
    """`resposta.json()` levanta `ValueError`, que NAO e `httpx.HTTPError` —
    era por aqui que o `/health/` devolvia 500."""
    monkeypatch.setattr(
        ollama_mod._obter_cliente(),
        "get",
        lambda url, **k: httpx.Response(200, text="nao sou json"),
    )

    assert ollama_mod.modelos_carregados_detalhe() == []
    assert ollama_mod.esta_de_pe() is True


# ---------------------------------------------------------------------------
# O cliente compartilhado e o cancelamento no Ollama
# ---------------------------------------------------------------------------
class _ServidorMudo:
    """Aceita a conexao, le o pedido e nunca responde.

    E o Ollama gerando: do ponto de vista do socket, um modelo que demora e um
    servidor que aceitou e ainda nao falou.
    """

    def __init__(self) -> None:
        self._escuta = socket.socket()
        self._escuta.bind(("127.0.0.1", 0))
        self._escuta.listen(1)
        self.porta = self._escuta.getsockname()[1]
        self.fechou = threading.Event()
        self.instante_do_fechamento = 0.0
        threading.Thread(target=self._servir, daemon=True).start()

    def _servir(self) -> None:
        try:
            conexao, _ = self._escuta.accept()
        except OSError:
            return
        with conexao:
            conexao.settimeout(15.0)
            try:
                conexao.recv(65536)
                # `recv` devolvendo b"" e o FIN do cliente: ele fechou.
                while conexao.recv(65536):
                    pass
            except OSError:
                pass
            self.instante_do_fechamento = time.monotonic()
            self.fechou.set()

    def fechar(self) -> None:
        self._escuta.close()


def test_o_readtimeout_fecha_a_conexao_mesmo_com_cliente_compartilhado():
    """A guarda do cliente unico, e o motivo dela ser um teste e nao um
    comentario.

    Num `ReadTimeout` o arbitro solta o lock, mas o Ollama nao foi avisado e
    continua gerando. O que o avisa e a conexao fechando: ele cancela a
    geracao ao ver o cliente sumir. Enquanto cada chamada tinha o seu
    `with httpx.Client(...)`, isso acontecia de graca na saida do bloco.

    Com um cliente compartilhado, passa a depender de o pool DESCARTAR a
    conexao em vez de devolve-la. Medido em httpx 0.28.1 / httpcore 1.0.9:
    descarta, e o servidor ve o fechamento no instante do timeout. Um
    `pip install -U httpx` que mude isso nao daria erro nenhum — o sintoma
    seria "a placa fica ocupada depois do timeout", meses depois, sem ninguem
    ligar uma coisa a outra.
    """
    servidor = _ServidorMudo()
    cliente = httpx.Client(base_url=f"http://127.0.0.1:{servidor.porta}", timeout=5.0)
    try:
        comeco = time.monotonic()
        with pytest.raises(httpx.ReadTimeout):
            cliente.post("/v1/chat/completions", json={"messages": []}, timeout=1.0)

        assert servidor.fechou.wait(5.0), "o servidor nunca viu a conexao fechar"
        assert servidor.instante_do_fechamento - comeco < 3.0, (
            "a conexao ficou pendurada depois do timeout; o Ollama nao seria avisado"
        )
        assert _conexoes_no_pool(cliente) in (0, None)
    finally:
        cliente.close()
        servidor.fechar()


def _conexoes_no_pool(cliente: httpx.Client):
    """Quantas conexoes o pool guardou. `None` se o httpx mudou de estrutura
    interna — a asercao de comportamento acima e que e o contrato."""
    try:
        return len(cliente._transport._pool.connections)
    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# A construcao do cliente pode falhar, e nao pode derrubar o processo
# ---------------------------------------------------------------------------
@pytest.fixture
def cliente_que_nao_constroi(ollama_mod, monkeypatch):
    """Reproduz uma falha REAL de producao.

    `httpx.Client()` monta um contexto SSL na construcao, e numa maquina o
    `create_ssl_context` levantou `FileNotFoundError` procurando o pacote de
    certificados. Nem `FileNotFoundError` nem os parentes dele estao em
    `httpx.HTTPError`, e o `/health/` passou a responder 500.
    """
    monkeypatch.setattr(ollama_mod, "_cliente", None)

    def nao_constroi(*a, **k):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(ollama_mod.httpx, "Client", nao_constroi)


def test_sem_cliente_as_leituras_degradam_em_vez_de_levantar(ollama_mod, cliente_que_nao_constroi):
    assert ollama_mod.esta_de_pe() is False
    assert ollama_mod.modelos_carregados_detalhe() == []
    assert ollama_mod.modelos_carregados() == []
    assert ollama_mod.modelos_no_disco() == []
    assert ollama_mod.descarregar_tudo() == []


def test_sem_cliente_a_rota_de_texto_devolve_503(ollama_mod, cliente_que_nao_constroi):
    """E nao 500, e nao um processo morto: e transitorio do ponto de vista do
    cliente, e o codigo diz qual parte esta doente."""
    with pytest.raises(ollama_mod.OllamaIndisponivel):
        ollama_mod.conversar({"messages": []}, {})


def test_sem_cliente_o_health_continua_respondendo(worker, cliente_que_nao_constroi):
    """A razao de o cliente ser construido PREGUICOSAMENTE e nao na importacao.

    Na importacao, esta mesma falha viraria "o servico nao sobe" — e levaria
    junto a imagem e a conversao, que nao precisam do Ollama para nada.
    """
    from fastapi.testclient import TestClient

    resposta = TestClient(worker.app).get("/health/")

    assert resposta.status_code == 200
    assert resposta.json()["ollama"]["de_pe"] is False
    assert resposta.json()["rotas"]["imagem"] is True


def test_o_cliente_e_construido_uma_vez_so(ollama_mod, monkeypatch):
    """Construir por chamada era o desperdicio que motivou o cliente unico:
    cada construcao monta um contexto SSL para falar HTTP com o loopback."""
    monkeypatch.setattr(ollama_mod, "_cliente", None)
    construcoes = []

    real = ollama_mod.httpx.Client

    def contando(*a, **k):
        construcoes.append(1)
        return real(*a, **k)

    monkeypatch.setattr(ollama_mod.httpx, "Client", contando)

    primeiro = ollama_mod._obter_cliente()
    segundo = ollama_mod._obter_cliente()

    assert primeiro is segundo
    assert len(construcoes) == 1
