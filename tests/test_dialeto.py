"""A traducao entre os dois dialetos, e a escolha do caminho.

Este arquivo guarda um defeito especifico que nao aparece: um pedido com
`num_ctx` respondendo 200 depois de rodar com a janela padrao, porque a camada
compativel do Ollama descartou o `options`. O prompt e cortado pelo COMECO,
entao o que morre primeiro e o prompt de sistema — e a resposta continua
parecendo certa.

Os testes de forma tambem estao aqui, e nao no `test_contrato.py`, por um
motivo: o que a traducao precisa garantir e que a resposta saia
INDISTINGUIVEL da que o caminho antigo devolvia. Quem integra copiou
`contrato/texto-resposta.json` e nao tem como saber qual caminho o pedido
tomou.
"""

from __future__ import annotations

import json
import logging

import pytest

import dialeto

RESPOSTA_NATIVA = {
    "model": "qwen2.5:7b-instruct",
    "created_at": "2026-09-20T12:00:00.000Z",
    "message": {"role": "assistant", "content": "O texto gerado."},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 120,
    "eval_count": 340,
}


# ---------------------------------------------------------------------------
# Qual caminho o pedido toma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "corpo,esperado",
    [
        ({"model": "m", "messages": []}, False),
        ({"model": "m", "messages": [], "temperature": 0.2}, False),
        ({"model": "m", "messages": [], "max_tokens": 500}, False),
        ({"model": "m", "messages": [], "options": {}}, False),
        ({"model": "m", "messages": [], "options": {"num_ctx": 16384}}, True),
        ({"model": "m", "messages": [], "keep_alive": "10m"}, True),
    ],
)
def test_so_vai_pelo_nativo_quem_perderia_algo(corpo, esperado):
    """A escolha e conservadora de proposito: ha dois clientes em producao e um
    deles nao precisa disto. Quem nao manda `options` nem `keep_alive` segue
    pelo caminho antigo e nao paga o risco de uma traducao.

    `options: {}` nao conta — ele nao pede nada, e trocar de caminho por ele
    seria risco por nada.
    """
    assert dialeto.precisa_do_nativo(corpo) is esperado


# ---------------------------------------------------------------------------
# Ida: dialeto da OpenAI -> nativo
# ---------------------------------------------------------------------------
def test_o_num_ctx_chega_ao_nativo():
    """O defeito que este modulo existe para consertar."""
    nativo = dialeto.para_nativo(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "oi"}],
            "options": {"num_ctx": 16384},
        }
    )

    assert nativo["options"]["num_ctx"] == 16384
    assert nativo["stream"] is False
    assert nativo["messages"] == [{"role": "user", "content": "oi"}]


def test_max_tokens_vira_num_predict():
    """A camada compativel ja fazia isso; o caminho nativo nao pode perder."""
    assert dialeto.para_nativo({"max_tokens": 512})["options"]["num_predict"] == 512
    assert dialeto.para_nativo({"max_completion_tokens": 512})["options"]["num_predict"] == 512


def test_o_options_explicito_ganha_do_campo_da_openai():
    """Quem escreveu os dois quis o segundo, ou nao teria escrito o segundo."""
    nativo = dialeto.para_nativo({"max_tokens": 100, "options": {"num_predict": 500}})

    assert nativo["options"]["num_predict"] == 500


def test_os_ajustes_de_geracao_nao_se_perdem_no_caminho_novo():
    nativo = dialeto.para_nativo(
        {
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": 7,
            "stop": ["FIM"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "options": {"num_ctx": 8192},
        }
    )

    assert nativo["options"] == {
        "num_ctx": 8192,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 7,
        "stop": ["FIM"],
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
    }


def test_o_json_schema_vira_format():
    esquema = {"type": "object", "properties": {"nota": {"type": "integer"}}}
    corpo = {
        "options": {"num_ctx": 8192},
        "response_format": {"type": "json_schema", "json_schema": {"name": "n", "schema": esquema}},
    }

    assert dialeto.para_nativo(corpo)["format"] == esquema


def test_o_json_object_vira_format_json():
    corpo = {"options": {"num_ctx": 8192}, "response_format": {"type": "json_object"}}

    assert dialeto.para_nativo(corpo)["format"] == "json"


def test_o_keep_alive_do_cliente_e_honrado():
    """No nativo ele existe de verdade. O que continua fora e a VARIAVEL de
    ambiente do worker, que valeria num caminho e nao no outro."""
    assert dialeto.para_nativo({"keep_alive": "10m"})["keep_alive"] == "10m"


def test_campo_sem_equivalente_vira_aviso_e_nunca_silencio(caplog):
    """A regra desta rota: o defeito que custa caro e o que responde 200 e
    entrega menos do que foi pedido. Se nao da para traduzir, que doa no
    journal."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.dialeto"):
        dialeto.para_nativo({"options": {"num_ctx": 8192}, "inventado_ontem": True})

    assert "inventado_ontem" in caplog.text


def test_um_n_maior_que_um_avisa(caplog):
    """O Ollama gera UMA resposta por pedido. Aparar em silencio devolveria
    menos do que o cliente pediu sem ele nunca saber."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.dialeto"):
        dialeto.para_nativo({"options": {"num_ctx": 8192}, "n": 3})

    assert "`n`=3" in caplog.text


def test_os_campos_aceitos_e_sem_efeito_nao_viram_aviso(caplog):
    """Avisar sobre `stream` a cada pedido enterraria os avisos que importam."""
    with caplog.at_level(logging.WARNING, logger="worker-gpu.dialeto"):
        dialeto.para_nativo(
            {"options": {"num_ctx": 8192}, "stream": False, "user": "x", "tool_choice": "auto"}
        )

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# Volta: nativo -> dialeto da OpenAI
# ---------------------------------------------------------------------------
def test_a_resposta_sai_na_forma_da_openai():
    saida = dialeto.para_openai(RESPOSTA_NATIVA)

    assert saida["object"] == "chat.completion"
    assert saida["id"].startswith("chatcmpl-")
    assert saida["model"] == "qwen2.5:7b-instruct"
    assert saida["choices"][0]["index"] == 0
    assert saida["choices"][0]["message"]["content"] == "O texto gerado."
    assert saida["choices"][0]["finish_reason"] == "stop"


def test_o_usage_carrega_o_detector_de_truncamento():
    """`prompt_tokens` e o sinal mais barato que o cliente tem para desconfiar
    de janela pequena: ele compara com o que mandou. Perde-lo aqui tiraria o
    unico detector de graca do defeito que este modulo veio consertar."""
    saida = dialeto.para_openai(RESPOSTA_NATIVA)

    assert saida["usage"] == {"prompt_tokens": 120, "completion_tokens": 340, "total_tokens": 460}


def test_uma_resposta_sem_contagem_nao_levanta():
    """Versao de Ollama que nao mande as contagens nao pode derrubar a rota."""
    saida = dialeto.para_openai({"message": {"role": "assistant", "content": "x"}})

    assert saida["usage"]["total_tokens"] == 0


@pytest.mark.parametrize(
    "done_reason,esperado",
    [("stop", "stop"), ("length", "length"), ("load", "stop"), (None, "stop")],
)
def test_o_finish_reason_preserva_o_length(done_reason, esperado):
    """`length` e o que denuncia geracao cortada pelo teto — e com
    `json_schema` e a diferenca entre "o modelo errou" e "o JSON veio pela
    metade"."""
    saida = dialeto.para_openai({**RESPOSTA_NATIVA, "done_reason": done_reason})

    assert saida["choices"][0]["finish_reason"] == esperado


def test_a_mensagem_vai_inteira():
    """Campo por campo perderia em silencio o que uma versao nova do Ollama
    acrescentasse ali — foi o caso do `thinking`. Cliente do dialeto da OpenAI
    ignora o que nao conhece; perder o dado nao tem volta."""
    saida = dialeto.para_openai(
        {**RESPOSTA_NATIVA, "message": {"role": "assistant", "content": "x", "thinking": "hmm"}}
    )

    assert saida["choices"][0]["message"]["thinking"] == "hmm"


def test_os_argumentos_da_ferramenta_viram_string():
    """No nativo eles vem como OBJETO; no dialeto da OpenAI, como STRING de
    JSON. Um cliente que faz `json.loads` — o que a doc da OpenAI manda —
    quebraria com o objeto, e quebraria no repositorio dele."""
    saida = dialeto.para_openai(
        {
            **RESPOSTA_NATIVA,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "buscar", "arguments": {"termo": "x"}}}],
            },
        }
    )

    chamada = saida["choices"][0]["message"]["tool_calls"][0]
    assert chamada["type"] == "function"
    assert chamada["id"].startswith("call_")
    assert json.loads(chamada["function"]["arguments"]) == {"termo": "x"}
    assert saida["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize(
    "nativo,esperado",
    [
        ({"error": "model 'x' not found"}, "model 'x' not found"),
        ({"error": {"message": "ja e objeto"}}, "ja e objeto"),
    ],
)
def test_o_erro_sai_sempre_na_mesma_forma(nativo, esperado):
    """O nativo devolve `{"error": "texto"}`, a camada compativel devolve
    `{"error": {"message": "texto"}}`. Sem normalizar, o MESMO erro chegaria ao
    cliente em duas formas conforme um caminho que ele nem escolheu."""
    assert dialeto.erro_para_openai(nativo)["error"]["message"] == esperado
