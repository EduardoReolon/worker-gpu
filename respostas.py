"""As respostas que mais de uma rota precisa dar igual.

Duplicar o corpo de um 503 em tres rotas garante que uma delas envelhece — e
o cliente do outro lado le `error.code` para decidir o que fazer.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse

from arbitro import GpuOcupada


def ocupada(erro: GpuOcupada) -> JSONResponse:
    """503 com `Retry-After` CALCULADO a partir do que esta rodando.

    Uma constante mandaria o cliente voltar em 60s quando falta meio segundo,
    ou em 60s quando falta um minuto e meio — desperdicio num caso, um 503
    extra garantido no outro.
    """
    corpo = {
        "code": "gpu_ocupada",
        "message": str(erro),
        "ocupante": erro.ocupante,
    }
    # So quando se sabe. `"modelo": null` diria "nenhum modelo", que e outra
    # coisa — o cliente leria isso como "a placa esta limpa".
    if erro.modelo:
        corpo["modelo"] = erro.modelo

    return JSONResponse({"error": corpo}, status_code=503, headers={"Retry-After": str(erro.falta)})


def indisponivel(codigo: str, mensagem: str, *, retry_after: int | None = 60) -> JSONResponse:
    """503 para o que e transitorio e nao e disputa de GPU: o Ollama fora do
    ar, a VRAM que nao coube, o tempo que estourou.

    `retry_after=None` OMITE o cabecalho, e existe para um caso so: o
    `timeout`. Para esse codigo a documentacao manda "nao repita igual, reduza
    o pedido" — mandar junto um `Retry-After` seria o contrato se contradizendo
    dentro da mesma resposta, e um cliente que obedece cabecalho antes de ler
    corpo obedeceria o errado. Cabecalho ausente e mais dificil de seguir por
    engano do que cabecalho presente que a doc pede para ignorar.
    """
    cabecalhos = {} if retry_after is None else {"Retry-After": str(retry_after)}

    return JSONResponse(
        {"error": {"code": codigo, "message": mensagem}},
        status_code=503,
        headers=cabecalhos,
    )


def falha_do_worker(codigo: str, mensagem: str) -> JSONResponse:
    """500 para quando o WORKER quebrou, e nao o pedido nem a hora.

    E o oposto do 503: nao ha o que esperar. O cliente marca o trabalho como
    falho e avisa quem cuida da maquina — repetir so repete a falha. Sem
    `Retry-After`, pelo mesmo motivo do `timeout`.
    """
    return JSONResponse({"error": {"code": codigo, "message": mensagem}}, status_code=500)
