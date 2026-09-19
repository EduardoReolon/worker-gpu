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


def indisponivel(codigo: str, mensagem: str, *, retry_after: int = 60) -> JSONResponse:
    """503 para o que e transitorio e nao e disputa de GPU: o Ollama fora do
    ar, a VRAM que nao coube, o tempo que estourou."""
    return JSONResponse(
        {"error": {"code": codigo, "message": mensagem}},
        status_code=503,
        headers={"Retry-After": str(retry_after)},
    )
