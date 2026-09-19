"""A credencial, num lugar so.

Estava duplicada nos dois servicos, com uma diferenca sutil: um aceitava
`Authorization: Bearer`, o outro so `X-Worker-Secret`. Duas rotas do mesmo
processo exigindo cabecalhos diferentes e uma armadilha para quem integra.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from config import SEGREDO


def conferir(
    authorization: str | None = Header(default=None),
    x_worker_secret: str | None = Header(default=None),
) -> None:
    """Dependencia do FastAPI: entra na assinatura de toda rota protegida.

    Aceita os dois cabecalhos. O `Bearer` porque e o que os clientes do
    dialeto OpenAI mandam sozinhos; o proprio porque um `curl` de diagnostico
    fica mais curto.

    `compare_digest` e nao `!=`: a comparacao natural e curto-circuitada byte
    a byte, e o tempo de resposta revela quantos bytes iniciais estao certos.
    """
    if not SEGREDO:
        raise HTTPException(500, "WORKER_SHARED_SECRET nao configurado.")

    recebido = x_worker_secret or ""
    if not recebido and authorization and authorization.lower().startswith("bearer "):
        recebido = authorization[7:].strip()

    if not recebido or not hmac.compare_digest(recebido, SEGREDO):
        raise HTTPException(401, "Credencial invalida.")
