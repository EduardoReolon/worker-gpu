"""Rota de animacao: imagem para video, no dialeto `/v1/videos` da OpenAI.

    POST /v1/videos               multipart: prompt, model, seconds, size,
                                  input_reference (a imagem)  -> o trabalho
    GET  /v1/videos/{id}          -> status: queued | in_progress | completed | failed
    GET  /v1/videos/{id}/content  -> o MP4

O dialeto da Sora, e nao um proprio, pelo mesmo motivo das outras rotas: o
cliente (PubliBot) fala igual com esta placa e com a API paga, trocando so o
endereco.

## Assincrono, sem virar fila

Animar leva dezenas de minutos (medido numa placa de 8 GB: ~9,5 min por trecho
de ~1,1 s). Segurar a conexao esse tempo todo nao e opcao. Entao:

- o `POST` toma a placa no arbitro NA HORA. Ocupada: 503 com `Retry-After`,
  como em qualquer rota — o cliente tenta de novo, com a fila DELE;
- livre: o lock passa para uma thread que gera, e o `POST` responde ja com o
  `id`. O `threading.Lock` do arbitro pode ser solto por outra thread, e e o
  que acontece no fim da geracao;
- os trabalhos ficam em memoria. Um restart os perde: o `GET` de um id
  desconhecido da 404, e o cliente pede de novo. Nao contraria o "nada de
  fila aqui": so entra trabalho com a placa tomada, um por vez.

## O FramePack roda em outro processo

O FramePack (lllyasviel/FramePack) pede torch 2.6 e Python 3.10, que nao
convivem com as versoes deste worker. Ele fica instalado a parte, no ambiente
dele (`FRAMEPACK_DIR`), e o worker chama `animar_framepack.py` com o Python
daquele ambiente (`ANIMACAO_PYTHON`). Antes, com o lock na mao, solta tudo o
que ocupa a placa: o Ollama e os modelos residentes daqui (imagem,
transcricao, vetorizacao) — o FramePack usa ~7,7 GB de 8.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

import ollama
import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    ANIMACAO_GUARDAR_SEGUNDOS,
    ANIMACAO_MEMORIA_PRESERVADA,
    ANIMACAO_PASSOS,
    ANIMACAO_PYTHON,
    ANIMACAO_RESOLUCAO,
    ANIMACAO_SEGUNDOS_MAXIMOS,
    ANIMACAO_TEACACHE,
    ANIMACAO_TEMPO_MAXIMO,
    FRAMEPACK_DIR,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.animacao")

router = APIRouter()

MODELO = "framepack"
SCRIPT = Path(__file__).resolve().parent / "animar_framepack.py"
IMAGEM_MAXIMA = 20 * 1024 * 1024


@dataclass
class Trabalho:
    id: str
    prompt: str
    segundos: int
    tamanho: str
    pasta: Path
    semente: int = 0
    # O tamanho em que o FramePack vai gerar, (largura, altura), ou None para
    # o da imagem na `ANIMACAO_RESOLUCAO`.
    medidas: tuple[int, int] | None = None
    status: str = "queued"
    progresso: int = 0
    criado: int = field(default_factory=lambda: int(time.time()))
    concluido: int | None = None
    erro: dict | None = None
    duracao: float = 0.0

    @property
    def video(self) -> Path:
        return self.pasta / "video.mp4"

    def como_json(self) -> dict:
        corpo = {
            "id": self.id,
            "object": "video",
            "status": self.status,
            "model": MODELO,
            "progress": self.progresso,
            "seconds": str(self.segundos),
            "size": self.tamanho,
            "seed": self.semente,
            "created_at": self.criado,
        }
        if self.concluido:
            corpo["completed_at"] = self.concluido
        if self.erro:
            corpo["error"] = self.erro
        return corpo


_trabalhos: dict[str, Trabalho] = {}
_mural = threading.Lock()
_ultimo: dict = {}


def conferir_configuracao() -> None:
    """Recusa subir com a animacao ligada e o FramePack ausente: sem isto, o
    erro so apareceria no primeiro pedido, dias depois."""
    raiz = Path(FRAMEPACK_DIR)
    if not (raiz / "diffusers_helper").is_dir():
        raise RuntimeError(
            f"ANIMACAO_ATIVA esta ligado, mas o FramePack nao esta em {raiz} "
            "(FRAMEPACK_DIR). Veja INTEGRACAO.md, 'Animacao'; ou ANIMACAO_ATIVA=nao."
        )
    if not os.access(ANIMACAO_PYTHON, os.X_OK):
        raise RuntimeError(
            f"ANIMACAO_PYTHON={ANIMACAO_PYTHON} nao existe ou nao executa: e o Python do "
            "ambiente do FramePack (venv dele)."
        )


def _soltar_a_placa() -> None:
    """Com o lock na mao: nada do worker nem do Ollama fica na VRAM."""
    import importlib

    ollama.descarregar_tudo()
    for nome in ("imagem", "transcricao", "vetorizacao"):
        try:
            importlib.import_module(nome).descarregar()
        except Exception as exc:  # um modulo que nao descarrega nao impede tentar
            logger.warning("Nao descarreguei %s antes de animar: %s", nome, exc)


def _comando(trabalho: Trabalho, imagem: Path) -> list[str]:
    comando = [
        ANIMACAO_PYTHON, str(SCRIPT),
        "--framepack-dir", FRAMEPACK_DIR,
        "--imagem", str(imagem),
        "--prompt", trabalho.prompt,
        "--segundos", str(trabalho.segundos),
        "--saida", str(trabalho.video),
        "--resolucao", str(ANIMACAO_RESOLUCAO),
        "--passos", str(ANIMACAO_PASSOS),
        "--memoria-preservada", str(ANIMACAO_MEMORIA_PRESERVADA),
    ]  # fmt: skip
    comando += ["--semente", str(trabalho.semente)]
    if trabalho.medidas:
        comando += ["--largura", str(trabalho.medidas[0]), "--altura", str(trabalho.medidas[1])]
    if ANIMACAO_TEACACHE:
        comando.append("--teacache")
    return comando


def _gerar(trabalho: Trabalho, imagem: Path, posse) -> None:
    """Roda na thread, com o lock que o `POST` tomou; solta no fim."""
    inicio = time.monotonic()
    saida_do_processo: list[str] = []
    try:
        trabalho.status = "in_progress"
        _soltar_a_placa()
        processo = subprocess.Popen(  # noqa: S603 - comando montado aqui, sem shell
            _comando(trabalho, imagem),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        relogio = threading.Timer(ANIMACAO_TEMPO_MAXIMO, processo.kill)
        relogio.daemon = True
        relogio.start()
        for linha in processo.stdout:
            linha = linha.rstrip()
            if linha.startswith("PROGRESSO "):
                novo = min(99, int(float(linha.split()[1])))
                # No journal a cada 10%: e o que diz, de fora e sem o id, que
                # o trabalho anda — e nao que a placa ficou presa.
                if novo // 10 > trabalho.progresso // 10:
                    logger.info(
                        "Animacao %s: %s%% em %.0f min.",
                        trabalho.id,
                        novo,
                        (time.monotonic() - inicio) / 60,
                    )
                trabalho.progresso = novo
            else:
                saida_do_processo.append(linha)
                del saida_do_processo[:-40]
        processo.wait()
        relogio.cancel()
        trabalho.duracao = time.monotonic() - inicio
        if processo.returncode == 0 and trabalho.video.exists():
            trabalho.status, trabalho.progresso = "completed", 100
            logger.info("Animacao %s pronta em %.0fs.", trabalho.id, trabalho.duracao)
        else:
            trabalho.status = "failed"
            trabalho.erro = _erro(processo.returncode, saida_do_processo, trabalho.duracao)
            logger.warning("Animacao %s falhou: %s", trabalho.id, trabalho.erro["message"])
    except Exception as exc:
        trabalho.status = "failed"
        trabalho.erro = {"code": "falha_na_animacao", "message": f"{type(exc).__name__}: {exc}"}
        logger.exception("Animacao %s quebrou.", trabalho.id)
    finally:
        trabalho.concluido = int(time.time())
        _ultimo.update(segundos=trabalho.segundos, tempo=round(trabalho.duracao))
        posse.__exit__(None, None, None)


def _erro(codigo: int, linhas: list[str], duracao: float) -> dict:
    texto = "\n".join(linhas[-12:])
    if duracao >= ANIMACAO_TEMPO_MAXIMO - 1:
        return {
            "code": "timeout",
            "message": f"passou de {ANIMACAO_TEMPO_MAXIMO}s (ANIMACAO_TEMPO_MAXIMO). Peca menos "
            "segundos ou baixe ANIMACAO_RESOLUCAO.",
        }
    if "out of memory" in texto.lower() or "outofmemory" in texto.lower():
        return {
            "code": "sem_vram",
            "message": "faltou VRAM. Suba ANIMACAO_MEMORIA_PRESERVADA ou baixe a resolucao.",
        }
    return {"code": "falha_na_animacao", "message": f"saiu com {codigo}: {texto[-800:]}"}


def _limpar() -> None:
    """Esquece os trabalhos prontos ha mais de ANIMACAO_GUARDAR_SEGUNDOS."""
    agora = time.time()
    with _mural:
        velhos = [
            t
            for t in _trabalhos.values()
            if t.concluido and agora - t.concluido > ANIMACAO_GUARDAR_SEGUNDOS
        ]
        for trabalho in velhos:
            _trabalhos.pop(trabalho.id, None)
            shutil.rmtree(trabalho.pasta, ignore_errors=True)


def medidas_do_pedido(size: str | None) -> tuple[int, int] | None:
    """`"448x800"` -> (448, 800), em multiplos de 32 e sem passar da area da
    `ANIMACAO_RESOLUCAO`. None: o pedido nao disse, vale o da imagem.

    O tempo do FramePack cresce com os pixels. Gerar maior que o pedido e
    pagar por pixels que o cliente vai jogar fora ao reescalar; por isso o
    tamanho pedido e respeitado, e so e REDUZIDO (na mesma proporcao) se
    passar do teto da maquina — nunca aumentado.
    """
    if not size:
        return None
    try:
        largura, _, altura = size.lower().partition("x")
        largura, altura = int(largura), int(altura)
    except ValueError as exc:
        raise ValueError(f"size={size!r}: use LARGURAxALTURA, como 448x800.") from exc
    if largura <= 0 or altura <= 0:
        raise ValueError(f"size={size!r}: os dois lados precisam ser positivos.")

    teto = ANIMACAO_RESOLUCAO * ANIMACAO_RESOLUCAO
    if largura * altura > teto:
        escala = (teto / (largura * altura)) ** 0.5
        largura, altura = largura * escala, altura * escala

    # Para baixo, em multiplos de 32 (o que o VAE do HunyuanVideo exige).
    return max(256, int(largura) // 32 * 32), max(256, int(altura) // 32 * 32)


def _invalido(mensagem: str, status: int = 422) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": "pedido_invalido", "message": mensagem}}, status_code=status
    )


@router.post("/v1/videos", dependencies=[Depends(conferir)])
def criar(
    prompt: str = Form(...),
    input_reference: UploadFile | None = None,
    model: str | None = Form(default=None),
    seconds: str | None = Form(default=None),
    size: str | None = Form(default=None),
    seed: str | None = Form(default=None),
):
    """`model` e aceito e ignorado (aqui e sempre o FramePack).

    `size` e respeitado: o FramePack gera nele (em multiplos de 32), so
    reduzindo se passar do teto da `ANIMACAO_RESOLUCAO`; a imagem e cortada ao
    centro para a proporcao pedida. `seed` fixa o resultado: a mesma imagem,
    o mesmo `prompt` e a mesma `seed` dao o mesmo clipe. Sem ela, sorteia — e
    a sorteada volta no trabalho, para o cliente poder repetir.
    """
    _limpar()
    if not prompt.strip():
        return _invalido("prompt vazio.")
    if input_reference is None:
        return _invalido("falta input_reference: a imagem a animar.")
    conteudo = input_reference.file.read(IMAGEM_MAXIMA + 1)
    if not conteudo:
        return _invalido("imagem vazia.")
    if len(conteudo) > IMAGEM_MAXIMA:
        return _invalido("imagem maior que 20 MB.", 413)
    try:
        segundos = int(float(seconds or 4))
    except ValueError:
        return _invalido(f"seconds={seconds!r}.")
    segundos = max(1, min(segundos, ANIMACAO_SEGUNDOS_MAXIMOS))
    try:
        semente = int(seed) if seed not in (None, "") else secrets.randbelow(2**31)
    except ValueError:
        return _invalido(f"seed={seed!r} nao e um inteiro.")
    try:
        medidas = medidas_do_pedido(size)
    except ValueError as exc:
        return _invalido(str(exc))

    posse = ARBITRO.usar("animacao", modelo=MODELO)
    try:
        posse.__enter__()
    except GpuOcupada as erro:
        return respostas.ocupada(erro)

    pasta = Path(tempfile.mkdtemp(prefix="animacao-"))
    imagem = pasta / "entrada.png"
    imagem.write_bytes(conteudo)
    trabalho = Trabalho(
        id=f"video_{uuid.uuid4().hex[:16]}",
        prompt=prompt.strip()[:1000],
        segundos=segundos,
        tamanho=size or "",
        pasta=pasta,
        semente=semente,
        medidas=medidas,
    )
    with _mural:
        _trabalhos[trabalho.id] = trabalho
    threading.Thread(target=_gerar, args=(trabalho, imagem, posse), daemon=True).start()
    return trabalho.como_json()


def _achar(identificador: str) -> Trabalho | None:
    with _mural:
        return _trabalhos.get(identificador)


def _nao_achado(identificador: str) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "code": "nao_encontrado",
                "message": f"{identificador}: trabalho desconhecido (o worker reiniciou?). "
                "Peca de novo.",
            }
        },
        status_code=404,
    )


@router.get("/v1/videos/{identificador}", dependencies=[Depends(conferir)])
def consultar(identificador: str):
    trabalho = _achar(identificador)
    return trabalho.como_json() if trabalho else _nao_achado(identificador)


@router.get("/v1/videos/{identificador}/content", dependencies=[Depends(conferir)])
def baixar(identificador: str):
    trabalho = _achar(identificador)
    if trabalho is None:
        return _nao_achado(identificador)
    if trabalho.status != "completed":
        return JSONResponse(
            {"error": {"code": "nao_pronto", "message": f"status: {trabalho.status}"}},
            status_code=409,
        )
    return FileResponse(trabalho.video, media_type="video/mp4", filename=f"{trabalho.id}.mp4")


def estado() -> dict:
    """O que o `/health/` mostra sobre esta rota."""
    with _mural:
        lista = list(_trabalhos.values())
    return {
        "motor": MODELO,
        "framepack_dir": FRAMEPACK_DIR,
        "resolucao": ANIMACAO_RESOLUCAO,
        "passos": ANIMACAO_PASSOS,
        "teacache": ANIMACAO_TEACACHE,
        "tempo_maximo": ANIMACAO_TEMPO_MAXIMO,
        "trabalhos": {
            "em_curso": sum(t.status in ("queued", "in_progress") for t in lista),
            "prontos": sum(t.status == "completed" for t in lista),
            "falhos": sum(t.status == "failed" for t in lista),
        },
        "ultimo": dict(_ultimo) or None,
        # O trabalho na placa agora, se houver: o `progress` e ha quanto tempo
        # roda. Sem isto, "esta processando ou travou?" so se respondia com o
        # id que o cliente guardou.
        "em_curso": [
            {
                "id": t.id,
                "status": t.status,
                "progress": t.progresso,
                "seconds": t.segundos,
                "ha_segundos": int(time.time()) - t.criado,
            }
            for t in lista
            if t.status in ("queued", "in_progress")
        ],
    }
