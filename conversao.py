"""Rota de conversao: `POST /parse/`, arbitrada.

Recebe um PDF (ou DOCX, PPTX, XLSX), devolve Markdown com a estrutura
interpretada — coluna dupla, tabela, cabecalho, rodape, legenda. E o que distingue este
caminho de um extrator de camada de texto, que num artigo de duas colunas
intercala as frases e produz um texto que PARECE correto.

Roda em CPU tambem, e isso importa: a analise de layout nao depende de GPU; a
placa muda o tempo, nao o resultado. Da para comecar sem placa nenhuma e
trocar depois com uma linha no `.env`.
"""

from __future__ import annotations

import gc
import hashlib
import hmac
import importlib.util
import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException, UploadFile

import respostas
from arbitro import ARBITRO, GpuOcupada
from config import (
    CONVERSAO_OCIOSO_SEGUNDOS,
    DOCLING_DEVICE,
    DOCLING_OCR,
    DOCLING_THREADS,
    MAX_PDF_BYTES,
)
from seguranca import conferir

logger = logging.getLogger("worker-gpu.conversao")

router = APIRouter()

# Os formatos aceitos, pela extensao do nome do arquivo — e por ela que o
# Docling escolhe o leitor. PDF passa pela analise de layout (os modelos de
# visao); os do Office declaram a estrutura e sao lidos direto, sem placa, com
# mais fidelidade em tabela e figura que um extrator de texto.
FORMATOS = (".pdf", ".docx", ".pptx", ".xlsx")

_conversor = None
_trava = threading.Lock()
_temporizador: threading.Timer | None = None


def conferir_ocr() -> None:
    """Recusa subir com OCR ligado sem o motor de OCR instalado.

    O Docling pede `rapidocr<4.0.0,>=3.3; python_version < "3.14"`. Em Python
    3.14 o marcador nao casa e o pacote nao entra. Sem esta conferencia o
    servico subiria, responderia `/health/` com `"ocr": true`, e falharia SO
    na primeira conversao de um PDF digitalizado — o caso mais raro, que e
    justamente quando ninguem esta olhando.
    """
    if not DOCLING_OCR or importlib.util.find_spec("rapidocr") is not None:
        return

    raise RuntimeError(
        f"DOCLING_OCR esta ligado, mas o motor de OCR (rapidocr) nao esta "
        f"instalado. Neste Python ({sys.version_info.major}.{sys.version_info.minor}) "
        f'o Docling nao o instala: ele o declara apenas para `python_version < "3.14"`.\n'
        f"Use um Python 3.12 ou 3.13 no venv do worker, ou deixe DOCLING_OCR=false "
        f"— PDF com camada de texto nao precisa de OCR."
    )


def _montar_conversor():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    dispositivos = {
        "cpu": AcceleratorDevice.CPU,
        "cuda": AcceleratorDevice.CUDA,
        "auto": AcceleratorDevice.AUTO,
    }
    if DOCLING_DEVICE not in dispositivos:
        raise RuntimeError(f"DOCLING_DEVICE={DOCLING_DEVICE!r} nao existe. Use cpu, cuda ou auto.")

    acelerador = AcceleratorOptions(device=dispositivos[DOCLING_DEVICE])
    if DOCLING_THREADS:
        acelerador.num_threads = DOCLING_THREADS

    opcoes = PdfPipelineOptions()
    opcoes.accelerator_options = acelerador
    opcoes.do_ocr = DOCLING_OCR
    opcoes.do_table_structure = True

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opcoes)}
    )


def obter_conversor():
    """Carga preguicosa: o modelo leva dezenas de segundos para subir."""
    global _conversor
    if _conversor is None:
        with _trava:
            if _conversor is None:
                logger.info(
                    "Carregando o Docling (dispositivo=%s, ocr=%s, threads=%s)...",
                    DOCLING_DEVICE,
                    DOCLING_OCR,
                    DOCLING_THREADS or "auto",
                )
                _conversor = _montar_conversor()
                logger.info("Docling pronto.")
    return _conversor


def descarregar() -> None:
    """Solta o conversor e devolve a memoria dele.

    O Docling ficava residente para sempre depois da primeira conversao: a
    rota de imagem tinha o seu temporizador de descarga e esta nao tinha nada.
    Sao 1 a 2 GB de memoria ANONIMA parada — e e a anonima parada que o kernel
    escreve no swap quando precisa de paginas, porque ela nao pode ser
    simplesmente descartada como um arquivo mapeado.

    Nao mexe na placa: `DOCLING_DEVICE` pode ser `cpu`, e mesmo em `cuda` quem
    arbitra a VRAM e o `arbitro`. Aqui o que se devolve e RAM.
    """
    global _conversor

    with _trava:
        if _conversor is None:
            return
        logger.info("Descarregando o Docling e devolvendo a memoria dele.")
        _conversor = None

    gc.collect()


def _agendar_descarga() -> None:
    """Marca a hora de soltar o conversor, e adia a marca a cada pedido novo.

    Mais alto que o da imagem de proposito: recarregar o Docling custa dezenas
    de segundos, e PDF costuma vir em lote. Soltar entre dois arquivos do mesmo
    lote seria pagar a carga duas vezes por nada.
    """
    global _temporizador

    if CONVERSAO_OCIOSO_SEGUNDOS <= 0:
        return
    if _temporizador is not None:
        _temporizador.cancel()

    _temporizador = threading.Timer(CONVERSAO_OCIOSO_SEGUNDOS, descarregar)
    # Daemon: um temporizador pendente nao pode segurar o desligamento.
    _temporizador.daemon = True
    _temporizador.start()


@router.post("/parse/", dependencies=[Depends(conferir)])
def parse(
    file: UploadFile,
    x_expected_sha256: str | None = Header(default=None),
):
    """Converte um documento (`FORMATOS`) em Markdown.

    `def` e nao `async def`: a conversao bloqueia por dezenas de segundos, e
    no event loop ela congelaria o processo inteiro.
    """
    # `file.file.read()` e nao `await file.read()`: num handler sincrono nao ha
    # corrotina a esperar. O `UploadFile` guarda o arquivo num
    # `SpooledTemporaryFile`, que e o objeto sincrono por tras.
    # So o NOME, sem pasta: o `filename` vem do cliente, e `pasta / "../x"`
    # escaparia do diretorio temporario.
    nome = Path(file.filename or "").name or "documento.pdf"
    extensao = Path(nome).suffix.lower()
    if extensao not in FORMATOS:
        # 422, e nao o 500 que a conversao daria la dentro: e o pedido que esta
        # errado, e pelo contrato um 500 diz que o worker quebrou.
        raise HTTPException(
            422,
            f"formato {extensao or '(sem extensao)'!r} nao aceito. "
            f"Use um de: {', '.join(FORMATOS)}.",
        )

    conteudo = file.file.read()

    if len(conteudo) > MAX_PDF_BYTES:
        raise HTTPException(413, f"Arquivo excede {MAX_PDF_BYTES} bytes.")

    digest = hashlib.sha256(conteudo).hexdigest()
    if x_expected_sha256 and not hmac.compare_digest(digest, x_expected_sha256):
        # O arquivo chegou corrompido ou nao e o esperado. Converter assim
        # produziria Markdown de um documento que ninguem pediu.
        raise HTTPException(422, "sha256 nao confere com o esperado.")

    inicio = time.perf_counter()
    try:
        with ARBITRO.usar("conversao"):
            with tempfile.TemporaryDirectory() as pasta:
                caminho = Path(pasta) / nome
                caminho.write_bytes(conteudo)
                resultado = obter_conversor().convert(str(caminho))
                markdown = resultado.document.export_to_markdown()
    except GpuOcupada as erro:
        return respostas.ocupada(erro)
    except Exception as exc:
        logger.exception("Falha ao converter %s", file.filename)
        raise HTTPException(500, f"Falha na conversao: {exc}") from exc
    finally:
        # No `finally`: um PDF que falhou deixa o conversor carregado do mesmo
        # jeito, e a memoria dele precisa ser devolvida igual.
        _agendar_descarga()

    duracao = int((time.perf_counter() - inicio) * 1000)
    logger.info("Convertido %s em %sms (%s bytes)", file.filename, duracao, len(conteudo))

    return {
        "markdown": markdown,
        "sha256": digest,
        "bytes": len(conteudo),
        "duration_ms": duracao,
    }


def estado() -> dict:
    """O que o `/health/` do worker mostra sobre esta rota."""
    return {
        "dispositivo": DOCLING_DEVICE,
        "ocr": DOCLING_OCR,
        "threads": DOCLING_THREADS or "auto",
        "carregado": _conversor is not None,
        "ocioso_segundos": CONVERSAO_OCIOSO_SEGUNDOS,
    }
