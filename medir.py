"""Mede quanto tempo o Docling leva NESTA maquina, e mostra o que ele produziu.

Existe para uma decisao concreta: CPU ou GPU. A resposta nao e a mesma em toda
maquina, e chutar sai caro nos dois sentidos — comprar placa sem precisar, ou
descobrir depois de meses que cada documento prende o worker por meia hora.

    python medir.py artigo.pdf              # como o servico esta configurado
    python medir.py artigo.pdf --cpu        # forcando CPU
    python medir.py artigo.pdf --cpu --threads 1
    python medir.py artigo.pdf --cuda

Rode na maquina que vai HOSPEDAR o servico. Quem chama nao converte nada —
so faz a requisicao HTTP —, entao medir do lado do cliente nao responde nada.

Sem `--threads`, usa todos os nucleos — e e assim que o servico vai rodar.
`--threads 1` serve para outra pergunta: quanto disso e paralelismo, ou como
seria numa maquina de um nucleo so. Nao confunda um com o outro na hora de
decidir.

Ele converte o mesmo arquivo DUAS vezes de proposito. A primeira carrega os
modelos junto; a segunda e o que o servico realmente paga por documento, porque
ele fica de pe entre uma conversao e outra. Olhar so a primeira superestima o
custo de cada PDF — e pode fazer comprar uma placa que nao era necessaria.

Primeira execucao baixa os modelos de layout do HuggingFace (algumas centenas
de MB). Numa rede que bloqueie `huggingface.co` isto falha, e a mensagem fala
de proxy, nao de Docling.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede a conversao do Docling nesta maquina.")
    parser.add_argument("pdf", help="Caminho do PDF a converter.")
    parser.add_argument("--cpu", action="store_true", help="Forca CPU.")
    parser.add_argument("--cuda", action="store_true", help="Forca a placa.")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Threads na CPU. 1 estima uma maquina de um nucleo so. 0 deixa o Docling decidir.",
    )
    parser.add_argument("--ocr", action="store_true", help="Liga o OCR (PDF digitalizado).")
    parser.add_argument(
        "--salvar", help="Grava o Markdown neste arquivo, para comparar com outra configuracao."
    )
    args = parser.parse_args()

    caminho = Path(args.pdf)
    if not caminho.is_file():
        print(f"ERRO: {caminho} nao existe.", file=sys.stderr)
        return 1

    if args.cpu and args.cuda:
        print("ERRO: escolha --cpu ou --cuda, nao os dois.", file=sys.stderr)
        return 1

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    if args.cpu:
        dispositivo = AcceleratorDevice.CPU
    elif args.cuda:
        dispositivo = AcceleratorDevice.CUDA
    else:
        dispositivo = AcceleratorDevice.AUTO

    acelerador = AcceleratorOptions(device=dispositivo)
    if args.threads:
        acelerador.num_threads = args.threads

    opcoes = PdfPipelineOptions()
    opcoes.accelerator_options = acelerador
    opcoes.do_ocr = args.ocr
    opcoes.do_table_structure = True

    tamanho = caminho.stat().st_size

    conversor = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opcoes)}
    )

    # Converte DUAS vezes, e a repeticao nao e por estatistica.
    #
    # Construir o `DocumentConverter` nao carrega modelo nenhum: o Docling monta
    # o pipeline dentro do PRIMEIRO `convert()` e o guarda em cache
    # (`document_converter.py::_get_pipeline`). Cronometrar so a construcao
    # devolve zero e sugere que a carga e gratis — quando na verdade ela esta
    # inteira dentro da primeira conversao.
    #
    # Isso importa para decidir: o servico fica de pe, entao o que ele paga por
    # documento e a SEGUNDA medida, nao a primeira. Confundir as duas
    # superestima o custo de cada PDF e pode comprar uma placa por engano.
    inicio = time.perf_counter()
    resultado = conversor.convert(str(caminho))
    primeira = time.perf_counter() - inicio

    inicio = time.perf_counter()
    resultado = conversor.convert(str(caminho))
    seguintes = time.perf_counter() - inicio

    markdown = resultado.document.export_to_markdown()
    paginas = len(resultado.document.pages) or 1

    print()
    print(f"Arquivo:     {caminho.name}  ({tamanho / 1024:.0f} KB, {paginas} pagina(s))")
    print(f"Dispositivo: {dispositivo.value}  threads={args.threads or 'auto'}  ocr={args.ocr}")
    print(f"1a conversao:  {primeira:6.1f}s   (inclui carregar os modelos)")
    print(
        f"As seguintes:  {seguintes:6.1f}s   "
        f"({seguintes / paginas:.1f}s por pagina)  <- e este que decide"
    )
    print(f"Carga dos modelos: ~{max(primeira - seguintes, 0):.1f}s, uma vez por processo")
    print()
    print("O servico fica de pe entre conversoes, entao ele paga a primeira linha")
    print("so depois de subir. Por documento, paga a segunda.")
    print()
    print("A pergunta nao e se o tempo e 'rapido': o worker converte UM por vez e")
    print("recusa o resto com 503, para o cliente adiar. A pergunta e se cabe no")
    print("seu ritmo de envio de documentos.")
    print()

    # Sinais de que a analise de layout funcionou. Sao o que distingue este
    # caminho do extrator local, e olhar o tempo sem olhar isto seria medir a
    # coisa errada.
    cabecalhos = [linha for linha in markdown.splitlines() if linha.startswith("#")]
    tabelas = markdown.count("\n|")
    print(f"Cabecalhos de secao reconhecidos: {len(cabecalhos)}")
    print(f"Linhas de tabela em Markdown:     {tabelas}")
    if not cabecalhos:
        print("  AVISO: nenhum cabecalho. Confira se o PDF tem camada de texto;")
        print("  se for digitalizado, rode de novo com --ocr.")
    print()

    if args.salvar:
        Path(args.salvar).write_text(markdown, encoding="utf-8")
        print(f"Markdown gravado em {args.salvar}")
    else:
        print("--- Markdown (primeiras 60 linhas) ---")
        print("\n".join(markdown.splitlines()[:60]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
