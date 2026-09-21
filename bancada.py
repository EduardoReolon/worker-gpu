"""Compara configuracoes de geracao de imagem, lado a lado, para voce olhar.

Irmao do `medir_imagem.py`, e a diferenca e o que se pergunta. O `medir_imagem`
responde **quanto custa**; esta bancada responde **qual fica melhor**. Nenhuma
metrica decide isso: quem decide e voce olhando, e por isso a saida daqui e uma
pagina com as imagens lado a lado, e nao uma tabela.

Ela existe para tirar o PubliBot do caminho. Ajustar qualidade pela tela de
revisao do PubliBot e lento e confunde as variaveis: cada rodada passa por
prompt gerado por LLM, fila, worker e navegador, e quando a imagem sai ruim nao
da para saber de quem foi a culpa. Aqui o prompt e fixo, a semente e fixa, e a
unica coisa que muda e o que voce pediu para mudar.

    ./venv/bin/python bancada.py
    ./venv/bin/python bancada.py --vaes '',madebyollin/sdxl-vae-fp16-fix
    ./venv/bin/python bancada.py --schedulers euler,dpm++2m_karras --passos 25,40
    ./venv/bin/python bancada.py --tamanhos 1024x576,1344x768
    ./venv/bin/python bancada.py --modelos SG161222/RealVisXL_V5.0,\
                                  stabilityai/stable-diffusion-xl-base-1.0
    ./venv/bin/python bancada.py --prompts meus-prompts.txt --sementes 3

## Duas regras que fazem a comparacao valer

**Semente fixa, e mais de uma.** Duas sementes por variante, no minimo. Julgar
um modelo por uma imagem e o jeito mais facil de escolher errado: a variacao
entre sementes do MESMO modelo e frequentemente maior que a variacao entre
modelos, e quem olha uma imagem de cada acaba escolhendo a sorte.

**Monta o pipeline por `imagem._montar_pipeline`**, e nao por um caminho
proprio. Uma bancada com a sua propria montagem mediria uma configuracao que o
servico nao usa — foi por isso que o `medir_imagem.py` copiou o
`enable_model_cpu_offload` em vez de usar `.to("cuda")`.

## Ela disputa a placa com o worker

Pare o worker antes:

    systemctl --user stop worker-gpu && ./venv/bin/python bancada.py
    systemctl --user start worker-gpu

Com os dois de pe, os dois modelos tentam caber na mesma VRAM e o que acontece
nao e um erro — e queda para CPU, dezenas de vezes mais lento. A bancada
confere o `/health/` e recusa, a menos que voce passe `--mesmo-com-o-worker`.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# Prompts de teste, em INGLES, e a lingua e deliberada: os codificadores de
# texto do SDXL foram treinados em legendas da web, esmagadoramente inglesas.
# Uma bancada com prompts em portugues mediria todos os modelos pelo pior deles.
#
# Os tres cobrem o que quebra em geracao realista, na ordem em que quebra:
# pele e mao humana, superficie e reflexo, e profundidade de cena.
PROMPTS_PADRAO = [
    "candid editorial photograph of a woman in her 40s working at a wooden desk, "
    "hands visible on a keyboard, soft window light, 50mm, shallow depth of field",
    "product photograph of a ceramic coffee cup on a brushed steel counter, "
    "water droplets, reflections, studio softbox lighting, high detail",
    "wide documentary photograph of a busy city street at dusk, wet asphalt, "
    "shop signs out of focus in the background, natural colors",
]


def _do_env() -> None:
    """Os mesmos valores que a unit usa, para a bancada valer."""
    arquivo = RAIZ / ".env"
    if not arquivo.is_file():
        return
    # Lidas TODAS antes de aplicar, e a ULTIMA ocorrencia de cada chave vence.
    #
    # E o que o systemd faz com `EnvironmentFile`, e precisa ser igual: com
    # `setdefault` linha a linha a PRIMEIRA venceria, e um `.env` com a chave
    # repetida — que e o que acontece quando alguem faz `cat >> .env` — poria
    # esta ferramenta medindo uma configuracao e o servico rodando outra. Sem
    # erro nenhum, e com a ferramenta jurando que mediu o que esta em uso.
    do_arquivo: dict[str, str] = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        do_arquivo[chave.strip()] = valor.strip().strip('"').strip("'")

    # `setdefault` e nao atribuicao: uma variavel de verdade no ambiente manda
    # mais que o arquivo, que e como se passa `IMAGEM_DEVICE=cpu ./bancada.py`.
    for chave, valor in do_arquivo.items():
        os.environ.setdefault(chave, valor)


def _lista(texto: str) -> list[str]:
    """`"a,,b"` -> `["a", "", "b"]`. O vazio e um valor: significa "o do modelo"."""
    return [parte.strip() for parte in texto.split(",")]


def _medidas(texto: str) -> tuple[int, int]:
    largura, _, altura = texto.lower().partition("x")
    return int(largura), int(altura)


def _worker_esta_com_a_placa() -> tuple[bool, str]:
    try:
        import httpx

        endereco = os.environ.get("BIND_HOST", "127.0.0.1")
        porta = os.environ.get("BIND_PORT", "8090")
        corpo = httpx.get(f"http://{endereco}:{porta}/health/", timeout=3.0).json()
    except Exception:
        # Worker fora do ar e exatamente o que queremos: segue.
        return False, ""

    if corpo.get("ocupada"):
        return (
            True,
            f"o worker esta gerando ({corpo.get('ocupante')}) ha {corpo.get('ha_segundos')}s",
        )

    carregados = (corpo.get("imagem") or {}).get("carregado")
    if carregados:
        return True, "o worker tem o modelo de imagem residente na placa"

    return False, "o worker esta de pe, mas com a placa livre"


def _pico_de_vram(torch) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def _rotulo(
    modelo: str, vae: str, scheduler: str, passos: int, guidance: float, tamanho: str
) -> str:
    partes = [modelo.split("/")[-1]]
    partes.append(f"vae={vae.split('/')[-1] if vae else 'do-modelo'}")
    partes.append(f"amostrador={scheduler or 'do-modelo'}")
    partes.append(f"{passos}p")
    partes.append(f"g={guidance}")
    partes.append(tamanho)
    return "  ".join(partes)


def _apelido(
    modelo: str, vae: str, scheduler: str, passos: int, guidance: float, tamanho: str
) -> str:
    """O nome do ARQUIVO, e ele precisa identificar a variante inteira sozinho.

    Nao e capricho: e um defeito que esta bancada ja teve. O nome so trazia
    modelo e passos, entao quem olhava as imagens e achava um defeito em
    `01-cena3-s1000-stable-diffusion-xl-base-1.0-25p.png` nao tinha como saber
    QUAL VAE e qual amostrador produziram aquilo — e a variante e justamente o
    que se estava comparando.

    O indice da faixa (`01`) tampouco resolvia: ele muda a cada rodada,
    conforme a ordem dos argumentos, entao o nome de ontem aponta para outra
    coisa hoje.
    """

    def limpo(texto: str, vazio: str) -> str:
        bruto = texto.split("/")[-1] if texto else vazio
        return "".join(c if c.isalnum() or c in "-." else "-" for c in bruto)[:28]

    return "__".join(
        (
            limpo(modelo, "modelo"),
            f"vae-{limpo(vae, 'padrao')}",
            f"amo-{limpo(scheduler, 'padrao')}",
            f"{passos}p",
            f"g{guidance}".replace(".", "-"),
            tamanho,
        )
    )


def _pagina(saida: Path, prompts: list[str], linhas: list[dict], sementes: list[int]) -> Path:
    """A pagina de comparacao. Agrupada por PROMPT, uma faixa por variante.

    Por prompt e nao por variante de proposito: o olho compara o que esta lado
    a lado, e o que precisa ser comparado e a MESMA cena entre configuracoes.
    Agrupado por variante, voce rolaria a pagina para comparar e julgaria de
    memoria.
    """
    pedacos = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Bancada de imagem</title>",
        "<style>",
        "body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;",
        "background:#111;color:#eee}",
        "h1{font-size:20px}h2{font-size:16px;margin:32px 0 4px;color:#9cf}",
        ".prompt{color:#aaa;font-style:italic;margin-bottom:16px;max-width:900px}",
        ".faixa{margin-bottom:24px;border-top:1px solid #333;padding-top:12px}",
        ".rot{font-family:ui-monospace,monospace;font-size:12px;color:#fc9;margin-bottom:6px}",
        ".tempo{color:#888;font-size:11px}",
        ".tiras{display:flex;gap:8px;flex-wrap:wrap}",
        "img{max-width:420px;height:auto;border:1px solid #333;background:#000}",
        "</style>",
        "<h1>Bancada de imagem</h1>",
        f"<p class='tempo'>{time.strftime('%Y-%m-%d %H:%M')} &middot; "
        f"sementes fixas: {', '.join(str(s) for s in sementes)} &middot; "
        "julgue pele, maos, reflexo e texto na cena</p>",
    ]

    for indice, prompt in enumerate(prompts):
        pedacos.append(f"<h2>Cena {indice + 1}</h2>")
        pedacos.append(f"<div class='prompt'>{html.escape(prompt)}</div>")

        for linha in linhas:
            arquivos = linha["arquivos"].get(indice)
            if not arquivos:
                continue
            pedacos.append("<div class='faixa'>")
            pedacos.append(f"<div class='rot'>{html.escape(linha['rotulo'])}</div>")
            pedacos.append(
                f"<div class='tempo'>{linha['segundos']:.1f}s por imagem &middot; "
                f"pico de VRAM {linha['vram']:.1f} GB</div>"
            )
            pedacos.append("<div class='tiras'>")
            for arquivo in arquivos:
                pedacos.append(f"<img loading='lazy' src='{html.escape(arquivo)}'>")
            pedacos.append("</div></div>")

    caminho = saida / "index.html"
    caminho.write_text("\n".join(pedacos), encoding="utf-8")
    return caminho


def main() -> int:
    _do_env()

    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--modelos", default=os.environ.get("IMAGEM_MODELO", ""))
    analisador.add_argument(
        "--vaes",
        default=os.environ.get("IMAGEM_VAE", ""),
        help="Lista. O vazio significa 'o VAE do modelo'. Ex.: ',madebyollin/sdxl-vae-fp16-fix'",
    )
    analisador.add_argument("--schedulers", default=os.environ.get("IMAGEM_SCHEDULER", ""))
    analisador.add_argument("--passos", default=os.environ.get("IMAGEM_PASSOS", "30"))
    analisador.add_argument("--guidance", default=os.environ.get("IMAGEM_GUIDANCE", "6.0"))
    analisador.add_argument("--tamanhos", default="1344x768")
    analisador.add_argument("--prompts", default="", help="Arquivo, um prompt por linha.")
    analisador.add_argument("--sementes", type=int, default=2)
    analisador.add_argument("--mesmo-com-o-worker", action="store_true")
    opcoes = analisador.parse_args()

    ocupada, motivo = _worker_esta_com_a_placa()
    if ocupada and not opcoes.mesmo_com_o_worker:
        print(f"ERRO: {motivo}.", file=sys.stderr)
        print(
            "  Os dois modelos nao cabem na mesma placa, e o que acontece nao e\n"
            "  um erro: e queda para CPU, dezenas de vezes mais lento.\n\n"
            "    systemctl --user stop worker-gpu\n"
            "    ./venv/bin/python bancada.py ...\n"
            "    systemctl --user start worker-gpu\n\n"
            "  Para ignorar:  --mesmo-com-o-worker",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
    except ImportError:
        print("ERRO: torch nao esta instalado neste venv.", file=sys.stderr)
        print("  ./venv/bin/pip install -r requirements.txt", file=sys.stderr)
        return 1

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if dispositivo == "cpu":
        print(
            "AVISO: sem placa visivel. Em CPU cada imagem leva minutos, e a\n"
            "       qualidade nao e a mesma. Ctrl+C interrompe.\n"
        )

    if opcoes.prompts:
        prompts = [
            linha.strip()
            for linha in Path(opcoes.prompts).read_text(encoding="utf-8").splitlines()
            if linha.strip() and not linha.startswith("#")
        ]
    else:
        prompts = PROMPTS_PADRAO

    sementes = [1000 + indice for indice in range(max(1, opcoes.sementes))]
    saida = RAIZ / "bancada" / time.strftime("%Y%m%d-%H%M%S")
    saida.mkdir(parents=True, exist_ok=True)

    import imagem

    # As variantes de PIPELINE exigem recarregar o modelo; as de CHAMADA nao.
    # Separadas, o modelo carrega uma vez por combinacao de (modelo, vae,
    # amostrador) em vez de uma vez por imagem.
    de_pipeline = list(
        itertools.product(
            _lista(opcoes.modelos or "stabilityai/stable-diffusion-xl-base-1.0"),
            _lista(opcoes.vaes),
            _lista(opcoes.schedulers),
        )
    )
    de_chamada = list(
        itertools.product(
            [int(passo) for passo in _lista(opcoes.passos) if passo],
            [float(g) for g in _lista(opcoes.guidance) if g],
            [t for t in _lista(opcoes.tamanhos) if t],
        )
    )

    total = len(de_pipeline) * len(de_chamada) * len(prompts) * len(sementes)
    print(
        f"{total} imagens: {len(de_pipeline)} pipeline(s) x {len(de_chamada)} ajuste(s) "
        f"x {len(prompts)} cena(s) x {len(sementes)} semente(s)"
    )
    print(f"Saida: {saida}\n")

    linhas: list[dict] = []
    feitas = 0

    for modelo, vae, scheduler in de_pipeline:
        print(
            f"--- carregando {modelo} (vae={vae or 'do modelo'}, "
            f"amostrador={scheduler or 'do modelo'})"
        )
        inicio = time.perf_counter()
        try:
            pipe = imagem._montar_pipeline(dispositivo, modelo=modelo, vae=vae, scheduler=scheduler)
        except Exception as erro:
            print(f"    FALHOU: {type(erro).__name__}: {erro}\n", file=sys.stderr)
            continue
        print(f"    pronto em {time.perf_counter() - inicio:.0f}s")

        for passos, guidance, tamanho in de_chamada:
            largura, altura = _medidas(tamanho)
            rotulo = _rotulo(modelo, vae, scheduler, passos, guidance, tamanho)
            apelido = _apelido(modelo, vae, scheduler, passos, guidance, tamanho)
            arquivos: dict[int, list[str]] = {}
            tempos: list[float] = []

            if dispositivo == "cuda":
                torch.cuda.reset_peak_memory_stats()

            for indice, prompt in enumerate(prompts):
                arquivos[indice] = []
                for semente in sementes:
                    gerador = torch.Generator(device="cpu").manual_seed(semente)
                    comeco = time.perf_counter()
                    try:
                        resultado = pipe(
                            prompt=prompt,
                            negative_prompt=imagem.IMAGEM_NEGATIVO or None,
                            num_inference_steps=passos,
                            guidance_scale=guidance,
                            width=largura,
                            height=altura,
                            generator=gerador,
                        )
                    except Exception as erro:
                        print(f"    FALHOU em {rotulo}: {erro}", file=sys.stderr)
                        break
                    tempos.append(time.perf_counter() - comeco)

                    nome = f"cena{indice + 1}__s{semente}__{apelido}.png"
                    resultado.images[0].save(saida / nome)
                    arquivos[indice].append(nome)

                    feitas += 1
                    print(f"    [{feitas}/{total}] {rotulo}  cena {indice + 1}  semente {semente}")

            if tempos:
                linhas.append(
                    {
                        "rotulo": rotulo,
                        "arquivos": arquivos,
                        "segundos": sum(tempos) / len(tempos),
                        "vram": _pico_de_vram(torch),
                    }
                )

        del pipe
        import gc

        gc.collect()
        if dispositivo == "cuda":
            torch.cuda.empty_cache()

    if not linhas:
        print("\nNada foi gerado. Veja os erros acima.", file=sys.stderr)
        return 1

    # Um manifesto ao lado das imagens. O nome do arquivo ja identifica a
    # variante; o manifesto guarda o que nele nao cabe — o prompt de cada cena,
    # o negativo em vigor e o custo medido de cada variante.
    (saida / "variantes.json").write_text(
        json.dumps(
            {
                "prompts": {f"cena{i + 1}": prompt for i, prompt in enumerate(prompts)},
                "negativo": imagem.IMAGEM_NEGATIVO,
                "sementes": sementes,
                "variantes": [
                    {
                        "rotulo": linha["rotulo"],
                        "segundos_por_imagem": round(linha["segundos"], 1),
                        "pico_vram_gb": round(linha["vram"], 1),
                    }
                    for linha in linhas
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pagina = _pagina(saida, prompts, linhas, sementes)

    print(f"\n{'variante':<58}  {'s/imagem':>9}  {'pico VRAM':>10}")
    print("-" * 82)
    for linha in linhas:
        print(f"{linha['rotulo'][:58]:<58}  {linha['segundos']:8.1f}s  {linha['vram']:8.1f} GB")

    print(f"\nAbra e julgue:  xdg-open {pagina}")
    print(f"Prompts, negativo e custo por variante: {saida / 'variantes.json'}")
    print(
        "\nOlhe nesta ordem, que e a ordem em que a geracao realista quebra:\n"
        "  1. maos e dedos       — o defeito que mais denuncia\n"
        "  2. pele               — plastico liso demais e o segundo\n"
        "  3. reflexo e metal    — onde o VAE em float16 deixa manchas\n"
        "  4. texto na cena      — placas e rotulos viram rabisco\n"
        "  5. fundo desfocado    — geometria que nao fecha\n"
        "\nO que ganhar vai para o `.env` do WORKER e precisa de restart:\n"
        "    IMAGEM_MODELO=      IMAGEM_VAE=      IMAGEM_SCHEDULER=\n"
        "    IMAGEM_PASSOS=      IMAGEM_GUIDANCE=\n"
        "    systemctl --user restart worker-gpu\n"
        "\nO TAMANHO nao: ele viaja no pedido, entao quem manda e o cliente."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
