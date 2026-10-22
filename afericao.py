"""Afere a maquina DURANTE uma animacao e sugere a configuracao do worker.

    python3 afericao.py                 5 minutos (rode com uma animacao em curso)
    python3 afericao.py --minutos 10
    nohup python3 afericao.py --minutos 10 &   pelo SSH, e pode sair: o relatorio fica salvo

So biblioteca padrao: roda com o `python3` do sistema, sem o venv.

Mede a cada `--intervalo` segundos: uso da GPU e da VRAM, leitura do disco,
memoria do sistema, a memoria do servico do worker (o FramePack roda dentro
dele, e o cache dele conta no `MemoryMax`) e o progresso da animacao. No fim:

- o RELATORIO, com o que esta rodando FORA do worker (uma VM, o navegador),
  porque a sugestao depende disso: com a VM desligada na hora da afericao,
  sobra RAM que nao vai sobrar amanha;
- a SUGESTAO de `.env` (`WORKER_MEMORY_MAX`, `ANIMACAO_RAM_FIXA_GB`);
- um PROMPT pronto para uma LLM, que pede a ela para conversar com voce
  sobre o uso normal da maquina ANTES de recomendar comprometer RAM.

Tudo vai para `medicoes/afericao-<data>.md` (fora do git).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# O transformer do FramePack (FramePackI2V_HY, bf16). Medido no disco.
TRANSFORMER_GB = 25.7
# O que fica de fora para o desktop, o cache dos outros programas e folga.
RESERVA_GB = 4.0
# Nomes de processos de maquina virtual: a memoria deles e reservada inteira
# quando a VM liga, mesmo que agora pareca pouca.
NOMES_DE_VM = ("qemu", "virtualbox", "vboxheadless", "vmware", "vmx")


def _do_env() -> None:
    """Le o `.env` como o systemd le: a ULTIMA ocorrencia de cada chave vence,
    e uma variavel de verdade no ambiente manda mais que o arquivo."""
    arquivo = RAIZ / ".env"
    if not arquivo.is_file():
        return

    do_arquivo: dict[str, str] = {}
    for linha in arquivo.read_text(encoding="utf-8").splitlines():
        limpa = linha.strip()
        if not limpa or limpa.startswith("#") or "=" not in limpa:
            continue
        chave, _, valor = limpa.partition("=")
        do_arquivo[chave.strip()] = valor.strip().strip('"').strip("'")

    for chave, valor in do_arquivo.items():
        os.environ.setdefault(chave, valor)


# ---------------------------------------------------------------------------
# Leituras
# ---------------------------------------------------------------------------
def _gb(kib: float) -> float:
    return kib / 1024 / 1024


def _meminfo() -> dict[str, float]:
    """Os campos do /proc/meminfo, em GB."""
    campos = {}
    for linha in Path("/proc/meminfo").read_text().splitlines():
        nome, _, resto = linha.partition(":")
        partes = resto.split()
        if partes and partes[0].isdigit():
            campos[nome] = _gb(int(partes[0]))
    return campos


def _lido_do_disco_kib() -> int:
    """KiB lidos do disco desde o boot (`pgpgin`, o mesmo do `bi` do vmstat)."""
    for linha in Path("/proc/vmstat").read_text().splitlines():
        if linha.startswith("pgpgin "):
            return int(linha.split()[1])
    return 0


def _gpu() -> dict | None:
    try:
        saida = subprocess.run(
            [  # noqa: S607 - o nvidia-smi do PATH e o que o dono da maquina usa
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    uso, usada, total, nome = [parte.strip() for parte in saida.splitlines()[0].split(",")]
    return {"uso": float(uso), "vram_usada_gb": float(usada) / 1024,
            "vram_total_gb": float(total) / 1024, "nome": nome}  # fmt: skip


def _cgroup_do_worker() -> Path | None:
    try:
        relativo = subprocess.run(
            ["systemctl", "--user", "show", "worker-gpu", "-p", "ControlGroup", "--value"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    caminho = Path("/sys/fs/cgroup") / relativo.lstrip("/")
    return caminho if relativo and caminho.is_dir() else None


def _memoria_do_servico(cg: Path | None) -> dict | None:
    if cg is None:
        return None
    try:
        teto = (cg / "memory.max").read_text().strip()
        estado = {"atual_gb": _gb(int((cg / "memory.current").read_text()) / 1024)}
        estado["teto_gb"] = None if teto == "max" else _gb(int(teto) / 1024)
        for linha in (cg / "memory.stat").read_text().splitlines():
            nome, valor = linha.split()
            if nome in ("anon", "file"):
                estado[f"{nome}_gb"] = _gb(int(valor) / 1024)
        for linha in (cg / "memory.events").read_text().splitlines():
            nome, valor = linha.split()
            if nome == "max":
                estado["eventos_max"] = int(valor)
    except (OSError, ValueError):
        return None
    return estado


def _saude(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{url}/health/", timeout=5) as resposta:  # noqa: S310
            return json.loads(resposta.read())
    except (OSError, ValueError):
        return None


def _processos_fora_do_worker(cg: Path | None, limite: int = 8) -> list[dict]:
    """Os processos que mais ocupam RAM e NAO sao do servico do worker."""
    do_worker = "/" + str(cg.relative_to("/sys/fs/cgroup")) if cg else None
    processos = []
    for pasta in Path("/proc").iterdir():
        if not pasta.name.isdigit():
            continue
        try:
            grupo = (pasta / "cgroup").read_text()
            if do_worker and do_worker in grupo:
                continue
            rss = 0
            for linha in (pasta / "status").read_text().splitlines():
                if linha.startswith("VmRSS:"):
                    rss = int(linha.split()[1])
                    break
            comando = (pasta / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if rss:
            processos.append({"pid": int(pasta.name), "rss_gb": _gb(rss), "comando": comando[:90]})
    processos.sort(key=lambda processo: processo["rss_gb"], reverse=True)
    return processos[:limite]


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------
def _animacao_em_curso(saude: dict | None) -> dict | None:
    if not saude:
        return None
    lista = (saude.get("animacao") or {}).get("em_curso") or []
    return lista[0] if lista else None


def coletar(minutos: float, intervalo: float, url: str) -> dict:
    cg = _cgroup_do_worker()
    inicio_saude = _saude(url)
    anim_inicio = _animacao_em_curso(inicio_saude)
    servico_inicio = _memoria_do_servico(cg)
    lido_antes = _lido_do_disco_kib()
    comeco = time.monotonic()
    amostras = []

    fim = comeco + minutos * 60
    while time.monotonic() < fim:
        antes, t0 = _lido_do_disco_kib(), time.monotonic()
        time.sleep(intervalo)
        depois, t1 = _lido_do_disco_kib(), time.monotonic()
        gpu = _gpu()
        memoria = _meminfo()
        amostra = {
            "gpu": gpu["uso"] if gpu else None,
            "vram_gb": gpu["vram_usada_gb"] if gpu else None,
            "disco_mb_s": (depois - antes) / 1024 / (t1 - t0),
            "disponivel_gb": memoria.get("MemAvailable"),
        }
        amostras.append(amostra)
        if len(amostras) % max(1, int(30 / intervalo)) == 0:
            passado = int(time.monotonic() - comeco)
            print(
                f"  {passado:4d}s  GPU {amostra['gpu'] or 0:3.0f}%  disco "
                f"{amostra['disco_mb_s']:5.0f} MB/s  RAM disponivel "
                f"{amostra['disponivel_gb'] or 0:4.1f} GB",
                flush=True,
            )

    duracao = time.monotonic() - comeco
    fim_saude = _saude(url)
    return {
        "quando": datetime.now().isoformat(timespec="minutes"),
        "duracao_s": duracao,
        "amostras": amostras,
        "disco_total_gb": _gb(_lido_do_disco_kib() - lido_antes),
        "memoria": _meminfo(),
        "gpu": _gpu(),
        "servico_inicio": servico_inicio,
        "servico_fim": _memoria_do_servico(cg),
        "animacao_inicio": anim_inicio,
        "animacao_fim": _animacao_em_curso(fim_saude),
        "fora_do_worker": _processos_fora_do_worker(cg),
        "config": {
            chave: os.environ.get(chave, padrao)
            for chave, padrao in (
                ("WORKER_MEMORY_MAX", "18G"),
                ("ANIMACAO_RAM_FIXA_GB", "0"),
                ("ANIMACAO_RESOLUCAO", "640"),
                ("ANIMACAO_PASSOS", "25"),
                ("ANIMACAO_TEACACHE", "sim"),
            )
        },
    }


# ---------------------------------------------------------------------------
# Analise e sugestao (sem leitura nenhuma: testavel)
# ---------------------------------------------------------------------------
def _media(valores: list) -> float | None:
    validos = [valor for valor in valores if valor is not None]
    return sum(validos) / len(validos) if validos else None


def _gb_da_config(texto: str) -> float:
    texto = str(texto).strip().upper()
    for sufixo, fator in (("G", 1.0), ("M", 1 / 1024), ("T", 1024.0)):
        if texto.endswith(sufixo):
            return float(texto[:-1]) * fator
    return float(texto) / 1024**3 if texto.isdigit() else 0.0


def analisar(medicao: dict) -> dict:
    amostras = medicao["amostras"]
    gpu = _media([amostra["gpu"] for amostra in amostras])
    disco = _media([amostra["disco_mb_s"] for amostra in amostras])
    acima_de_90 = (
        sum(1 for amostra in amostras if (amostra["gpu"] or 0) >= 90) / len(amostras)
        if amostras
        else 0
    )

    inicio, fim = medicao.get("animacao_inicio"), medicao.get("animacao_fim")
    passo_s = None
    if inicio and fim and inicio["id"] == fim["id"] and fim["progress"] > inicio["progress"]:
        passos = int(medicao["config"]["ANIMACAO_PASSOS"])
        secoes = max(round(int(fim["seconds"]) * 30 / 36), 1)
        por_passo = 90 / (secoes * passos)
        andados = (fim["progress"] - inicio["progress"]) / por_passo
        passo_s = medicao["duracao_s"] / andados if andados else None

    s0, s1 = medicao.get("servico_inicio") or {}, medicao.get("servico_fim") or {}
    eventos = (s1.get("eventos_max") or 0) - (s0.get("eventos_max") or 0)

    if not (inicio or fim):
        gargalo = "sem_animacao"
    elif gpu is not None and gpu >= 85:
        gargalo = "gpu"
    elif disco is not None and disco >= 100:
        gargalo = "disco"
    else:
        gargalo = "misto"

    return {
        "gpu_media": gpu,
        "gpu_acima_de_90": acima_de_90,
        "disco_mb_s": disco,
        "passo_s": passo_s,
        "eventos_max": eventos,
        "gargalo": gargalo,
    }


def sugerir(medicao: dict, analise: dict) -> dict:
    """O `.env` que aproveita a RAM que esta livre AGORA, e o que ele supoe."""
    total = medicao["memoria"].get("MemTotal", 0)
    servico_fim = medicao.get("servico_fim") or {}
    # Fora do worker: o que o sistema tem ocupado de verdade (sem cache, que e
    # recuperavel) menos a memoria propria do servico. A lista de processos e
    # so para mostrar quem e quem: somar os maiores esqueceria os pequenos.
    usado = total - medicao["memoria"].get("MemAvailable", total)
    fora = max(0.0, usado - (servico_fim.get("anon_gb") or 0))
    vms = [
        processo
        for processo in medicao["fora_do_worker"]
        if any(nome in processo["comando"].lower() for nome in NOMES_DE_VM)
    ]

    fixa_atual = float(medicao["config"]["ANIMACAO_RAM_FIXA_GB"] or 0)
    base = max(0.0, (servico_fim.get("anon_gb") or 0) - fixa_atual)
    teto_atual = _gb_da_config(medicao["config"]["WORKER_MEMORY_MAX"])
    teto = max(math.floor(total - fora - RESERVA_GB), math.ceil(teto_atual))

    # O que falta para o transformer caber inteiro no cache do servico. NAO
    # depende de fixar camadas (ANIMACAO_RAM_FIXA_GB): o que se fixa sai do
    # mesmo teto que o cache, e a conta e a mesma. So diminui subindo o teto
    # ou diminuindo o modelo (o transformer em fp8 tem ~13 GB).
    falta_atual = max(0.0, TRANSFORMER_GB + base - teto_atual)
    falta = max(0.0, TRANSFORMER_GB + base - teto)
    falta_fp8 = max(0.0, TRANSFORMER_GB / 2 + base - teto_atual)

    if analise["gargalo"] == "sem_animacao":
        motivo = "sem animacao em curso nao da para saber o gargalo: afira durante uma."
    elif analise["gargalo"] == "gpu":
        motivo = "a GPU ja trabalha quase o tempo todo: mais RAM quase nao muda o tempo."
    elif falta == 0:
        motivo = "com este teto o transformer cabe inteiro no cache: o disco sai do caminho."
    else:
        motivo = (
            f"o disco limita: hoje faltam ~{falta_atual:.0f} GB de cache para o transformer; "
            f"com este teto, ~{falta:.0f} GB. Para zerar sem mais RAM, so o transformer em "
            f"fp8 (~{TRANSFORMER_GB / 2:.0f} GB), que no teto atual deixaria faltar "
            f"~{falta_fp8:.0f} GB."
        )

    return {
        "total_gb": total,
        "fora_gb": fora,
        "vms": vms,
        "base_gb": base,
        "teto_atual_gb": teto_atual,
        "falta_atual_gb": falta_atual,
        "falta_gb": falta,
        "falta_fp8_gb": falta_fp8,
        "WORKER_MEMORY_MAX": f"{teto}G",
        # Fixar camadas nao reduz a falta (veja acima): fica desligado.
        "ANIMACAO_RAM_FIXA_GB": 0,
        "motivo": motivo,
    }


# ---------------------------------------------------------------------------
# Relatorio
# ---------------------------------------------------------------------------
def _f(valor, formato="{:.0f}", vazio="?"):
    return vazio if valor is None else formato.format(valor)


def relatorio(medicao: dict, analise: dict, sugestao: dict) -> str:
    gpu = medicao.get("gpu") or {}
    s1 = medicao.get("servico_fim") or {}
    anim_i, anim_f = medicao.get("animacao_inicio"), medicao.get("animacao_fim")
    linhas = [
        f"# Afericao do worker-gpu — {medicao['quando']}",
        "",
        f"Medido por {medicao['duracao_s'] / 60:.1f} min, {len(medicao['amostras'])} amostras.",
        "",
        "## Hardware",
        f"- GPU: {gpu.get('nome', '?')}, VRAM {_f(gpu.get('vram_total_gb'), '{:.1f}')} GB",
        f"- RAM: {sugestao['total_gb']:.1f} GB",
        "",
        "## Durante a medicao",
        f"- GPU: media {_f(analise['gpu_media'])}%, "
        f"{analise['gpu_acima_de_90'] * 100:.0f}% do tempo acima de 90%",
        f"- VRAM usada (media): "
        f"{_f(_media([a['vram_gb'] for a in medicao['amostras']]), '{:.1f}')} GB",
        f"- Disco, leitura media: {_f(analise['disco_mb_s'])} MB/s "
        f"({medicao['disco_total_gb']:.1f} GB no total)",
        f"- RAM disponivel (media): "
        f"{_f(_media([a['disponivel_gb'] for a in medicao['amostras']]), '{:.1f}')} GB",
        f"- Servico do worker: {_f(s1.get('atual_gb'), '{:.1f}')} GB de "
        f"{_f(s1.get('teto_gb'), '{:.1f}', 'sem teto')} (memoria propria "
        f"{_f(s1.get('anon_gb'), '{:.1f}')} GB, cache {_f(s1.get('file_gb'), '{:.1f}')} GB); "
        f"o teto espremeu o cache {analise['eventos_max']} vezes",
    ]
    if anim_i or anim_f:
        de = anim_i["progress"] if anim_i else "?"
        para = anim_f["progress"] if anim_f else "terminou"
        linhas.append(
            f"- Animacao: de {de}% para {para}%; "
            f"{_f(analise['passo_s'], '{:.0f} s por passo', 'tempo por passo desconhecido')}"
        )
    else:
        linhas.append("- Animacao: NENHUMA em curso — a medicao nao representa a geracao de video.")
    linhas += [f"- Gargalo: **{analise['gargalo']}**", "", "## Fora do worker (agora)"]
    linhas += [
        f"- {processo['rss_gb']:.1f} GB  {processo['comando']}"
        for processo in medicao["fora_do_worker"]
    ]
    if sugestao["vms"]:
        linhas.append(
            "- **Ha maquina virtual ligada.** A memoria dela na hora da medicao pode ser menor que "
            "a configurada: confira com `virsh dominfo <nome>`."
        )
    else:
        linhas.append(
            "- Nenhuma maquina virtual ligada AGORA. Se voce costuma ligar uma, a sugestao abaixo "
            "conta com a RAM dela."
        )
    linhas += [
        "",
        "## Configuracao atual",
        *[f"- {chave}={valor}" for chave, valor in medicao["config"].items()],
        "",
        "## Sugestao (vale para a carga que estava rodando agora)",
        f"- WORKER_MEMORY_MAX={sugestao['WORKER_MEMORY_MAX']}",
        f"- ANIMACAO_RAM_FIXA_GB={sugestao['ANIMACAO_RAM_FIXA_GB']}",
        f"- Por que: {sugestao['motivo']}",
        f"- Conta do teto: {sugestao['total_gb']:.0f} GB de RAM - {sugestao['fora_gb']:.1f} GB "
        f"fora do worker - {RESERVA_GB:.0f} GB de reserva. Cache que falta para o transformer "
        f"(~{TRANSFORMER_GB:.0f} GB + {sugestao['base_gb']:.1f} GB de memoria propria - teto): "
        f"hoje ~{sugestao['falta_atual_gb']:.0f} GB, com o teto sugerido "
        f"~{sugestao['falta_gb']:.0f} GB.",
        "- Aplicar: editar o `.env` e rodar `./deploy/instalar.sh` ENTRE animacoes (o restart "
        "cancela a que estiver rodando).",
    ]
    return "\n".join(linhas)


def prompt(texto_do_relatorio: str) -> str:
    return f"""Voce vai me ajudar a ajustar a memoria de um servico de GPU na minha maquina.

Contexto: o "worker-gpu" e um servico (systemd de usuario) que gera texto, imagem e
video numa placa de 8 GB. O video e o FramePack: um transformer de ~{TRANSFORMER_GB:.0f} GB que
fica mapeado do disco e e lido INTEIRO a cada passo da difusao. O que nao cabe no cache
de disco e relido do disco a cada passo, e a GPU fica esperando. O servico tem um teto
de memoria (WORKER_MEMORY_MAX, o MemoryMax do systemd) que conta tambem o cache dele.
O que falta de cache e aproximadamente: transformer + memoria propria do servico - teto.
So duas coisas diminuem essa falta: subir o teto (tirando RAM do resto da maquina) ou
usar o transformer em fp8 (~{TRANSFORMER_GB / 2:.0f} GB, com uma pequena perda de qualidade).
Fixar parte do modelo na RAM (ANIMACAO_RAM_FIXA_GB) NAO ajuda: a parte fixa sai do mesmo
teto que o cache.

Abaixo esta uma afericao feita DURANTE uma geracao de video, com uma sugestao calculada
a partir do que estava rodando NAQUELE momento.

Antes de recomendar qualquer valor, CONVERSE comigo. Pergunte, uma coisa de cada vez:
1. o que eu costumo deixar ligado enquanto o worker gera video (maquina virtual e
   quanta RAM ela tem configurada, navegador, IDE, jogos) e se a medicao foi feita
   num momento tipico ou atipico (por exemplo, com a VM desligada);
2. o que e mais importante para mim: video mais rapido ou a maquina responsiva para
   eu usar ao mesmo tempo;
3. se o computador fica sozinho gerando a noite (ai da para comprometer mais).

So depois, me de: o valor de WORKER_MEMORY_MAX para o meu uso normal, se vale a pena
partir para o transformer em fp8, quanto eu ganharia de tempo por passo (use o gargalo
e o uso da GPU medidos; se a GPU ja passa de 85% na media, diga que mais RAM quase nao
ajuda), o risco (memoria
apertada demais faz o kernel matar o servico, ou leva a VM e o desktop para o swap) e
os comandos para aplicar (editar o .env e rodar ./deploy/instalar.sh entre animacoes).
Se a medicao nao tiver animacao em curso, diga que ela nao serve e peca outra.

{texto_do_relatorio}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--minutos", type=float, default=5.0)
    parser.add_argument("--intervalo", type=float, default=2.0)
    parser.add_argument("--url", help="URL do worker (padrao: BIND_HOST:BIND_PORT do .env)")
    argumentos = parser.parse_args()

    _do_env()
    url = argumentos.url or (
        f"http://{os.environ.get('BIND_HOST', '127.0.0.1')}:{os.environ.get('BIND_PORT', '8090')}"
    )

    print(f"Aferindo por {argumentos.minutos:g} min (a cada {argumentos.intervalo:g} s)...")
    if _animacao_em_curso(_saude(url)) is None:
        print("  AVISO: nenhuma animacao em curso. A medicao so vale durante uma.", flush=True)

    medicao = coletar(argumentos.minutos, argumentos.intervalo, url)
    analise = analisar(medicao)
    sugestao = sugerir(medicao, analise)
    texto = relatorio(medicao, analise, sugestao)

    pasta = RAIZ / "medicoes"
    pasta.mkdir(exist_ok=True)
    arquivo = pasta / f"afericao-{datetime.now():%Y%m%d-%H%M}.md"
    arquivo.write_text(
        f"{texto}\n\n## Prompt para uma LLM\n\nCopie o bloco abaixo inteiro.\n\n"
        f"```text\n{prompt(texto)}```\n",
        encoding="utf-8",
    )

    print()
    print(texto)
    print(f"\nRelatorio e prompt para a LLM: {arquivo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
