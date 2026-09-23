"""Quanta memoria este processo esta usando, e se ele foi para o swap.

Existe por um defeito de diagnostico. Esta maquina tem 32 GB e roda, ao lado
do worker, um modelo de texto de 30B — e quem foi para o swap nao foi o
Ollama: foi o worker. Nada no `/health/` dizia isso, entao a suspeita caiu no
lugar errado por dias.

E o worker retem RAM POR DESENHO, o que torna a medicao ainda mais necessaria:
`enable_model_cpu_offload()` mantem os pesos do modelo de imagem na RAM e sobe
para a placa so o submodulo em uso. E o que faz o pico de VRAM cair de ~7 GB
para ~5,4 GB, e o que permite a difusao caber ao lado do Ollama — mas o preco
sao 5 a 6 GB de memoria ANONIMA parada entre pedidos.

Memoria anonima parada e exatamente o candidato preferido do kernel quando ele
precisa de paginas. Os pesos do Ollama, ao contrario, sao um arquivo mapeado
(`mmap`): sob pressao o kernel descarta aquelas paginas de graca, porque elas
estao limpas e podem ser lidas do disco de novo. As do worker, nao — elas
precisam ser ESCRITAS no swap antes de serem tomadas.

Por isso o numero que interessa aqui e o `VmSwap` deste processo, e nao o do
sistema: ele responde "o worker esta desgastando o SSD?" com um numero, e nao
com um palpite.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("worker-gpu.recursos")

_ARQUIVO_DE_ESTADO = "/proc/self/status"

# Ja avisamos sobre o swap? O aviso e uma vez por subida, e nao por leitura: o
# `/health/` pode ser consultado em laco pelo systemd, e um aviso por consulta
# enterraria tudo o que importa no journal.
_avisou_do_swap = False


def _kb_de(linhas: list[str], campo: str) -> int | None:
    alvo = f"{campo}:"
    for linha in linhas:
        if linha.startswith(alvo):
            partes = linha.split()
            if len(partes) >= 2 and partes[1].isdigit():
                return int(partes[1])
    return None


def memoria() -> dict:
    """RSS e swap deste processo, em MB. `{}` onde nao der para saber.

    Sem levantar e sem depender de pacote: le o `/proc/self/status`, que existe
    em qualquer Linux. Numa plataforma sem `/proc` devolve `{}` — o `/health/`
    perde um campo e continua respondendo, que e a regra dele.
    """
    try:
        with open(_ARQUIVO_DE_ESTADO, encoding="utf-8") as arquivo:
            linhas = arquivo.read().splitlines()
    except OSError:
        return {}

    rss = _kb_de(linhas, "VmRSS")
    swap = _kb_de(linhas, "VmSwap")

    estado: dict = {}
    if rss is not None:
        estado["rss_mb"] = round(rss / 1024, 1)
    if swap is not None:
        estado["swap_mb"] = round(swap / 1024, 1)
        _avisar_do_swap(swap)

    return estado


def _avisar_do_swap(swap_kb: int) -> None:
    """Uma linha no journal na primeira vez que este processo toca o swap.

    E o aviso que faltava. Um worker no swap nao da erro e nao fica lento de um
    jeito que se note: ele fica lento de um jeito que se atribui ao modelo, e
    quem investiga vai olhar a GPU. O journal dizendo "fui eu" economiza dias.
    """
    global _avisou_do_swap

    if _avisou_do_swap or swap_kb <= 0:
        return

    _avisou_do_swap = True
    logger.warning(
        "Este processo tem %.0f MB no swap. Nao e erro, mas e escrita no disco "
        "e leitura de volta a cada acesso — e os pesos do modelo de imagem "
        "ficam na RAM por desenho (enable_model_cpu_offload). Veja a secao "
        "'Convivendo com a maquina' no README: MemorySwapMax=0 na unit e zram "
        "resolvem isso sem desligar o swap do sistema.",
        swap_kb / 1024,
    )


def esquecer_o_aviso() -> None:
    """So para os testes: o estado do aviso e de processo e vaza entre eles."""
    global _avisou_do_swap
    _avisou_do_swap = False
