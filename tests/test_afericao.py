"""A analise e a sugestao da afericao, com medicoes sinteticas.

As leituras (/proc, nvidia-smi, cgroup) ficam de fora: o que se exercita e a
conta que transforma os numeros em gargalo e em `.env`.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def afericao():
    import afericao

    return afericao


def _medicao(gpu=55, disco=350, vm=True, disponivel=20.0, anon=4.9, progresso=(36, 70)):
    amostras = [{"gpu": gpu, "vram_gb": 5.2, "disco_mb_s": disco, "disponivel_gb": disponivel}] * 10
    fora = [{"pid": 1, "rss_gb": 4.0, "comando": "/usr/bin/qemu-system-x86_64 -name guest=win11"}]
    return {
        "quando": "2026-10-22T11:00",
        "duracao_s": 300,
        "amostras": amostras,
        "disco_total_gb": disco * 300 / 1024,
        "memoria": {"MemTotal": 30.0, "MemAvailable": disponivel},
        "gpu": {"nome": "RTX 3050", "vram_total_gb": 8.0},
        "servico_inicio": {"eventos_max": 100},
        "servico_fim": {
            "eventos_max": 5100,
            "anon_gb": anon,
            "file_gb": 13.0,
            "atual_gb": 17.9,
            "teto_gb": 18.0,
        },
        "animacao_inicio": {"id": "v", "progress": progresso[0], "seconds": 2},
        "animacao_fim": {"id": "v", "progress": progresso[1], "seconds": 2},
        "fora_do_worker": fora if vm else [],
        "config": {
            "WORKER_MEMORY_MAX": "18G",
            "ANIMACAO_RAM_FIXA_GB": "0",
            "ANIMACAO_RESOLUCAO": "640",
            "ANIMACAO_PASSOS": "25",
            "ANIMACAO_TEACACHE": "sim",
        },
    }


def test_disco_alto_e_gpu_pela_metade_e_gargalo_de_disco(afericao):
    analise = afericao.analisar(_medicao())

    assert analise["gargalo"] == "disco"
    assert analise["eventos_max"] == 5000


def test_gpu_quase_sempre_cheia_e_gargalo_de_gpu(afericao):
    assert afericao.analisar(_medicao(gpu=92, disco=20))["gargalo"] == "gpu"


def test_sem_animacao_a_medicao_nao_vale(afericao):
    medicao = _medicao()
    medicao["animacao_inicio"] = medicao["animacao_fim"] = None

    assert afericao.analisar(medicao)["gargalo"] == "sem_animacao"


def test_o_tempo_por_passo_sai_do_progresso(afericao):
    """2 s de video = 2 trechos x 25 passos: cada passo vale 1,8%. De 36 a 70%
    em 5 min sao ~19 passos, ~16 s cada."""
    analise = afericao.analisar(_medicao())

    assert analise["passo_s"] == pytest.approx(300 / (34 / 1.8))


def test_a_sugestao_sobe_o_teto_e_mostra_a_falta_de_cache(afericao):
    medicao = _medicao(disponivel=20.0, anon=4.9)
    sugestao = afericao.sugerir(medicao, afericao.analisar(medicao))

    # 30 de RAM, 10 usados (5,1 fora do worker), reserva de 4: teto 20.
    assert sugestao["WORKER_MEMORY_MAX"] == "20G"
    # Falta = 25,7 + 4,9 - teto: 12,6 hoje (18), 10,6 com o sugerido (20).
    assert sugestao["falta_atual_gb"] == pytest.approx(12.6)
    assert sugestao["falta_gb"] == pytest.approx(10.6)
    assert "fp8" in sugestao["motivo"]
    assert sugestao["vms"]


def test_fixar_camadas_nunca_e_sugerido(afericao):
    """O que se fixa sai do mesmo teto que o cache: a falta nao muda. Foi uma
    sugestao errada de uma versao anterior (fixar 12 GB com o teto em 20)."""
    for gpu, disco in ((55, 350), (92, 20)):
        medicao = _medicao(gpu=gpu, disco=disco)
        assert afericao.sugerir(medicao, afericao.analisar(medicao))["ANIMACAO_RAM_FIXA_GB"] == 0


def test_com_gpu_cheia_diz_que_ram_nao_ajuda(afericao):
    medicao = _medicao(gpu=92, disco=20)
    sugestao = afericao.sugerir(medicao, afericao.analisar(medicao))

    assert "quase nao muda" in sugestao["motivo"]


def test_o_teto_sugerido_nunca_e_menor_que_o_atual(afericao):
    medicao = _medicao(disponivel=8.0)
    sugestao = afericao.sugerir(medicao, afericao.analisar(medicao))

    assert sugestao["WORKER_MEMORY_MAX"] == "18G"


def test_o_prompt_manda_a_llm_perguntar_antes_e_leva_o_relatorio(afericao):
    medicao = _medicao(vm=False)
    analise = afericao.analisar(medicao)
    texto = afericao.relatorio(medicao, analise, afericao.sugerir(medicao, analise))

    pedido = afericao.prompt(texto)

    assert "CONVERSE comigo" in pedido
    assert "VM desligada" in pedido
    assert "Nenhuma maquina virtual ligada AGORA" in pedido


def test_sem_animacao_nao_sugere_e_diz_por_que(afericao):
    medicao = _medicao()
    medicao["animacao_inicio"] = medicao["animacao_fim"] = None

    sugestao = afericao.sugerir(medicao, afericao.analisar(medicao))

    assert sugestao["ANIMACAO_RAM_FIXA_GB"] == 0
    assert "sem animacao" in sugestao["motivo"]


def test_no_fp8_nao_falta_cache_e_mostra_a_folga(afericao):
    """No fp8 o transformer ja esta na memoria propria: somar o bf16 por cima
    (como fazia) dizia que faltavam ~23 GB, com o disco parado."""
    medicao = _medicao(gpu=96, disco=1, disponivel=7.0, anon=15.6)
    medicao["animacao_fim"]["model"] = "framepack-fp8"
    medicao["modelos"] = {"framepack": "bf16", "framepack-fp8": "fp8"}

    sugestao = afericao.sugerir(medicao, afericao.analisar(medicao))

    assert sugestao["fp8"] is True
    assert sugestao["falta_gb"] == 0
    assert sugestao["folga_gb"] == pytest.approx(2.4)
    assert "fp8" in sugestao["motivo"]
