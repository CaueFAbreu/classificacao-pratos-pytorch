import json
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.avaliacao import calcular_metricas
from src.config import CLASSES
from src.relatorio import carregar_resultados, gerar_tabela
from src.treino import executar_epoca, treinar


def test_epoca_de_treino_atualiza_os_pesos():
    torch.manual_seed(0)
    modelo = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    dados = TensorDataset(torch.randn(16, 3, 2, 2), torch.randint(0, 3, (16,)))
    pesos_antes = modelo[1].weight.detach().clone()

    resultado = executar_epoca(
        modelo,
        DataLoader(dados, batch_size=8),
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        torch.optim.SGD(modelo.parameters(), lr=0.1),
    )

    assert not torch.equal(pesos_antes, modelo[1].weight)
    assert math.isfinite(resultado["perda"])
    assert 0 <= resultado["acuracia"] <= 1
    assert len(resultado["previstos"]) == 16


def test_epoca_de_avaliacao_nao_altera_os_pesos():
    modelo = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    dados = TensorDataset(torch.randn(8, 3, 2, 2), torch.randint(0, 3, (8,)))
    pesos_antes = modelo[1].weight.detach().clone()

    executar_epoca(modelo, DataLoader(dados), nn.CrossEntropyLoss(), torch.device("cpu"))

    assert torch.equal(pesos_antes, modelo[1].weight)


def test_metricas_e_confusoes_mais_frequentes():
    reais = [0, 0, 1, 1, 1, 2]
    previstos = [0, 1, 1, 0, 0, 2]
    metricas = calcular_metricas(reais, previstos, ["a", "b", "c"])

    assert metricas["acuracia"] == 3 / 6
    assert metricas["matriz_confusao"] == [[1, 1, 0], [2, 1, 0], [0, 0, 1]]
    assert metricas["confusoes_mais_frequentes"][0] == {
        "real": "b",
        "previsto": "a",
        "quantidade": 2,
    }


def test_pipeline_completo_gera_relatorios(food101_falso, tmp_path):
    dir_relatorios = tmp_path / "reports"
    resultado = treinar(
        "cnn_simples",
        food101_falso.parent,
        epocas=2,
        tamanho_lote=16,
        num_workers=0,
        dir_relatorios=dir_relatorios,
        dir_artefatos=tmp_path / "artifacts",
    )

    dir_saida = dir_relatorios / "cnn_simples"
    for arquivo in ("metricas.json", "historico.csv", "curvas.png", "matriz_confusao.png"):
        assert (dir_saida / arquivo).exists()
    assert (tmp_path / "artifacts" / "cnn_simples.pt").exists()

    metricas = json.loads((dir_saida / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["melhor_epoca"] in (1, 2)
    assert len(metricas["teste"]["matriz_confusao"]) == len(CLASSES)
    assert resultado["teste"]["acuracia"] == metricas["teste"]["acuracia"]

    tabela = gerar_tabela(carregar_resultados(dir_relatorios))
    assert "CNN de 4 blocos" in tabela
