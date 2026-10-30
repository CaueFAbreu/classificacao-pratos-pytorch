from collections import Counter

import torch

from src.config import CLASSES, TAMANHO_IMAGEM
from src.dados import (
    DatasetPratos,
    criar_dataloaders,
    criar_transformacoes,
    dividir_treino_validacao,
    listar_amostras,
)
from tests.conftest import IMAGENS_TESTE_POR_CLASSE, IMAGENS_TREINO_POR_CLASSE


def test_listar_amostras_le_apenas_a_particao_pedida(food101_falso):
    treino = listar_amostras(food101_falso, "train")
    teste = listar_amostras(food101_falso, "test")

    assert len(treino) == len(CLASSES) * IMAGENS_TREINO_POR_CLASSE
    assert len(teste) == len(CLASSES) * IMAGENS_TESTE_POR_CLASSE
    assert all(caminho.exists() for caminho, _ in treino + teste)
    assert {caminho for caminho, _ in treino}.isdisjoint({caminho for caminho, _ in teste})


def test_rotulo_corresponde_a_pasta_da_classe(food101_falso):
    for caminho, rotulo in listar_amostras(food101_falso, "train"):
        assert caminho.parent.name == CLASSES[rotulo]


def test_divisao_e_estratificada_e_sem_sobreposicao(food101_falso):
    amostras = listar_amostras(food101_falso, "train")
    treino, validacao = dividir_treino_validacao(amostras, fracao_validacao=0.25)

    assert len(treino) + len(validacao) == len(amostras)
    assert set(treino).isdisjoint(validacao)
    contagem = Counter(rotulo for _, rotulo in validacao)
    assert set(contagem.values()) == {IMAGENS_TREINO_POR_CLASSE // 4}


def test_divisao_e_reproduzivel(food101_falso):
    amostras = listar_amostras(food101_falso, "train")
    assert dividir_treino_validacao(amostras) == dividir_treino_validacao(amostras)


def test_dataset_devolve_tensor_normalizado_no_tamanho_esperado(food101_falso):
    amostras = listar_amostras(food101_falso, "test")
    imagem, rotulo = DatasetPratos(amostras, criar_transformacoes(treino=False))[0]

    assert imagem.shape == (3, TAMANHO_IMAGEM, TAMANHO_IMAGEM)
    assert imagem.dtype == torch.float32
    assert rotulo == amostras[0][1]


def test_dataloaders_cobrem_as_tres_particoes(food101_falso):
    carregadores = criar_dataloaders(food101_falso, tamanho_lote=16, num_workers=0)
    imagens, rotulos = next(iter(carregadores["treino"]))

    assert set(carregadores) == {"treino", "validacao", "teste"}
    assert imagens.shape[1:] == (3, TAMANHO_IMAGEM, TAMANHO_IMAGEM)
    assert len(carregadores["teste"].dataset) == len(CLASSES) * IMAGENS_TESTE_POR_CLASSE
