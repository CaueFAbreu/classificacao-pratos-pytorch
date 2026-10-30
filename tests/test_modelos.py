import pytest
import torch

from src.modelos import contar_parametros, criar_modelo

NUM_CLASSES = 10


@pytest.mark.parametrize("nome", ["cnn_simples", "resnet18_congelada", "resnet18_ajuste_fino"])
def test_saida_tem_uma_coluna_por_classe(nome):
    modelo = criar_modelo(nome, NUM_CLASSES, pre_treinada=False).eval()
    with torch.no_grad():
        saida = modelo(torch.randn(2, 3, 224, 224))
    assert saida.shape == (2, NUM_CLASSES)


def test_resnet_congelada_treina_apenas_a_cabeca():
    modelo = criar_modelo("resnet18_congelada", NUM_CLASSES, pre_treinada=False)
    treinaveis = {nome for nome, p in modelo.named_parameters() if p.requires_grad}

    assert treinaveis == {"fc.1.weight", "fc.1.bias"}
    _, parametros_treinaveis = contar_parametros(modelo)
    assert parametros_treinaveis == 512 * NUM_CLASSES + NUM_CLASSES


def test_ajuste_fino_treina_todas_as_camadas():
    modelo = criar_modelo("resnet18_ajuste_fino", NUM_CLASSES, pre_treinada=False)
    total, treinaveis = contar_parametros(modelo)
    assert total == treinaveis


def test_modelo_desconhecido_gera_erro():
    with pytest.raises(ValueError, match="desconhecido"):
        criar_modelo("vgg16", NUM_CLASSES)
