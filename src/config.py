from pathlib import Path

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
DIR_RELATORIOS = RAIZ_PROJETO / "reports"
DIR_ARTEFATOS = RAIZ_PROJETO / "artifacts"

# A ordem define o índice de cada classe.
CLASSES = (
    "chicken_wings",
    "donuts",
    "french_fries",
    "fried_rice",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "lasagna",
    "pizza",
    "sushi",
)
NOMES_EXIBICAO = {
    "chicken_wings": "Asinha de frango",
    "donuts": "Donut",
    "french_fries": "Batata frita",
    "fried_rice": "Arroz frito",
    "hamburger": "Hambúrguer",
    "hot_dog": "Cachorro-quente",
    "ice_cream": "Sorvete",
    "lasagna": "Lasanha",
    "pizza": "Pizza",
    "sushi": "Sushi",
}

TAMANHO_IMAGEM = 224
# Estatísticas do ImageNet, exigidas pelos pesos pré-treinados da ResNet18.
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

SEMENTE = 42
FRACAO_VALIDACAO = 0.15
TAMANHO_LOTE = 64

EXPERIMENTOS = {
    "cnn_simples": {
        "descricao": "CNN de 4 blocos treinada do zero",
        "epocas": 20,
        "taxa_aprendizado": 1e-3,
    },
    "resnet18_congelada": {
        "descricao": "ResNet18 pré-treinada (ImageNet) com a base congelada",
        "epocas": 8,
        "taxa_aprendizado": 1e-3,
    },
    "resnet18_ajuste_fino": {
        "descricao": "ResNet18 pré-treinada (ImageNet) com ajuste fino de todas as camadas",
        "epocas": 8,
        "taxa_aprendizado": 1e-4,
    },
}
