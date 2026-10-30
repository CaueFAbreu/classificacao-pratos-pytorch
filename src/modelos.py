from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNSimples(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        canais = (3, 32, 64, 128, 256)
        blocos = []
        for entrada, saida in zip(canais, canais[1:], strict=False):
            blocos += [
                nn.Conv2d(entrada, saida, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(saida),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.extrator = nn.Sequential(*blocos)
        self.classificador = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(canais[-1], num_classes),
        )

    def forward(self, x):
        return self.classificador(self.extrator(x))


def criar_resnet18(
    num_classes: int, congelar_base: bool, pre_treinada: bool = True, dropout: float = 0.3
) -> nn.Module:
    modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pre_treinada else None)
    if congelar_base:
        for parametro in modelo.parameters():
            parametro.requires_grad = False
    # A nova cabeça é criada depois do congelamento, então continua treinável.
    modelo.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(modelo.fc.in_features, num_classes))
    return modelo


def criar_modelo(nome: str, num_classes: int, pre_treinada: bool = True) -> nn.Module:
    if nome == "cnn_simples":
        return CNNSimples(num_classes)
    if nome == "resnet18_congelada":
        return criar_resnet18(num_classes, congelar_base=True, pre_treinada=pre_treinada)
    if nome == "resnet18_ajuste_fino":
        return criar_resnet18(num_classes, congelar_base=False, pre_treinada=pre_treinada)
    raise ValueError(f"Modelo desconhecido: {nome}")


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in modelo.parameters())
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, treinaveis
