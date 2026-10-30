import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Food101

from src.config import (
    CLASSES,
    DESVIO_IMAGENET,
    FRACAO_VALIDACAO,
    MEDIA_IMAGENET,
    SEMENTE,
    TAMANHO_IMAGEM,
    TAMANHO_LOTE,
)


def garantir_food101(dir_dados: Path) -> Path:
    dir_food101 = Path(dir_dados) / "food-101"
    if not (dir_food101 / "meta" / "train.json").exists():
        Food101(root=str(dir_dados), split="train", download=True)
    return dir_food101


def listar_amostras(
    dir_food101: Path, particao: str, classes: tuple[str, ...] = CLASSES
) -> list[tuple[Path, int]]:
    caminho_meta = Path(dir_food101) / "meta" / f"{particao}.json"
    meta = json.loads(caminho_meta.read_text(encoding="utf-8"))
    return [
        (Path(dir_food101) / "images" / f"{item}.jpg", indice)
        for indice, classe in enumerate(classes)
        for item in meta[classe]
    ]


def dividir_treino_validacao(
    amostras: list[tuple[Path, int]],
    fracao_validacao: float = FRACAO_VALIDACAO,
    semente: int = SEMENTE,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    rotulos = [rotulo for _, rotulo in amostras]
    return train_test_split(
        amostras, test_size=fracao_validacao, stratify=rotulos, random_state=semente
    )


def criar_transformacoes(treino: bool) -> transforms.Compose:
    normalizacao = transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET)
    if treino:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(TAMANHO_IMAGEM, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalizacao,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(TAMANHO_IMAGEM),
            transforms.ToTensor(),
            normalizacao,
        ]
    )


class DatasetPratos(Dataset):
    def __init__(self, amostras: list[tuple[Path, int]], transformacao: transforms.Compose):
        self.amostras = amostras
        self.transformacao = transformacao

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int) -> tuple[torch.Tensor, int]:
        caminho, rotulo = self.amostras[indice]
        with Image.open(caminho) as imagem:
            imagem = imagem.convert("RGB")
        return self.transformacao(imagem), rotulo


def criar_dataloaders(
    dir_food101: Path,
    tamanho_lote: int = TAMANHO_LOTE,
    num_workers: int = 2,
    semente: int = SEMENTE,
) -> dict[str, DataLoader]:
    treino, validacao = dividir_treino_validacao(
        listar_amostras(dir_food101, "train"), semente=semente
    )
    teste = listar_amostras(dir_food101, "test")
    usar_pin_memory = torch.cuda.is_available()

    def carregador(amostras, treino_ativo):
        return DataLoader(
            DatasetPratos(amostras, criar_transformacoes(treino_ativo)),
            batch_size=tamanho_lote,
            shuffle=treino_ativo,
            num_workers=num_workers,
            pin_memory=usar_pin_memory,
            generator=torch.Generator().manual_seed(semente) if treino_ativo else None,
        )

    return {
        "treino": carregador(treino, True),
        "validacao": carregador(validacao, False),
        "teste": carregador(teste, False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baixa o Food-101 e mostra o tamanho de cada partição."
    )
    parser.add_argument("--dados", type=Path, required=True, help="Pasta onde o dataset será salvo")
    args = parser.parse_args()

    dir_food101 = garantir_food101(args.dados)
    treino, validacao = dividir_treino_validacao(listar_amostras(dir_food101, "train"))
    teste = listar_amostras(dir_food101, "test")
    print(f"Classes: {len(CLASSES)}")
    print(f"Treino: {len(treino)} | Validação: {len(validacao)} | Teste: {len(teste)}")


if __name__ == "__main__":
    main()
