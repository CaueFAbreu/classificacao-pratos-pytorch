import json

import numpy as np
import pytest
from PIL import Image

from src.config import CLASSES

IMAGENS_TREINO_POR_CLASSE = 8
IMAGENS_TESTE_POR_CLASSE = 2


@pytest.fixture
def food101_falso(tmp_path):
    dir_food101 = tmp_path / "food-101"
    (dir_food101 / "meta").mkdir(parents=True)
    gerador = np.random.default_rng(0)
    meta = {"train": {}, "test": {}}

    for classe in CLASSES:
        (dir_food101 / "images" / classe).mkdir(parents=True)
        quantidades = {"train": IMAGENS_TREINO_POR_CLASSE, "test": IMAGENS_TESTE_POR_CLASSE}
        contador = 0
        for particao, quantidade in quantidades.items():
            meta[particao][classe] = []
            for _ in range(quantidade):
                item = f"{classe}/{contador}"
                pixels = gerador.integers(0, 256, size=(40, 48, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(dir_food101 / "images" / f"{item}.jpg")
                meta[particao][classe].append(item)
                contador += 1

    for particao, conteudo in meta.items():
        (dir_food101 / "meta" / f"{particao}.json").write_text(json.dumps(conteudo))
    return dir_food101
