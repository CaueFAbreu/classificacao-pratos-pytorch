from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def calcular_metricas(
    reais: list[int], previstos: list[int], nomes_classes: list[str], top_confusoes: int = 5
) -> dict:
    indices = list(range(len(nomes_classes)))
    matriz = confusion_matrix(reais, previstos, labels=indices)

    fora_diagonal = matriz.copy()
    np.fill_diagonal(fora_diagonal, 0)
    pares = np.dstack(np.unravel_index(np.argsort(fora_diagonal, axis=None)[::-1], matriz.shape))[0]
    confusoes = [
        {"real": nomes_classes[i], "previsto": nomes_classes[j], "quantidade": int(matriz[i, j])}
        for i, j in pares[:top_confusoes]
        if matriz[i, j] > 0
    ]

    return {
        "acuracia": float(accuracy_score(reais, previstos)),
        "f1_macro": float(f1_score(reais, previstos, labels=indices, average="macro")),
        "por_classe": classification_report(
            reais,
            previstos,
            labels=indices,
            target_names=nomes_classes,
            output_dict=True,
            zero_division=0,
        ),
        "matriz_confusao": matriz.tolist(),
        "confusoes_mais_frequentes": confusoes,
    }


def salvar_matriz_confusao(
    matriz: list[list[int]], nomes_classes: list[str], caminho: Path
) -> None:
    figura, eixo = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(np.array(matriz), display_labels=nomes_classes).plot(
        ax=eixo, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)


def salvar_curvas(historico: list[dict], caminho: Path) -> None:
    epocas = [linha["epoca"] for linha in historico]
    figura, (eixo_perda, eixo_acuracia) = plt.subplots(1, 2, figsize=(11, 4))
    for particao in ("treino", "validacao"):
        eixo_perda.plot(epocas, [linha[f"perda_{particao}"] for linha in historico], label=particao)
        eixo_acuracia.plot(
            epocas, [linha[f"acuracia_{particao}"] for linha in historico], label=particao
        )
    eixo_perda.set(title="Perda", xlabel="Época")
    eixo_acuracia.set(title="Acurácia", xlabel="Época")
    for eixo in (eixo_perda, eixo_acuracia):
        eixo.legend()
        eixo.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
