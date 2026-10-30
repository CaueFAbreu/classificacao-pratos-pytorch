import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import DIR_RELATORIOS, EXPERIMENTOS  # noqa: E402


def carregar_resultados(dir_relatorios: Path = DIR_RELATORIOS) -> list[dict]:
    resultados = []
    for nome in EXPERIMENTOS:
        caminho = Path(dir_relatorios) / nome / "metricas.json"
        if caminho.exists():
            resultados.append(json.loads(caminho.read_text(encoding="utf-8")))
    return resultados


def formatar_milhar(numero: int) -> str:
    return f"{numero:,}".replace(",", ".")


def formatar_decimal(numero: float, casas: int) -> str:
    return f"{numero:.{casas}f}".replace(".", ",")


def gerar_tabela(resultados: list[dict]) -> str:
    linhas = [
        "| Modelo | Parâmetros treináveis | Épocas (melhor) | Tempo de treino | "
        "Acurácia (teste) | F1 macro (teste) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in resultados:
        linhas.append(
            f"| {r['descricao']} | {formatar_milhar(r['parametros_treinaveis'])} | "
            f"{r['epocas']} ({r['melhor_epoca']}) | "
            f"{formatar_decimal(r['tempo_treino_segundos'] / 60, 1)} min | "
            f"{formatar_decimal(r['teste']['acuracia'] * 100, 1)}% | "
            f"{formatar_decimal(r['teste']['f1_macro'], 3)} |"
        )
    return "\n".join(linhas)


def gerar_grafico(resultados: list[dict], caminho: Path) -> None:
    nomes = [r["modelo"] for r in resultados]
    acuracias = [r["teste"]["acuracia"] for r in resultados]
    figura, eixo = plt.subplots(figsize=(7, 4))
    barras = eixo.bar(nomes, acuracias, color="#2f6fb0")
    eixo.bar_label(
        barras, labels=[f"{formatar_decimal(a * 100, 1)}%" for a in acuracias], padding=3
    )
    eixo.set(ylim=(0, 1), ylabel="Acurácia no teste", title="Comparação dos modelos")
    eixo.grid(axis="y", alpha=0.3)
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)


def main() -> None:
    resultados = carregar_resultados()
    if not resultados:
        raise SystemExit("Nenhum resultado em reports/. Rode src.treino antes.")
    tabela = gerar_tabela(resultados)
    (DIR_RELATORIOS / "comparacao.md").write_text(tabela + "\n", encoding="utf-8")
    gerar_grafico(resultados, DIR_RELATORIOS / "comparacao.png")
    print(tabela)


if __name__ == "__main__":
    main()
