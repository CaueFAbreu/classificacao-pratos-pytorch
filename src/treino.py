import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.avaliacao import calcular_metricas, salvar_curvas, salvar_matriz_confusao
from src.config import (
    CLASSES,
    DIR_ARTEFATOS,
    DIR_RELATORIOS,
    EXPERIMENTOS,
    NOMES_EXIBICAO,
    SEMENTE,
    TAMANHO_LOTE,
)
from src.dados import criar_dataloaders, garantir_food101
from src.modelos import contar_parametros, criar_modelo


def definir_semente(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def executar_epoca(
    modelo: nn.Module,
    carregador: DataLoader,
    criterio: nn.Module,
    dispositivo: torch.device,
    otimizador: torch.optim.Optimizer | None = None,
    escalador: torch.amp.GradScaler | None = None,
) -> dict:
    treinando = otimizador is not None
    modelo.train(treinando)
    usar_amp = dispositivo.type == "cuda"
    perda_total, reais, previstos = 0.0, [], []

    with torch.set_grad_enabled(treinando):
        for imagens, rotulos in carregador:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)
            with torch.autocast(device_type=dispositivo.type, enabled=usar_amp):
                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)

            if treinando:
                otimizador.zero_grad(set_to_none=True)
                if escalador is not None:
                    escalador.scale(perda).backward()
                    escalador.step(otimizador)
                    escalador.update()
                else:
                    perda.backward()
                    otimizador.step()

            perda_total += perda.item() * rotulos.size(0)
            reais += rotulos.tolist()
            previstos += saidas.argmax(dim=1).tolist()

    acertos = sum(r == p for r, p in zip(reais, previstos, strict=True))
    return {
        "perda": perda_total / len(reais),
        "acuracia": acertos / len(reais),
        "reais": reais,
        "previstos": previstos,
    }


def treinar(
    nome_modelo: str,
    dir_dados: Path,
    epocas: int | None = None,
    taxa_aprendizado: float | None = None,
    tamanho_lote: int = TAMANHO_LOTE,
    num_workers: int = 2,
    semente: int = SEMENTE,
    pre_treinada: bool = True,
    dir_relatorios: Path = DIR_RELATORIOS,
    dir_artefatos: Path = DIR_ARTEFATOS,
) -> dict:
    experimento = EXPERIMENTOS[nome_modelo]
    epocas = epocas or experimento["epocas"]
    taxa_aprendizado = taxa_aprendizado or experimento["taxa_aprendizado"]

    definir_semente(semente)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modelo: {nome_modelo} | Dispositivo: {dispositivo}")

    carregadores = criar_dataloaders(
        garantir_food101(dir_dados), tamanho_lote, num_workers, semente
    )
    modelo = criar_modelo(nome_modelo, len(CLASSES), pre_treinada).to(dispositivo)
    total_parametros, parametros_treinaveis = contar_parametros(modelo)

    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(
        (p for p in modelo.parameters() if p.requires_grad),
        lr=taxa_aprendizado,
        weight_decay=1e-4,
    )
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=epocas)
    escalador = torch.amp.GradScaler("cuda") if dispositivo.type == "cuda" else None

    dir_saida = Path(dir_relatorios) / nome_modelo
    dir_saida.mkdir(parents=True, exist_ok=True)
    Path(dir_artefatos).mkdir(parents=True, exist_ok=True)
    caminho_checkpoint = Path(dir_artefatos) / f"{nome_modelo}.pt"

    historico, melhor_acuracia, melhor_epoca = [], -1.0, 0
    inicio = time.perf_counter()
    for epoca in range(1, epocas + 1):
        treino = executar_epoca(
            modelo, carregadores["treino"], criterio, dispositivo, otimizador, escalador
        )
        validacao = executar_epoca(modelo, carregadores["validacao"], criterio, dispositivo)
        agendador.step()

        historico.append(
            {
                "epoca": epoca,
                "perda_treino": treino["perda"],
                "acuracia_treino": treino["acuracia"],
                "perda_validacao": validacao["perda"],
                "acuracia_validacao": validacao["acuracia"],
            }
        )
        print(
            f"Época {epoca:>2}/{epocas} | "
            f"treino: perda {treino['perda']:.3f}, acurácia {treino['acuracia']:.3f} | "
            f"validação: perda {validacao['perda']:.3f}, acurácia {validacao['acuracia']:.3f}"
        )
        if validacao["acuracia"] > melhor_acuracia:
            melhor_acuracia, melhor_epoca = validacao["acuracia"], epoca
            torch.save(modelo.state_dict(), caminho_checkpoint)
    tempo_treino = time.perf_counter() - inicio

    # O teste é usado uma única vez, com o checkpoint escolhido pela validação.
    modelo.load_state_dict(torch.load(caminho_checkpoint, map_location=dispositivo))
    teste = executar_epoca(modelo, carregadores["teste"], criterio, dispositivo)
    nomes_classes = [NOMES_EXIBICAO[classe] for classe in CLASSES]
    metricas_teste = calcular_metricas(teste["reais"], teste["previstos"], nomes_classes)

    resultado = {
        "modelo": nome_modelo,
        "descricao": experimento["descricao"],
        "dispositivo": str(dispositivo),
        "epocas": epocas,
        "taxa_aprendizado": taxa_aprendizado,
        "tamanho_lote": tamanho_lote,
        "semente": semente,
        "parametros_totais": total_parametros,
        "parametros_treinaveis": parametros_treinaveis,
        "melhor_epoca": melhor_epoca,
        "melhor_acuracia_validacao": melhor_acuracia,
        "tempo_treino_segundos": round(tempo_treino, 1),
        "teste": metricas_teste,
    }

    (dir_saida / "metricas.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(dir_saida / "historico.csv", "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=historico[0].keys())
        escritor.writeheader()
        escritor.writerows(historico)
    salvar_curvas(historico, dir_saida / "curvas.png")
    salvar_matriz_confusao(
        metricas_teste["matriz_confusao"], nomes_classes, dir_saida / "matriz_confusao.png"
    )

    print(
        f"Teste: acurácia {metricas_teste['acuracia']:.3f} | "
        f"F1 macro {metricas_teste['f1_macro']:.3f}"
    )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina e avalia um dos modelos do projeto.")
    parser.add_argument("--modelo", choices=EXPERIMENTOS.keys(), required=True)
    parser.add_argument("--dados", type=Path, required=True, help="Pasta do Food-101")
    parser.add_argument("--epocas", type=int, help="Substitui o padrão do experimento")
    parser.add_argument("--taxa-aprendizado", type=float, help="Substitui o padrão do experimento")
    parser.add_argument("--tamanho-lote", type=int, default=TAMANHO_LOTE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--semente", type=int, default=SEMENTE)
    args = parser.parse_args()

    treinar(
        args.modelo,
        args.dados,
        epocas=args.epocas,
        taxa_aprendizado=args.taxa_aprendizado,
        tamanho_lote=args.tamanho_lote,
        num_workers=args.num_workers,
        semente=args.semente,
    )


if __name__ == "__main__":
    main()
