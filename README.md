# Classificação de Pratos com PyTorch

[![CI](https://github.com/CaueFAbreu/classificacao-pratos-pytorch/actions/workflows/ci.yml/badge.svg)](https://github.com/CaueFAbreu/classificacao-pratos-pytorch/actions/workflows/ci.yml)

Classificação de imagens de 10 pratos comuns em pedidos de delivery a partir do dataset Food-101. O projeto compara uma CNN treinada do zero com duas formas de transfer learning da ResNet18 (base congelada e ajuste fino), sob o mesmo protocolo de avaliação.

> **Status:** código e testes concluídos. Os resultados serão publicados em `reports/` após o treinamento em GPU.

## Stack

Python · PyTorch · torchvision · scikit-learn · matplotlib · pytest · Ruff · GitHub Actions · Google Colab

## Resultados

_Em breve._

## Dados

- **Fonte:** [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard et al., 2014), baixado automaticamente pelo `torchvision`.
- **Classes (10):** asinha de frango, donut, batata frita, arroz frito, hambúrguer, cachorro-quente, sorvete, lasanha, pizza e sushi.
- **Partições:** 750 imagens de treino e 250 de teste por classe, conforme a divisão oficial do dataset. 15% do treino é separado, de forma estratificada, para validação.

| Partição | Imagens | Uso |
|---|---:|---|
| Treino | 6.375 | Ajuste dos pesos |
| Validação | 1.125 | Escolha da melhor época |
| Teste | 2.500 | Avaliação final, usada uma única vez |

Segundo os autores, as imagens de treino do Food-101 contêm algum ruído de rótulo, e as de teste foram revisadas manualmente.

## Metodologia

| Experimento | Modelo | Parâmetros treinados | Épocas | Taxa de aprendizado |
|---|---|---|---:|---:|
| `cnn_simples` | CNN de 4 blocos convolucionais, treinada do zero | Todos | 20 | 1e-3 |
| `resnet18_congelada` | ResNet18 pré-treinada no ImageNet, base congelada | Somente a última camada | 8 | 1e-3 |
| `resnet18_ajuste_fino` | ResNet18 pré-treinada no ImageNet | Todos | 8 | 1e-4 |

Decisões comuns aos três experimentos:

- **Aumento de dados apenas no treino:** recorte aleatório, espelhamento horizontal e variação de cor. Validação e teste usam um recorte central fixo, para que a avaliação não dependa de sorteio.
- **Normalização com as estatísticas do ImageNet,** exigida pelos pesos pré-treinados.
- **Otimização:** AdamW com decaimento de peso de 1e-4, agenda de taxa de aprendizado em cosseno e precisão mista na GPU.
- **Seleção do modelo pela validação:** o checkpoint da época com maior acurácia de validação é o único avaliado no teste.
- **Reprodutibilidade:** semente fixa para a divisão dos dados, a ordem dos lotes e a inicialização dos pesos.
- **Métricas:** acurácia, F1 macro, precisão e recall por classe, matriz de confusão e os pares de classes mais confundidos.

## Como reproduzir

### No Google Colab (GPU)

Abra [`notebooks/treinamento_colab.ipynb`](notebooks/treinamento_colab.ipynb) no Colab, selecione uma GPU e execute todas as células. O notebook baixa o dataset, treina os três modelos e gera os relatórios.

### Localmente

Pré-requisitos: Python 3.13 e Git. O treino em CPU é possível, mas lento.

```bash
git clone https://github.com/CaueFAbreu/classificacao-pratos-pytorch.git
cd classificacao-pratos-pytorch
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements-dev.txt
```

`requirements.txt` contém apenas as dependências de execução; `requirements-dev.txt` acrescenta as de teste e lint.

```bash
python -m src.dados --dados data                                  # baixa o Food-101 (~5 GB)
python -m src.treino --modelo resnet18_ajuste_fino --dados data   # treina e avalia um modelo
python -m src.relatorio                                           # gera a tabela comparativa
```

## Verificações

Os testes usam um Food-101 falso e pequeno, gerado na hora, e rodam em CPU em poucos segundos. São os mesmos comandos executados pela integração contínua:

```bash
ruff check .
ruff format --check .
pytest
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `src/config.py` | Classes, hiperparâmetros e configuração dos experimentos |
| `src/dados.py` | Download, seleção das classes, divisão e DataLoaders |
| `src/modelos.py` | CNN simples e variações da ResNet18 |
| `src/treino.py` | Laço de treino, escolha do checkpoint e avaliação no teste |
| `src/avaliacao.py` | Métricas e gráficos |
| `src/relatorio.py` | Tabela e gráfico comparativos |
| `tests/` | Testes de dados, modelos e do pipeline completo |
| `notebooks/` | Execução no Google Colab |
| `reports/` | Métricas, curvas de treino e matrizes de confusão |

## Referência

Bossard, L., Guillaumin, M. e Van Gool, L. *Food-101: Mining Discriminative Components with Random Forests.* European Conference on Computer Vision (ECCV), 2014.
