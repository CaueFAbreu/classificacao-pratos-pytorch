| Modelo | Parâmetros treináveis | Épocas (melhor) | Tempo de treino | Acurácia (teste) | F1 macro (teste) |
|---|---:|---:|---:|---:|---:|
| CNN de 4 blocos treinada do zero | 391.466 | 20 (20) | 14,6 min | 68,6% | 0,681 |
| ResNet18 pré-treinada (ImageNet) com a base congelada | 5.130 | 8 (7) | 5,6 min | 86,3% | 0,863 |
| ResNet18 pré-treinada (ImageNet) com ajuste fino de todas as camadas | 11.181.642 | 8 (8) | 5,9 min | 94,2% | 0,942 |
