# Redes Neuronales II — Fundamentos de Deep Learning

Actividad individual sobre fundamentos de Deep Learning: Backpropagation, funciones de activación
(Sigmoide y ReLU), clasificación binaria y clasificación multiclase con TensorFlow/Keras y PyTorch.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `notebooks/01_Fundamentos_Perceptron_Backpropagation.ipynb` | Perceptrón simple, red de una capa (Sigmoide) y red multicapa con Backpropagation manual (Sigmoide vs ReLU). Dataset: Breast Cancer Wisconsin (clasificación binaria). |
| `notebooks/02_TensorFlow_Keras_MNIST.ipynb` | Clasificación multiclase (dígitos 0-9) con red densa en TensorFlow/Keras. Sin capas convolucionales. |
| `notebooks/03_PyTorch_MNIST.ipynb` | Clasificación multiclase (dígitos 0-9) con red densa en PyTorch. Sin capas convolucionales. |
| `scripts/` | Versión en `.py` de cada práctica, para ejecución local. |
| `Documento_Tecnico_Redes_Neuronales_II.pdf` | Documento técnico con marco conceptual, código, resultados y evidencias de ejecución. |

## Cómo ejecutar

1. Abrir cada notebook en [Google Colab](https://colab.research.google.com/).
2. Ejecutar las celdas en orden. Los notebooks 2 y 3 descargan automáticamente el dataset MNIST
   (requieren conexión a internet, disponible por defecto en Colab).
3. Los notebooks 1 no requieren descargas externas: usan datos sintéticos (compuertas lógicas) y el
   dataset Breast Cancer Wisconsin incluido en `scikit-learn`.

## Resumen de resultados

| Modelo | Dataset | Exactitud (prueba) |
|---|---|---|
| Perceptrón (AND / OR) | Sintético | 100% |
| Red de una capa (Sigmoide) | Breast Cancer | 97.4% |
| Red multicapa - Sigmoide | Breast Cancer | 95.6% |
| Red multicapa - ReLU | Breast Cancer | 96.5% |
| TensorFlow/Keras (denso) | MNIST | ~97-98% (ejecutar en Colab) |
| PyTorch (denso) | MNIST | ~97-98% (ejecutar en Colab) |

## Autor

Jerson Baquero Cháves
