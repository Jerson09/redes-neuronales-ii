"""
05 - Clasificación multiclase con PyTorch sobre el dataset MNIST
IMPORTANTE: Solo capas densas (Linear) - NO se usan redes convolucionales.
Ejecutar este script en Google Colab (requiere descargar MNIST de internet).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

print("Versión de PyTorch:", torch.__version__)
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", dispositivo)

# 1. Cargar y preparar los datos
transformacion = transforms.Compose([
    transforms.ToTensor(),                      # escala a [0, 1]
    transforms.Normalize((0.1307,), (0.3081,)), # normalización estándar de MNIST
    transforms.Lambda(lambda x: x.view(-1)),    # aplanar 28x28 -> 784
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transformacion)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transformacion)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# 2. Definir la arquitectura del modelo (perceptrón multicapa)
class RedMulticapaMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa_oculta_1 = nn.Linear(28 * 28, 128)
        self.capa_oculta_2 = nn.Linear(128, 64)
        self.capa_salida = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.capa_oculta_1(x))
        x = self.relu(self.capa_oculta_2(x))
        x = self.capa_salida(x)  # logits; softmax se aplica dentro de la función de pérdida
        return x


modelo = RedMulticapaMNIST().to(dispositivo)
print(modelo)

# 3. Definir función de pérdida y optimizador
criterio = nn.CrossEntropyLoss()          # incluye softmax internamente
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# 4. Entrenamiento (backpropagation con autograd de PyTorch)
n_epochs = 10
historial_perdida = []
historial_exactitud = []

for epoca in range(n_epochs):
    modelo.train()
    perdida_epoca = 0.0
    correctos = 0
    total = 0

    for X_lote, y_lote in train_loader:
        X_lote, y_lote = X_lote.to(dispositivo), y_lote.to(dispositivo)

        optimizador.zero_grad()
        salidas = modelo(X_lote)
        perdida = criterio(salidas, y_lote)
        perdida.backward()          # backpropagation automático
        optimizador.step()          # actualización de pesos

        perdida_epoca += perdida.item() * X_lote.size(0)
        correctos += (salidas.argmax(dim=1) == y_lote).sum().item()
        total += y_lote.size(0)

    perdida_prom = perdida_epoca / total
    exactitud = correctos / total
    historial_perdida.append(perdida_prom)
    historial_exactitud.append(exactitud)
    print(f"Época {epoca+1}/{n_epochs} - Pérdida: {perdida_prom:.4f} - Exactitud: {exactitud:.4f}")

# 5. Evaluación sobre el conjunto de prueba
modelo.eval()
todas_predicciones, todas_reales = [], []
with torch.no_grad():
    correctos, total = 0, 0
    for X_lote, y_lote in test_loader:
        X_lote, y_lote = X_lote.to(dispositivo), y_lote.to(dispositivo)
        salidas = modelo(X_lote)
        predicciones = salidas.argmax(dim=1)
        correctos += (predicciones == y_lote).sum().item()
        total += y_lote.size(0)
        todas_predicciones.extend(predicciones.cpu().numpy())
        todas_reales.extend(y_lote.cpu().numpy())

print(f"\nExactitud en prueba: {correctos/total:.4f}")
print("\nMatriz de confusión:\n", confusion_matrix(todas_reales, todas_predicciones))
print("\nReporte de clasificación:\n", classification_report(todas_reales, todas_predicciones))

# 6. Gráficas de entrenamiento (evidencia de ejecución)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(historial_exactitud, marker="o")
axs[0].set_title("Exactitud por época - PyTorch/MNIST")
axs[0].set_xlabel("Época")
axs[0].set_ylabel("Exactitud (entrenamiento)")
axs[0].grid(True)

axs[1].plot(historial_perdida, marker="o", color="orange")
axs[1].set_title("Pérdida por época - PyTorch/MNIST")
axs[1].set_xlabel("Época")
axs[1].set_ylabel("Pérdida (entrenamiento)")
axs[1].grid(True)

plt.tight_layout()
plt.savefig("evidencia_05_pytorch_mnist.png", dpi=150)
print("\nGráfica guardada como evidencia_05_pytorch_mnist.png")

# 7. Guardar el modelo entrenado (opcional, útil para el repositorio de GitHub)
torch.save(modelo.state_dict(), "modelo_pytorch_mnist.pt")
print("Modelo guardado como modelo_pytorch_mnist.pt")
