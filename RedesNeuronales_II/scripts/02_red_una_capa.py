"""
02 - Red neuronal de una capa (una sola neurona de salida, activación Sigmoide)
Clasificación binaria sobre el dataset real Breast Cancer Wisconsin (sklearn)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


class RedUnaCapa:
    def __init__(self, n_entradas, lr=0.05, epochs=300):
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 0.01, n_entradas)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.historial_perdida = []

    def forward(self, X):
        z = X @ self.w + self.b
        return sigmoide(z)

    def fit(self, X, y):
        n = X.shape[0]
        for epoca in range(self.epochs):
            y_hat = self.forward(X)
            # Entropía cruzada binaria
            eps = 1e-9
            perdida = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
            self.historial_perdida.append(perdida)

            # Gradientes (backpropagation de una sola capa)
            error = y_hat - y
            grad_w = (X.T @ error) / n
            grad_b = np.mean(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict(self, X, umbral=0.5):
        return (self.forward(X) >= umbral).astype(int)


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    escalador = StandardScaler()
    X_train = escalador.fit_transform(X_train)
    X_test = escalador.transform(X_test)

    modelo = RedUnaCapa(n_entradas=X.shape[1], lr=0.1, epochs=500)
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_test = modelo.predict(X_test)

    print("=== Red de una capa (Sigmoide) - Breast Cancer Dataset ===")
    print("Exactitud entrenamiento:", accuracy_score(y_train, pred_train))
    print("Exactitud prueba:", accuracy_score(y_test, pred_test))
    print("\nMatriz de confusión (prueba):\n", confusion_matrix(y_test, pred_test))
    print("\nReporte de clasificación (prueba):\n", classification_report(y_test, pred_test))

    plt.figure(figsize=(6, 4))
    plt.plot(modelo.historial_perdida)
    plt.title("Curva de pérdida - Red de una capa (Sigmoide)")
    plt.xlabel("Época")
    plt.ylabel("Entropía cruzada binaria")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/claude/proyecto/evidencia_02_red_una_capa.png", dpi=150)
    print("\nGráfica guardada como evidencia_02_red_una_capa.png")
