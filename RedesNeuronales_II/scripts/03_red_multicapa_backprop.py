"""
03 - Red neuronal multicapa (1 capa oculta) con Backpropagation implementado desde cero
Comparación de funciones de activación: Sigmoide vs ReLU
Clasificación binaria sobre el dataset real Breast Cancer Wisconsin (sklearn)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


# ---------- Funciones de activación y sus derivadas ----------
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def sigmoide_deriv(a):
    # a = sigmoide(z) ya calculado (derivada en función de la salida)
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)


class RedMulticapa:
    """
    Arquitectura: entrada -> capa oculta (activación configurable) -> salida (sigmoide)
    Entrenada con backpropagation y descenso de gradiente.
    """
    def __init__(self, n_entradas, n_ocultas, activacion="sigmoide", lr=0.05, epochs=1000, seed=42):
        rng = np.random.default_rng(seed)
        # Inicialización He/Xavier simplificada
        self.W1 = rng.normal(0, np.sqrt(2 / n_entradas), (n_entradas, n_ocultas))
        self.b1 = np.zeros(n_ocultas)
        self.W2 = rng.normal(0, np.sqrt(2 / n_ocultas), (n_ocultas, 1))
        self.b2 = np.zeros(1)
        self.activacion = activacion
        self.lr = lr
        self.epochs = epochs
        self.historial_perdida = []

    def _activar_oculta(self, z):
        return sigmoide(z) if self.activacion == "sigmoide" else relu(z)

    def _derivar_oculta(self, z, a):
        return sigmoide_deriv(a) if self.activacion == "sigmoide" else relu_deriv(z)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._activar_oculta(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoide(self.z2).ravel()
        return self.a2

    def backward(self, X, y):
        n = X.shape[0]
        eps = 1e-9

        # --- Forward (guardado en self) ---
        y_hat = self.forward(X)
        perdida = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

        # --- Backpropagation ---
        # Capa de salida: dL/dz2 = (y_hat - y)  [derivada conjunta BCE + sigmoide]
        dz2 = (y_hat - y).reshape(-1, 1)
        dW2 = self.a1.T @ dz2 / n
        db2 = np.mean(dz2, axis=0)

        # Propagar el error hacia la capa oculta
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self._derivar_oculta(self.z1, self.a1)
        dW1 = X.T @ dz1 / n
        db1 = np.mean(dz1, axis=0)

        # --- Actualización de pesos (descenso de gradiente) ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return perdida

    def fit(self, X, y):
        for epoca in range(self.epochs):
            perdida = self.backward(X, y)
            self.historial_perdida.append(perdida)
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

    resultados = {}
    plt.figure(figsize=(7, 5))

    for act in ["sigmoide", "relu"]:
        modelo = RedMulticapa(
            n_entradas=X.shape[1], n_ocultas=16, activacion=act, lr=0.1, epochs=800
        )
        modelo.fit(X_train, y_train)

        pred_train = modelo.predict(X_train)
        pred_test = modelo.predict(X_test)
        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)
        cm = confusion_matrix(y_test, pred_test)

        resultados[act] = {"acc_train": acc_train, "acc_test": acc_test, "cm": cm}

        print(f"=== Red multicapa - Activación: {act.upper()} ===")
        print(f"Exactitud entrenamiento: {acc_train:.4f}")
        print(f"Exactitud prueba: {acc_test:.4f}")
        print("Matriz de confusión (prueba):\n", cm)
        print()

        plt.plot(modelo.historial_perdida, label=f"Oculta: {act}")

    plt.title("Curva de pérdida - Red multicapa (Backpropagation)\nSigmoide vs ReLU")
    plt.xlabel("Época")
    plt.ylabel("Entropía cruzada binaria")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/claude/proyecto/evidencia_03_red_multicapa.png", dpi=150)
    print("Gráfica guardada como evidencia_03_red_multicapa.png")

    print("\n=== Resumen comparativo ===")
    for act, r in resultados.items():
        print(f"{act.upper():10s} -> Exactitud prueba: {r['acc_test']:.4f}")
