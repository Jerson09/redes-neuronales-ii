"""
01 - Perceptrón simple (regla de Rosenblatt)
Clasificación binaria: compuertas lógicas AND y OR
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, lr=0.1, epochs=20):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.historial_errores = []

    def activacion_escalon(self, z):
        return np.where(z >= 0, 1, 0)

    def predict(self, X):
        z = X @ self.w + self.b
        return self.activacion_escalon(z)

    def fit(self, X, y):
        for epoca in range(self.epochs):
            errores = 0
            for xi, yi in zip(X, y):
                y_pred = self.predict(xi.reshape(1, -1))[0]
                error = yi - y_pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errores += 1
            self.historial_errores.append(errores)
        return self


if __name__ == "__main__":
    # Dataset: compuerta AND
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])

    perceptron_and = Perceptron(n_inputs=2, lr=0.1, epochs=10)
    perceptron_and.fit(X_and, y_and)
    pred_and = perceptron_and.predict(X_and)

    print("=== Perceptrón - Compuerta AND ===")
    print("Pesos finales:", perceptron_and.w, "| Bias:", perceptron_and.b)
    print("Predicciones:", pred_and, "| Reales:", y_and)
    print("Exactitud:", np.mean(pred_and == y_and))
    print("Errores por época:", perceptron_and.historial_errores)

    # Dataset: compuerta OR
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])

    perceptron_or = Perceptron(n_inputs=2, lr=0.1, epochs=10)
    perceptron_or.fit(X_or, y_or)
    pred_or = perceptron_or.predict(X_or)

    print("\n=== Perceptrón - Compuerta OR ===")
    print("Pesos finales:", perceptron_or.w, "| Bias:", perceptron_or.b)
    print("Predicciones:", pred_or, "| Reales:", y_or)
    print("Exactitud:", np.mean(pred_or == y_or))
    print("Errores por época:", perceptron_or.historial_errores)

    # Gráfica de convergencia
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(range(1, len(perceptron_and.historial_errores) + 1), perceptron_and.historial_errores, marker="o")
    axs[0].set_title("Convergencia Perceptrón - AND")
    axs[0].set_xlabel("Época")
    axs[0].set_ylabel("Errores de clasificación")
    axs[0].grid(True)

    axs[1].plot(range(1, len(perceptron_or.historial_errores) + 1), perceptron_or.historial_errores, marker="o", color="orange")
    axs[1].set_title("Convergencia Perceptrón - OR")
    axs[1].set_xlabel("Época")
    axs[1].set_ylabel("Errores de clasificación")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("/home/claude/proyecto/evidencia_01_perceptron.png", dpi=150)
    print("\nGráfica guardada como evidencia_01_perceptron.png")
