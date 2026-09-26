"""
04 - Clasificación multiclase con TensorFlow + Keras sobre el dataset MNIST
IMPORTANTE: Solo capas densas (Dense) - NO se usan redes convolucionales.
Ejecutar este script en Google Colab (requiere descargar MNIST de internet).
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Versión de TensorFlow:", tf.__version__)

# 1. Cargar y preparar los datos
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalizar los píxeles al rango [0, 1] y aplanar las imágenes 28x28 -> 784
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0

print("Forma de X_train:", X_train.shape)
print("Forma de X_test:", X_test.shape)

# 2. Definir la arquitectura del modelo (perceptrón multicapa)
modelo = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation="relu", name="capa_oculta_1"),
    layers.Dense(64, activation="relu", name="capa_oculta_2"),
    layers.Dense(10, activation="softmax", name="capa_salida"),
])

modelo.summary()

# 3. Compilar el modelo
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 4. Entrenar el modelo (backpropagation gestionado internamente por Keras)
historial = modelo.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2,
)

# 5. Evaluar el modelo sobre el conjunto de prueba
perdida_test, exactitud_test = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\nPérdida en prueba: {perdida_test:.4f}")
print(f"Exactitud en prueba: {exactitud_test:.4f}")

# 6. Matriz de confusión y reporte
from sklearn.metrics import confusion_matrix, classification_report

y_pred = np.argmax(modelo.predict(X_test), axis=1)
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# 7. Gráficas de entrenamiento (evidencia de ejecución)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(historial.history["accuracy"], label="entrenamiento")
axs[0].plot(historial.history["val_accuracy"], label="validación")
axs[0].set_title("Exactitud por época - Keras/MNIST")
axs[0].set_xlabel("Época")
axs[0].set_ylabel("Exactitud")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(historial.history["loss"], label="entrenamiento")
axs[1].plot(historial.history["val_loss"], label="validación")
axs[1].set_title("Pérdida por época - Keras/MNIST")
axs[1].set_xlabel("Época")
axs[1].set_ylabel("Pérdida")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("evidencia_04_tensorflow_keras_mnist.png", dpi=150)
print("\nGráfica guardada como evidencia_04_tensorflow_keras_mnist.png")

# 8. Guardar el modelo entrenado (opcional, útil para el repositorio de GitHub)
modelo.save("modelo_keras_mnist.h5")
print("Modelo guardado como modelo_keras_mnist.h5")
