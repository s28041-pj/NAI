"""
NAI Project point 3

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad5/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

## Requirements
    - Python 3.13
    - numpy>=1.23, pandas>=1.5, scikit-learn>=1.2, tensorflow>=2.12, matplotlib>=3.7, protobuf>=3.20,<5

## How to run the script:

    - Install Python (>=3.13)
    - Clone this repository
    -pip install requirements.txt / numpy>=1.23, pandas>=1.5, scikit-learn>=1.2, tensorflow>=2.12, matplotlib>=3.7, protobuf>=3.20,<5
    -Run the script:
        ```bash
        python NAI_05_3.py
        ```
"""

# IMPORTS
import os
import sys
import argparse
import random
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import layers, models

# LOGGER
@dataclass
class TeeLogger:
    """
    Logger that writes messages both to standard output
    and to a log file.

    This ensures:
    - persistent experiment records
    - easy debugging and result inspection
    """
    path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")

    def print(self, *args):
        text = " ".join(str(a) for a in args)
        print(text)
        self.f.write(text + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

# UTILITY FUNCTIONS
def set_seeds(seed: int):
    """
    Set random seeds to ensure reproducibility.

    Controls randomness in:
    - Python's random module
    - NumPy
    - TensorFlow

    This is important when comparing model performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_confusion_matrix(cm, labels, path, title):
    """
    Save a confusion matrix as an image file.

    Confusion matrix structure:
    - Rows    → true labels
    - Columns → predicted labels

    Allows detailed inspection of class-wise performance,
    e.g. which clothing items are most often confused.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# FASHION-MNIST LABELS
# Class names corresponding to Fashion-MNIST numeric labels
FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# CNN MODEL DEFINITION
def build_cnn(input_shape):
    """
    Build a Convolutional Neural Network for Fashion-MNIST.

    Architecture overview:
    - Conv2D layers extract local spatial patterns (edges, textures)
    - MaxPooling reduces spatial resolution and computation
    - Dense layers perform high-level reasoning
    - Dropout reduces overfitting by random neuron deactivation
    - Softmax outputs class probabilities

    Loss:
    - sparse_categorical_crossentropy
      (labels are integers, not one-hot vectors)
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # First convolutional block
    model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D())

    # Second convolutional block
    model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling2D())

    # Fully connected part
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))

    # Output layer (10 clothing classes)
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# MAIN
def main():
    """
    Main pipeline for Task 5 – Point 3.

    Steps:
    1) Load Fashion-MNIST dataset
    2) Normalize images and reshape
    3) Build CNN model
    4) Train with early stopping
    5) Evaluate on test set
    6) Save confusion matrix and training curve
    """

    parser = argparse.ArgumentParser(
        description="NAI – Task 5 (Fashion-MNIST)"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)

    logger = TeeLogger("outputs/logs/NAI_05_3_log.txt")
    logger.print("TASK 5")
    logger.print("Dataset: Fashion-MNIST")
    logger.print("Framework: TensorFlow / Keras")
    logger.print(f"epochs={args.epochs}, batch_size={args.batch_size}, seed={args.seed}")
    logger.print("")

    # Load Fashion-MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Normalize pixel values to [0, 1]
    # Add channel dimension: (28, 28) -> (28, 28, 1)
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    input_shape = x_train.shape[1:]

    logger.print("Train samples:", x_train.shape[0])
    logger.print("Test samples:", x_test.shape[0])
    logger.print("Input shape:", input_shape)
    logger.print("")

    # Build and summarize CNN
    model = build_cnn(input_shape)
    logger.print(model.summary())

    # Early stopping:
    # Stops training when validation accuracy stops improving
    # Helps prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
    )

    # Train model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[early_stop],
    )

    # Evaluation on test set
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)

    logger.print(f"Test accuracy: {acc:.4f}")
    logger.print(classification_report(y_test, y_pred, target_names=FASHION_LABELS))

    # Confusion matrix
    cm_path = "outputs/plots/NAI_05_3_confusion_matrix.png"
    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(
        cm,
        FASHION_LABELS,
        cm_path,
        "Confusion Matrix – Fashion-MNIST"
    )
    logger.print("Saved confusion matrix to:", cm_path)

    # Training curve (learning curve)
    curve_path = "outputs/plots/NAI_05_3_training_curve.png"
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curve – Fashion-MNIST")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=200)
    plt.close()
    logger.print("Saved training curve to:", curve_path)

    logger.print("Log saved to: outputs/logs/NAI_05_3_log.txt")
    logger.print("END")
    logger.close()

if __name__ == "__main__":
    main()
