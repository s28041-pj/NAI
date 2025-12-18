"""
NAI Project point 2

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
        python NAI_05_2.py
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
    Logger that writes output both to console and to a file.

    This is useful for:
    - experiment documentation
    - reproducibility
    - easy inspection of training results
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
    Set random seeds for reproducibility.

    This controls randomness in:
    - Python
    - NumPy
    - TensorFlow

    Deterministic behavior is important when comparing models.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_confusion_matrix(cm, labels, path, title):
    """
    Save a confusion matrix visualization to a PNG file.

    Confusion matrix interpretation:
    - Rows    → true classes
    - Columns → predicted classes

    This helps analyze:
    - which animals are confused with each other
    - class-wise performance of CNN models
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

# CIFAR-10 SETUP
# CIFAR-10 class mapping:
# 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
# 5 dog, 6 frog, 7 horse, 8 ship, 9 truck

ANIMAL_IDS = [2, 3, 4, 5, 6, 7]
ANIMAL_LABELS = ["bird", "cat", "deer", "dog", "frog", "horse"]

def filter_animals(x, y):
    """
    Filter CIFAR-10 dataset to keep only animal classes.

    Steps:
    1) Create a boolean mask for selected class IDs
    2) Remove non-animal samples
    3) Remap original CIFAR labels to range [0, 5]

    This reduces the problem from 10-class to 6-class classification.
    """
    y = y.reshape(-1)
    mask = np.isin(y, ANIMAL_IDS)

    x_animals = x[mask]
    y_animals = y[mask]

    # Remap class IDs: {2→0, 3→1, ..., 7→5}
    mapping = {cid: i for i, cid in enumerate(ANIMAL_IDS)}
    y_remap = np.array([mapping[int(v)] for v in y_animals])

    return x_animals, y_remap

# CNN MODEL BUILDER
def build_cnn(model_size, input_shape, num_classes):
    """
    Build a Convolutional Neural Network (CNN).

    CNN components:
    - Conv2D layers: extract spatial features using learnable filters
    - ReLU activation: introduces non-linearity
    - MaxPooling: spatial downsampling (translation invariance)
    - Dropout: regularization to reduce overfitting
    - Dense layers: high-level reasoning
    - Softmax output: class probability distribution

    Two architectures are supported:
    - SMALL: shallow network, fewer parameters
    - LARGE: deeper network, higher expressive power
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    if model_size == "small":
        #Small CNN
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())

        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dropout(0.3))

    elif model_size == "large":
        #Large CNN
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(32, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(64, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(128, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D())
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.4))

    else:
        raise ValueError("model_size must be 'small' or 'large'")

    # Output layer:
    # Softmax converts logits into class probabilities
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# TRAIN & EVALUATE
def train_and_evaluate(
    logger,
    model_size,
    x_train,
    y_train,
    x_test,
    y_test,
    epochs,
    batch_size,
    seed,
    cm_path
):
    """
    Train and evaluate a CNN model.

    Steps:
    1) Set random seed
    2) Build CNN architecture
    3) Train model with validation split
    4) Evaluate on test set
    5) Save confusion matrix
    """
    set_seeds(seed)

    model = build_cnn(
        model_size=model_size,
        input_shape=x_train.shape[1:],
        num_classes=len(ANIMAL_LABELS)
    )

    logger.print(f"\nTraining CNN ({model_size})")

    model.fit(
        x_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    acc = accuracy_score(y_test, y_pred)
    logger.print(f"Accuracy ({model_size}): {acc:.4f}")
    logger.print(classification_report(y_test, y_pred, target_names=ANIMAL_LABELS))

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(
        cm,
        ANIMAL_LABELS,
        cm_path,
        f"Confusion Matrix – CNN {model_size}"
    )

    logger.print("Saved confusion matrix:", cm_path)
    return acc

# MAIN
def main():
    """
    Main pipeline for Task 5 – Point 2.

    Workflow:
    1) Load CIFAR-10 dataset
    2) Filter animal classes
    3) Normalize pixel values
    4) Train CNN SMALL
    5) Train CNN LARGE
    6) Compare results
    """

    parser = argparse.ArgumentParser(description="NAI – Task 5 (CIFAR-10 animals)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = TeeLogger("outputs/logs/NAI_05_2_log.txt")

    logger.print("TASK 5")
    logger.print("Dataset: CIFAR-10 (animals only)")
    logger.print("Framework: TensorFlow / Keras")
    logger.print("")

    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep only animal classes
    x_train, y_train = filter_animals(x_train, y_train)
    x_test, y_test = filter_animals(x_test, y_test)

    # Normalize pixel intensities to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    logger.print("Train samples:", x_train.shape[0])
    logger.print("Test samples:", x_test.shape[0])

    #Train SMALL CNN
    acc_small = train_and_evaluate(
        logger,
        "small",
        x_train,
        y_train,
        x_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.seed,
        "outputs/plots/NAI_05_2_confusion_matrix_small.png"
    )

    #Train LARGE CNN
    acc_large = train_and_evaluate(
        logger,
        "large",
        x_train,
        y_train,
        x_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.seed,
        "outputs/plots/NAI_05_2_confusion_matrix_large.png"
    )

    #Summary
    logger.print("\nSUMMARY")
    logger.print(f"CNN SMALL accuracy: {acc_small:.4f}")
    logger.print(f"CNN LARGE accuracy: {acc_large:.4f}")
    logger.print("Log saved to: outputs/logs/NAI_05_2_log.txt")
    logger.print("END")

    logger.close()

if __name__ == "__main__":
    main()
