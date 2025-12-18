"""
NAI Project point 4

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
        python NAI_05_4.py
        ```
"""

# IMPORTS
import os
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import layers, models

# CONFIGURATION
@dataclass
class Cfg:
    """
    Global configuration container.

    Contains:
    - training hyperparameters
    - directory paths
    - dataset download information
    """
    seed: int = 42
    test_size: float = 0.2
    epochs: int = 30
    batch_size: int = 32

    data_dir: Path = Path("data/phishing_url")
    logs_dir: Path = Path("outputs/logs")
    plots_dir: Path = Path("outputs/plots")

    url: str = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
    zip_file: str = "phiusiil+phishing+url+dataset.zip"

# UTILITY FUNCTIONS
def set_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Ensures consistent results across runs by
    controlling randomness in:
    - Python
    - NumPy
    - TensorFlow
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dirs(cfg: Cfg):
    """
    Create required directories if they do not exist.
    """
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)

def download_and_extract(cfg: Cfg) -> Path:
    """
    Download and extract the phishing URL dataset.

    Returns:
        Path to the extracted CSV file.
    """
    zip_path = cfg.data_dir / cfg.zip_file
    extracted_file = cfg.data_dir / "PhiUSIIL_Phishing_URL_Dataset.csv"

    if extracted_file.exists():
        return extracted_file

    print("Downloading phishing URL dataset...")
    urllib.request.urlretrieve(cfg.url, zip_path)

    print("Extracting ZIP archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cfg.data_dir)

    if not extracted_file.exists():
        raise RuntimeError("Extraction failed — CSV file not found")

    return extracted_file

def save_confusion_matrix(cm, labels, path, title):
    """
    Save confusion matrix as a PNG image.

    Confusion matrix allows detailed inspection
    of false positives and false negatives,
    which is especially important in phishing detection.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_training_curve(history, path, title="Training Curve"):
    """
    Save training curve showing accuracy over epochs.

    Displays:
    - training accuracy
    - validation accuracy

    Useful for diagnosing underfitting or overfitting.
    """
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

# MODEL DEFINITION
def build_mlp(input_dim: int, model_size: str):
    """
    Build a Multilayer Perceptron (MLP) classifier.

    Two variants:
    - small: shallow network
    - large: deeper network with more neurons

    Used to compare model capacity and performance.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    if model_size == "small":
        model.add(layers.Dense(32, activation="relu"))
    else:
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))

    # Binary classification output
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

# MAIN
def main():
    """
    Main execution pipeline for Task 5 – Point 4.

    Steps:
    1) Dataset download and preprocessing
    2) Feature encoding and scaling
    3) Training two MLP models
    4) Evaluation and visualization
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Cfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )

    ensure_dirs(cfg)
    set_seeds(cfg.seed)

    log_path = cfg.logs_dir / "NAI_05_4_log.txt"
    if log_path.exists():
        log_path.unlink()

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("TASK 5")
    log("Custom use case: Phishing URL classification")
    log(f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, seed={cfg.seed}")
    log("")

    # Load dataset
    csv_file = download_and_extract(cfg)
    df = pd.read_csv(csv_file)

    # Remove textual columns not suitable for MLP
    text_columns = ["FILENAME", "URL", "Domain", "Title"]
    df = df.drop(columns=text_columns, errors="ignore")

    # One-hot encode TLD categorical feature
    if "TLD" in df.columns:
        df = pd.get_dummies(df, columns=["TLD"], drop_first=True)

    if "label" not in df.columns:
        raise RuntimeError("Expected a column named 'label' in the dataset CSV")

    # Split features and labels
    X = df.drop("label", axis=1).values.astype(float)
    y = df["label"].values.astype(int)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=y
    )

    # Feature scaling (important for neural networks)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models_results = {}

    # Train two MLP variants
    for size in ["small", "large"]:
        log(f"Training MLP ({size})")

        model = build_mlp(X_train.shape[1], size)

        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=1,
        )

        # Save training curve
        curve_path = cfg.plots_dir / f"NAI_05_4_training_curve_{size}.png"
        plot_training_curve(
            history,
            curve_path,
            title=f"Training Curve ({size})"
        )
        log(f"Saved training curve to: {curve_path.as_posix()}")

        # Evaluation
        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        models_results[size] = acc

        log(f"Accuracy ({size}): {acc:.4f}")
        log(classification_report(
            y_test,
            y_pred,
            target_names=["legit", "phishing"]
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_path = cfg.plots_dir / f"NAI_05_4_confusion_matrix_{size}.png"
        save_confusion_matrix(
            cm,
            ["legit", "phishing"],
            cm_path,
            f"Confusion Matrix ({size})"
        )
        log(f"Saved confusion matrix to: {cm_path.as_posix()}")

    # Summary
    log("\nSummary")
    for model_name, acc in models_results.items():
        log(f"{model_name}: {acc:.4f}")

    log("\nDone")

if __name__ == "__main__":
    main()
