"""
NAI Project point 1

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
        python NAI_05_1.py
        ```
"""

# IMPORTS
import os
import sys
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# LOGGER
@dataclass
class TeeLogger:
    """
    Logger that prints messages both to console and to a log file.

    This mimics the Unix `tee` command:
    - messages are printed to stdout
    - the same messages are written to a file

    This is useful for experiment reproducibility and reporting.
    """
    path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")

    def print(self, *args):
        text = " ".join(str(a) for a in args)
        print(text)
        self.file.write(text + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

# UTILITY FUNCTIONS
def set_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Ensures deterministic behavior across:
    - Python's random module
    - NumPy
    - TensorFlow

    Reproducibility is critical in ML experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_confusion_matrix(cm, labels, path, title):
    """
    Save a confusion matrix visualization as a PNG file.

    Confusion matrix C ∈ R^{2×2}:
        C[i, j] = number of samples of true class i
                  predicted as class j

    This visualization helps analyze:
    - true positives
    - false positives
    - false negatives
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    plt.xticks(range(len(labels)), labels, rotation=45)
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

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def build_mlp(input_dim, hidden_units=64, hidden_layers=2):
    """
    Build a Multilayer Perceptron (MLP) for binary classification.

    Architecture:
    - Input layer: dimension = number of preprocessed features
    - Hidden layers: fully connected (Dense) with ReLU activation
    - Output layer: single neuron with sigmoid activation

    Mathematical model:
        ŷ = σ(W_L · ReLU(... ReLU(W_1 x + b_1) ...) + b_L)

    where:
        σ(z) = 1 / (1 + e^{-z}) is the sigmoid function

    Loss function:
        Binary cross-entropy:
        L = − [y log(ŷ) + (1 − y) log(1 − ŷ)]
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# MAIN – POINT 1
def main():
    """
    Main execution pipeline for Task 5 – Point 1.

    Steps:
    1) Load and validate dataset
    2) Define binary target variable (mild vs severe)
    3) Split data into train/test sets
    4) Preprocess features (scaling + encoding)
    5) Train and evaluate:
       - Decision Tree
       - SVM (RBF kernel)
       - Neural Network (MLP)
    6) Save confusion matrix for Neural Network
    """

    DATA_PATH = "data/drugs_side_effects.csv"
    LOG_PATH = "outputs/logs/NAI_05_1_log.txt"
    CM_PATH = "outputs/plots/NAI_05_1_confusion_matrix.png"

    logger = TeeLogger(LOG_PATH)
    set_seeds(42)

    logger.print("TASK 5")
    logger.print("Dataset: Drugs and Side Effects")
    logger.print("Problem: Classification of side effects (mild vs severe)")
    logger.print("")

    # Load and validate dataset
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Patient_ID",
        "Age",
        "Gender",
        "Condition",
        "Drug_Name",
        "Dosage_mg",
        "Treatment_Duration_days",
        "Side_Effects",
        "Improvement_Score"
    ]

    if not all(col in df.columns for col in required_columns):
        logger.print("ERROR: CSV file has missing columns!")
        logger.print("Available columns:", list(df.columns))
        logger.close()
        sys.exit(1)

    # Target variable creation
    mild_effects = {
        "nausea",
        "dizziness",
        "headache",
        "fatigue",
        "dry mouth",
        "tiredness",
        "sleepiness"
    }

    df["Side_Effects"] = (
        df["Side_Effects"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    y = df["Side_Effects"].apply(
        lambda x: 0 if x in mild_effects else 1
    )

    logger.print("Class distribution:")
    logger.print(y.value_counts())
    logger.print("")

    # Feature matrix
    X = df.drop(columns=["Patient_ID", "Side_Effects"])

    numeric_features = [
        "Age",
        "Dosage_mg",
        "Treatment_Duration_days",
        "Improvement_Score"
    ]

    categorical_features = [
        "Gender",
        "Condition",
        "Drug_Name"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # Decision Tree
    logger.print("Decision Tree")

    dt = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(random_state=42))
    ])

    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    logger.print("Accuracy:", accuracy_score(y_test, dt_pred))
    logger.print(classification_report(y_test, dt_pred))
    logger.print("")

    # SVM (RBF)
    logger.print("SVM (RBF)")

    svm = Pipeline([
        ("prep", preprocessor),
        ("clf", SVC(kernel="rbf"))
    ])

    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    logger.print("Accuracy:", accuracy_score(y_test, svm_pred))
    logger.print(classification_report(y_test, svm_pred))
    logger.print("")

    # Neural Network (MLP)
    logger.print("Neural Network (MLP – TensorFlow/Keras)")

    X_train_nn = preprocessor.fit_transform(X_train).toarray()
    X_test_nn = preprocessor.transform(X_test).toarray()

    model = build_mlp(
        input_dim=X_train_nn.shape[1],
        hidden_units=64,
        hidden_layers=2
    )

    model.fit(
        X_train_nn,
        y_train,
        epochs=40,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )

    y_prob = model.predict(X_test_nn, verbose=0).ravel()
    y_pred_nn = (y_prob >= 0.5).astype(int)

    logger.print("Accuracy:", accuracy_score(y_test, y_pred_nn))
    logger.print(classification_report(y_test, y_pred_nn))
    logger.print("")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_nn)
    save_confusion_matrix(
        cm,
        labels=["Mild", "Severe"],
        path=CM_PATH,
        title="Confusion Matrix – Neural Network"
    )

    logger.print("Confusion matrix saved to:", CM_PATH)
    logger.print("")
    logger.print("END")

    logger.close()

if __name__ == "__main__":
    main()
