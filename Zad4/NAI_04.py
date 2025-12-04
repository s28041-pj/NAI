#!/usr/bin/env python3
"""
ML Project: Wine + Drugs Classification

Rules:
See the README.md file in this repository.
https://github.com/s28041-pj/NAI/blob/main/Zad4/readme.md

Authors:
- Mikołaj Gurgul, Łukasz Aleksandrowicz

## Requirements
    - Python 3.13
    - pandas, numpy, matplotlib, seaborn, scikit-learn, tkinter (built-in)

## How to run the script:

    - Install Python (>=3.13)
    - Clone this repository
    -pip install requirements.txt / pandas, numpy, matplotlib, seaborn, scikit-learn, tkinter (built-in)
    -Run the script:
        ```bash
        python NAI_04.py
        ```
"""

# IMPORTS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import svm

import tkinter as tk
from tkinter import messagebox

# HELPER FUNCTIONS

def load_dataset(path, **kwargs):
    """
    Load a dataset from CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    kwargs : dict
        Additional arguments passed to pandas.read_csv().

    Returns
    -------
    DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path, **kwargs)


def plot_histograms(df, title):
    """
    Draw histogram plots for all numerical columns in the dataset.

    -----
    Useful for visualizing the distribution of features and spotting outliers.

    - A histogram is an empirical estimator for the probability density
      function of a continuous variable. Choosing bin width is a bias/variance
      tradeoff: too wide → smoothed, too narrow → noisy.
    - Histograms are useful to identify skewness, multimodality, and outliers,
      which can inform preprocessing (e.g., transformations or winsorization).
    """
    numeric = df.select_dtypes(include=[np.number])
    numeric.hist(figsize=(14, 10), bins=20)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_heatmap(df, title):
    """
    Draw a correlation heatmap for numerical features.

    --------------------
    A heatmap is a visual encoding of a pairwise statistic across features.
    The typical choice here is the Pearson correlation coefficient r_ij.

    1) Data matrix X ∈ R^{n×d} (n samples, d features).
    2) For features x_i, x_j compute Pearson correlation:

           r_ij = cov(x_i, x_j) / (σ_i σ_j)

       where cov(x_i, x_j) = E[(x_i - μ_i)(x_j - μ_j)] estimated from sample.
       In matrix notation: let S be the sample covariance matrix, and D = diag(σ_1, ..., σ_d),
       then the correlation matrix is C = D^{-1} S D^{-1}.

    3) Properties:
       - Symmetric: C = C^T
       - Diagonal entries = 1
       - Values in [-1, 1], where:
           +1 -> perfect positive linear relationship,
           -1 -> perfect negative linear relationship,
            0 -> no linear relationship.

    4) Limitations:
       - Pearson measures only linear dependency; nonlinear relationships may be hidden.
       - Sensitive to outliers, which inflate covariances.

    5) Interpretation in ML:
       - Strong absolute correlations between features indicate multicollinearity.
       - Multicollinearity can destabilize linear models; for tree-based models it's less problematic.
       - Highly correlated features may be candidates for dimensionality reduction (PCA) or feature selection.
    """
    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=False, cmap="coolwarm")
    plt.title(title)
    plt.show()


# 1) WINE QUALITY ANALYSIS

def analyze_wine():
    """
    Perform complete analysis of the Wine Quality dataset:
    - load dataset
    - visualize distributions and correlations
    - split data
    - train Decision Tree and SVM models
    - compare kernels
    - display example prediction

    -----
    SVM is wrapped in a Pipeline that applies StandardScaler,
    because SVM is highly sensitive to feature scaling.
    """

    print(" WINE QUALITY DATASET")

    # Load dataset
    wine = load_dataset("resources/winequality-red.csv", sep=";")

    # Create a binary target variable:
    # 1 = good wine (quality >= 6)
    # 0 = bad wine
    wine["quality_label"] = (wine["quality"] >= 6).astype(int)

    # Separate features and target
    X = wine.drop(columns=["quality", "quality_label"])
    y = wine["quality_label"]

    # Visualizations
    plot_histograms(wine, "Wine – Histograms")
    plot_heatmap(wine, "Wine – Heatmap")

    # Split data; stratify preserves class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Decision Tree
    # A Decision Tree builds a hierarchical structure by recursively
    # partitioning the feature space. At each node the algorithm selects
    # a feature and threshold that maximize the reduction in impurity.
    #
    # For classification, impurity measures include:
    #   - Gini impurity: G = Σ_k p_k (1 - p_k)
    #   - Entropy: H = -Σ_k p_k log p_k
    #
    # where p_k is the class probability in the node.
    #
    # The chosen split is the one that maximizes information gain:
    #   gain = impurity(parent) - [w_left * impurity(left) + w_right * impurity(right)]
    #
    # Stopping criteria: purity, minimum samples, max depth, or no further gain.
    # Decision Trees produce axis-aligned splits (hyperrectangles).
    # They are non-parametric and can model complex interactions, but are prone to overfitting.
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)  # Learn rules from data
    pred_tree = tree.predict(X_test)

    print("\nWine: Decision Tree")
    print("Accuracy:", accuracy_score(y_test, pred_tree))
    print(classification_report(y_test, pred_tree))
    print(confusion_matrix(y_test, pred_tree))

    # SVM
    # StandardScaler explanation:
    # SVM relies on dot products / distances in feature space. If different
    # features have different scales, those with larger numeric range dominate.
    # StandardScaler performs z-score normalization:
    #       x' = (x - μ) / σ
    # After scaling: mean 0, std 1 for each feature (empirical).
    # This is important for regularized margin-based methods like SVM.
    #
    # SVM geometric interpretation:
    # - Seek hyperplane w·x + b = 0 maximizing margin 2/||w||.
    # - Primal optimization (hard-margin):
    #       minimize  1/2 ||w||^2
    #       subject to y_i (w·x_i + b) >= 1  for all i
    # - Soft-margin adds slack variables ξ_i and penalty C:
    #       minimize  1/2 ||w||^2 + C Σ ξ_i
    #       subject to y_i (w·x_i + b) >= 1 - ξ_i,  ξ_i >= 0
    #
    # Kernel trick:
    # - Replace dot product x·x' with kernel K(x, x') = φ(x)·φ(x') allowing
    #   implicit mapping to high-dimensional φ-space without computing φ explicitly.
    #
    # RBF kernel is chosen here as default for its flexibility and locality.
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", svm.SVC(kernel="rbf"))
    ])
    svm_model.fit(X_train, y_train)
    pred_svm = svm_model.predict(X_test)

    print("\nWine: SVM (RBF)")
    print("Accuracy:", accuracy_score(y_test, pred_svm))
    print(classification_report(y_test, pred_svm))
    print(confusion_matrix(y_test, pred_svm))

    # Kernel comparison
    # Each kernel defines a different implicit mapping φ and therefore a
    # different hypothesis class for the SVM decision function:
    #
    # - linear:  K(x, x') = x^T x'
    #   No mapping; decision boundary linear in input space.
    #
    # - polynomial: K(x, x') = (γ x^T x' + r)^d
    #   Models interactions up to degree d. Degree and γ control capacity.
    #
    # - rbf (Gaussian): K(x, x') = exp(−γ ||x − x'||^2)
    #   Local similarity measure; corresponds to infinite-dimensional φ.
    #
    # - sigmoid: K(x, x') = tanh(γ x^T x' + r)
    #   Similar to a two-layer neural network activation; not always positive-definite.
    #
    # - RBF often performs well out-of-the-box because it can model complex
    #   local decision boundaries while remaining smooth.
    print("\nWine: Kernel Comparison")
    for k in ["linear", "poly", "rbf", "sigmoid"]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", svm.SVC(kernel=k))
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"{k}: {accuracy_score(y_test, pred)}")

    # Example prediction
    # We present a single example prediction to satisfy "example prediction" requirement.
    example = X_test.iloc[0:1]  # Pick first example directly
    example_pred = svm_model.predict(example)

    print("\nExample Wine Prediction")
    print("Input:", example.to_dict())
    print("Prediction:", "Good wine" if example_pred[0] == 1 else "Bad wine")

    messagebox.showinfo("Wine", "Wine analysis completed!")

# 2) DRUG EFFECTIVENESS CLASSIFICATION
def analyze_drugs():
    """
    Perform full ML pipeline for the Drug Effectiveness dataset.

    Includes:
    - feature engineering (side effect length, word count)
    - one-hot encoding categorical variables
    - training Decision Tree and SVM classifiers
    - kernel comparison
    - example prediction

    Notes
    -----
    The dataset is high-dimensional after encoding (many drug names),
    so SVM with RBF usually performs better than Decision Tree for complex patterns.
    """

    print("DRUG DATASET")
    print("Source: Kaggle – 1000 Drugs and Side Effects")
    print("Link: https://www.kaggle.com/datasets/palakjain9/1000-drugs-and-side-effects")

    df = load_dataset("resources/real_drug_dataset.csv")

    # Binary target: 1 = patient improved, 0 = no improvement
    df["Improved"] = (df["Improvement_Score"] >= 7).astype(int)

    # Feature engineering:
    # - side_length: number of characters in the Side_Effects text
    # - side_words: number of whitespace-separated tokens (approximate word count)
    #
    # - These are extremely simple text-derived features (bag-of-lengths).
    # - They may capture signal (longer descriptions -> more severe/multiple effects),
    #   but lose semantic information; for richer models use TF-IDF or embeddings.
    df["side_length"] = df["Side_Effects"].astype(str).apply(len)
    df["side_words"] = df["Side_Effects"].astype(str).apply(lambda x: len(x.split()))

    # One-hot encoding converts categorical variables into binary indicator columns.
    # Mathematical explanation:
    # - For a categorical variable with k categories, one-hot produces a k-dimensional
    #   basis vector e_i with a single 1 in the index corresponding to the category.
    # - This embeds nominal data into R^k enabling linear algebraic operations.
    # - drop_first=True removes one column to avoid perfect multicollinearity
    #   (dummy variable trap) which can be problematic for certain algorithms.
    df = pd.get_dummies(
        df,
        columns=["Gender", "Condition", "Drug_Name", "Side_Effects"],
        drop_first=True
    )

    X = df.drop(columns=["Patient_ID", "Improvement_Score", "Improved"])
    y = df["Improved"]

    # Visualizations
    plot_histograms(df, "Drugs – Histograms")
    plot_heatmap(df, "Drugs – Heatmap")

    # Train/test split with stratification for class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Decision Tree
    # Decision trees are robust to feature scaling and to mixed data types,
    # but they can overfit and are sensitive to small changes in data.
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train, y_train)
    pred_tree = tree.predict(X_test)

    print("\nDrugs: Decision Tree")
    print("Accuracy:", accuracy_score(y_test, pred_tree))
    print(classification_report(y_test, pred_tree))
    print(confusion_matrix(y_test, pred_tree))

    # SVM
    # For high-dimensional one-hot encoded data, kernel choice matters; RBF
    # considers locality via pairwise distances and can handle nonlinear patterns.
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", svm.SVC(kernel="rbf"))
    ])
    svm_model.fit(X_train, y_train)
    pred_svm = svm_model.predict(X_test)

    print("\nDrugs: SVM (RBF)")
    print("Accuracy:", accuracy_score(y_test, pred_svm))
    print(classification_report(y_test, pred_svm))
    print(confusion_matrix(y_test, pred_svm))

    # Kernel comparison
    print("\nDrugs: Kernel Comparison")
    for k in ["linear", "poly", "rbf", "sigmoid"]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", svm.SVC(kernel=k))
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"{k}: {accuracy_score(y_test, pred)}")

    # Example prediction
    example = X_test.iloc[0:1]
    example_pred = svm_model.predict(example)

    print("\nExample Drug Prediction")
    print("Input:", example.to_dict())
    print("Prediction:", "Improved (1)" if example_pred[0] == 1 else "Not Improved (0)")

    messagebox.showinfo("Drugs", "Drug analysis completed!")

def kernel_summary():
    """
    Display a short explanation of kernel functions used in SVM.

    Notes
    -----
    Kernel functions transform input data into a (possibly much higher)
    dimensional feature space where a linear separator may exist. The trick
    is that the mapping φ is usually implicit: SVM uses K(x, x') = φ(x)·φ(x')
    so we never compute φ explicitly.
    """
    text = (
        "Kernel Function Summary:\n\n"
        "- LINEAR:\n"
        "  Works well when data is linearly separable. Fast and simple.\n\n"
        "- POLY:\n"
        "  Polynomial mapping captures nonlinear interactions, but may overfit.\n\n"
        "- RBF (Gaussian):\n"
        "  Most robust and widely used kernel. Handles complex nonlinear patterns.\n\n"
        "- SIGMOID:\n"
        "  Behaves similarly to neural networks but often unstable.\n\n"
        "In both datasets (Wine + Drugs), RBF kernel achieved the best accuracy."
    )
    messagebox.showinfo("Kernel Summary", text)

def run_gui():
    """
    Create and launch the GUI for running analyses.

    Notes
    -----
    Tkinter is used for simplicity; buttons map directly to analysis functions.
    The GUI is intentionally minimal: it triggers heavy computations in the
    main thread and blocks the window while plotting; for production use a
    worker thread and progress indicator are recommended.
    """
    window = tk.Tk()
    window.title("ML Project: Wine + Drugs")
    window.geometry("420x390")

    tk.Label(window, text="Choose dataset:", font=("Arial", 16)).pack(pady=20)

    tk.Button(window, text="Analyze Wine Dataset", font=("Arial", 12),
              command=analyze_wine).pack(pady=10)

    tk.Button(window, text="Analyze Drug Dataset", font=("Arial", 12),
              command=analyze_drugs).pack(pady=10)

    tk.Button(window, text="Kernel Summary", font=("Arial", 12),
              command=kernel_summary).pack(pady=10)

    tk.Button(window, text="EXIT", font=("Arial", 12),
              command=window.destroy).pack(pady=20)

    window.mainloop()

if __name__ == "__main__":
    run_gui()
