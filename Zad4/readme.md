This script performs classification tasks on two datasets:

1. **Wine Quality Prediction**
   - Predicts whether wine is of "good" or "bad" quality.
   - Uses Decision Tree and SVM classifiers.
   - Performs kernel comparison: linear, polynomial, rbf, sigmoid.
   - Generates example predictions (assignment requirement).
![wine_decision_tree.png](wine/wine_decision_tree.png)
![wine_svm.png](wine/wine_svm.png)
![wine_kernel.png](wine/wine_kernel.png)
![wine_example_prediction.png](wine/wine_example_prediction.png)
![wine_analysis_complete.png](wine/wine_analysis_complete.png)

2. **Drug Effectiveness Prediction**
   - Predicts whether a patient improved after taking a medication.
   - Includes feature engineering and one-hot encoding.
   - Supports the same ML models as the wine dataset.
   - Also includes example prediction.
![drug_decision_tree.png](drug/drug_decision_tree.png)
![drug_svm.png](drug/drug_svm.png)
![drug_kernel.png](drug/drug_kernel.png)
![drug_example_prediction.png](drug/drug_example_prediction.png)
![drug_analysis_complete.png](drug/drug_analysis_complete.png)

3. **Visualization**
   - Histograms for numeric features.
   - Heatmap for feature correlations.

**Wine**

![wine_histogram.png](wine/wine_histogram.png)
![wine_heatmap.png](wine/wine_heatmap.png)

**Drug**

![drug_histogram.png](drug/drug_histogram.png)
![drug_heatmap.png](drug/drug_heatmap.png)

4. **GUI (Tkinter)**
   - User-friendly interface to run analysis.
   - Displays kernel function summary.

![gui.png](gui.png)

This project demonstrates SVM behaviour under different kernel functions:

- Linear kernel — linearly separable decision boundaries.
- Polynomial — nonlinear mapping with controllable degree.
- RBF — Gaussian kernel; flexible, smooth nonlinear boundaries.
- Sigmoid — behaves similarly to shallow neural networks.

![kernel_summary.png](kernel_summary.png)