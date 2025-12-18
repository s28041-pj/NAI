
**Point 1**

Dataset:
- 1000 Drugs and Side Effects (tabular dataset)

Problem statement:
Binary classification of drug side effects:
    0 – mild side effects
    1 – severe side effects

Compared machine learning approaches:
- Decision Tree (tree-based, non-parametric model)
- Support Vector Machine (RBF kernel)
- Neural Network (MLP implemented in TensorFlow/Keras)

Outputs:
- Log file: outputs/logs/NAI_05_1_log.txt
- Confusion matrix (Neural Network): outputs/plots/NAI_05_1_confusion_matrix.png

Frameworks and libraries:
- scikit-learn (Decision Tree, SVM, preprocessing)
- TensorFlow / Keras (Neural Network)
- NumPy, pandas, matplotlib

![NAI_05_1_screen_1.png](resources/NAI_05_1_screen_1.png)
![NAI_05_1_screen_2.png](resources/NAI_05_1_screen_2.png)
![NAI_05_1_confusion_matrix.png](outputs/plots/NAI_05_1_confusion_matrix.png)

**Point 2**

Goal:
Train convolutional neural networks (CNNs) to recognize ANIMALS
using the CIFAR-10 image dataset.

Animal classes selected from CIFAR-10:
- bird
- cat
- deer
- dog
- frog
- horse

This task is a multi-class image classification problem
with 6 output classes.

Compared models:
- CNN SMALL  (shallow architecture, fewer parameters)
- CNN LARGE  (deeper architecture, higher representational capacity)

Framework:
- TensorFlow / Keras (consistent with the whole assignment)

Outputs:
- logs: outputs/logs/NAI_05_2_log.txt
- confusion matrices:
  - outputs/plots/NAI_05_2_confusion_matrix_small.png
  - outputs/plots/NAI_05_2_confusion_matrix_large.png

Only animal classes are used (vehicles are removed).
Pixel values are normalized to [0, 1].
Evaluation is based on accuracy and confusion matrices.

![NAI_05_2_screen_1.png](resources/NAI_05_2_screen_1.png)
![NAI_05_2_screen_2.png](resources/NAI_05_2_screen_2.png)
![NAI_05_2_confusion_matrix_large.png](outputs/plots/NAI_05_2_confusion_matrix_large.png)
![NAI_05_2_confusion_matrix_small.png](outputs/plots/NAI_05_2_confusion_matrix_small.png)

**Point 3**

Goal:
Train a convolutional neural network (CNN) to recognize clothing items
using the Fashion-MNIST dataset.

Dataset:
- Fashion-MNIST
- 60,000 training images
- 10,000 test images
- Grayscale images, size 28×28
- 10 classes of clothing items

Model:
- Convolutional Neural Network (CNN)
- Implemented using TensorFlow / Keras

Evaluation:
- test accuracy
- classification report
- confusion matrix (PNG)
- training curve (PNG)

Outputs:
- outputs/logs/NAI_05_3_log.txt
- outputs/plots/NAI_05_3_confusion_matrix.png
- outputs/plots/NAI_05_3_training_curve.png

Fashion-MNIST is more challenging than classic MNIST digits
CNN is well-suited for spatial image data
Early stopping is used to prevent overfitting

![NAI_05_3_screen_1.png](resources/NAI_05_3_screen_1.png)
![NAI_05_3_screen_2.png](resources/NAI_05_3_screen_2.png)
![NAI_05_3_confusion_matrix.png](outputs/plots/NAI_05_3_confusion_matrix.png)
![NAI_05_3_training_curve.png](outputs/plots/NAI_05_3_training_curve.png)

**Point 4**

Custom use case:
Phishing URL classification using tabular data.

Dataset:
- PhiUSIIL Phishing URL Dataset (UCI)
- Binary classification: legitimate vs phishing URLs

Data source (downloaded automatically):
https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip

Description:
This script:
1) Downloads and preprocesses the dataset
2) Encodes categorical features (TLD)
3) Standardizes numerical features
4) Trains two Multilayer Perceptron (MLP) models:
   - small network
   - larger network
5) Evaluates performance using:
   - accuracy
   - classification report
   - confusion matrix
   - training curve

Outputs:
- outputs/logs/NAI_05_4_log.txt
- outputs/plots/NAI_05_4_confusion_matrix_small.png
- outputs/plots/NAI_05_4_confusion_matrix_large.png
- outputs/plots/NAI_05_4_training_curve_small.png
- outputs/plots/NAI_05_4_training_curve_large.png

![NAI_05_4_screen_1.png](resources/NAI_05_4_screen_1.png)
![NAI_05_4_screen_2.png](resources/NAI_05_4_screen_2.png)
![NAI_05_4_screen_3.png](resources/NAI_05_4_screen_3.png)
**large**
![NAI_05_4_confusion_matrix_large.png](outputs/plots/NAI_05_4_confusion_matrix_large.png)
![NAI_05_4_training_curve_large.png](outputs/plots/NAI_05_4_training_curve_large.png)
**small**
![NAI_05_4_confusion_matrix_small.png](outputs/plots/NAI_05_4_confusion_matrix_small.png)
![NAI_05_4_training_curve_small.png](outputs/plots/NAI_05_4_training_curve_small.png)