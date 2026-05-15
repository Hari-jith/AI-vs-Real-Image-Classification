# AI vs Real Image Classification using Deep Learning

## Overview

This project focuses on binary image classification to distinguish between:

* AI-generated images
* Real-world photographs

The project was developed using TensorFlow and Keras and includes both:

1. A custom Convolutional Neural Network (CNN) built from scratch
2. A pretrained MobileNetV2 transfer learning model

The objective of the project is to analyze and compare the performance of a lightweight CNN architecture against a pretrained deep learning model for synthetic image detection on a small and moderately imbalanced dataset.

The project also explores:

* Data preprocessing
* Data augmentation
* Train/validation/test splitting
* Transfer learning
* Class imbalance handling
* Early stopping
* Performance comparison
* Confusion matrix analysis
* Classification metrics

---

# Dataset

Dataset Used:

AI vs Real Images Dataset

Dataset Link:

[https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset](https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset)

---

## Dataset Structure

```text
Ai_generated_dataset/
    animals/
    city/
    food/
    nature/
    people/

real_dataset/
    animals/
    city/
    food/
    nature/
    people/
```

---

## Dataset Details

| Property         | Value                 |
| ---------------- | --------------------- |
| Total Classes    | 2                     |
| Categories       | 5                     |
| Image Types      | RGB                   |
| Image Resolution | 512x512               |
| Task             | Binary Classification |

Classes:

* ai_generated
* real

Categories included:

* Animals
* City
* Food
* Nature
* People

---

# Technologies Used

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | Programming Language        |
| TensorFlow   | Deep Learning Framework     |
| Keras        | Model Building              |
| NumPy        | Numerical Operations        |
| Matplotlib   | Visualization               |
| Seaborn      | Heatmaps & Confusion Matrix |
| Scikit-learn | Metrics & Evaluation        |
| Google Colab | Training Environment        |

---

# Project Workflow

## 1. Dataset Preparation

* Dataset downloaded from Kaggle
* ZIP file uploaded to Google Drive
* Dataset extracted in Google Colab
* Images collected automatically from all categories

---

## 2. Dataset Splitting

The dataset was automatically divided into:

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

Final structure:

```text
final_dataset/
    train/
        ai_generated/
        real/

    valid/
        ai_generated/
        real/

    test/
        ai_generated/
        real/
```

---

## 3. Data Preprocessing

The following preprocessing techniques were applied:

* Image resizing
* Pixel normalization
* Data augmentation

Image settings:

```python
IMG_SIZE = 96
BATCH_SIZE = 8
```

These settings were chosen to reduce computational cost and make the project suitable for low-end systems and Google Colab free runtime.

---

## 4. Data Augmentation

The following augmentation techniques were applied to the training set:

* Rotation
* Zoom
* Horizontal flipping

This helps improve model generalization and reduce overfitting.

---

# Models Used

## 1. Simple CNN Model

A lightweight CNN architecture was built from scratch using:

* Convolutional layers
* MaxPooling layers
* Dense layers
* Dropout regularization

Architecture Summary:

```text
Conv2D (16 filters)
MaxPooling2D

Conv2D (32 filters)
MaxPooling2D

Conv2D (64 filters)
MaxPooling2D

Flatten
Dense (64)
Dropout
Output Layer (Sigmoid)
```

---

## 2. MobileNetV2 (Transfer Learning)

A pretrained MobileNetV2 model was used for transfer learning.

Configuration:

* ImageNet pretrained weights
* Top layers removed
* Custom dense layers added
* EarlyStopping applied

The MobileNetV2 base layers were initially frozen to reduce training time and computational cost.

---

# Training Configuration

| Parameter     | Value               |
| ------------- | ------------------- |
| Optimizer     | Adam                |
| Learning Rate | 0.00001             |
| Loss Function | Binary Crossentropy |
| Epochs        | Up to 10            |
| Callback      | EarlyStopping       |

---

# Class Imbalance Handling

The dataset was moderately imbalanced.

To reduce model bias toward the majority class:

* Class weights were computed using Scikit-learn
* Weighted training was applied during model fitting

This improved fairness between classes and helped the models focus more on detecting AI-generated images.

---

# Evaluation Metrics

The following evaluation metrics were used:

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

---

# Final Results

| Model       | Accuracy | Loss   |
| ----------- | -------- | ------ |
| Simple CNN  | 81.82%   | 0.4034 |
| MobileNetV2 | 61.54%   | 0.6743 |

---

# Confusion Matrix Results

## Simple CNN

```text
[[19, 12],
 [14, 98]]
```

Interpretation:

* 19 AI-generated images correctly classified
* 98 real images correctly classified
* Better overall balance between both classes

---

## MobileNetV2

```text
[[8, 23],
 [32, 80]]
```

Interpretation:

* MobileNetV2 struggled with minority-class detection
* Higher number of false positives and false negatives
* Transfer learning did not generalize well on the small dataset

---

# Classification Report (Simple CNN)

```text
              precision    recall  f1-score   support

ai_generated       0.83      0.63      0.72        30
real               0.91      0.96      0.94       112

accuracy                                0.89       142
macro avg          0.87      0.80      0.83       142
weighted avg       0.89      0.89      0.89       142
```

---

# Observations and Analysis

## Why the Simple CNN Performed Better

Despite MobileNetV2 being a powerful pretrained architecture, the custom CNN achieved better performance on this dataset.

Possible reasons include:

* Small dataset size
* Moderate class imbalance
* Simpler architecture generalizing better
* Transfer learning not adapting effectively to synthetic image artifacts
* Overfitting tendencies in pretrained models

The custom CNN learned task-specific features more effectively for this particular dataset.

---

# Key Learnings

This project helped in understanding:

* CNN architecture design
* Transfer learning
* Binary image classification
* Model evaluation techniques
* Effects of class imbalance
* Importance of EarlyStopping
* Overfitting and generalization
* Practical deep learning experimentation

---

# Future Improvements

Possible future improvements include:

* Fine-tuning MobileNetV2 layers
* Trying EfficientNet or ResNet
* Larger and more balanced datasets
* Grad-CAM visualization
* ROC-AUC analysis
* Real-time web application deployment
* Streamlit or Flask interface
* Ensemble learning approaches

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <repository-link>
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Run Notebook

Open the notebook in:

* Google Colab
* Jupyter Notebook

and execute all cells sequentially.

---

# Requirements

```text
tensorflow
numpy
matplotlib
pandas
seaborn
scikit-learn
```

---

# Project Highlights

* Lightweight deep learning project
* Suitable for low-end systems
* End-to-end TensorFlow pipeline
* Includes transfer learning comparison
* Includes performance analysis
* Includes confusion matrix and metrics
* Beginner-to-intermediate level computer vision project

---

# Conclusion

This project demonstrates an end-to-end deep learning workflow for AI-generated image detection using both custom CNNs and transfer learning approaches.

The experiments showed that for small and moderately imbalanced datasets, a lightweight custom CNN can outperform a pretrained transfer learning model in terms of generalization and class balance.

The project also highlights the importance of:

* Proper evaluation metrics
* Class imbalance handling
* EarlyStopping
* Comparative experimentation
* Model behavior analysis

Overall, the project serves as a practical introduction to deep learning-based image classification and transfer learning experimentation.
