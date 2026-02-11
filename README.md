# Unsupervised Machine Learning — Final Peer Assignment

## Human Activity Recognition Using Smartphones: PCA, Classification & Clustering

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange.svg)](https://scikit-learn.org/)

---

## Project Overview

This project is the final peer-graded assignment for the **IBM Machine Learning Professional Certificate — Unsupervised Machine Learning** course. It demonstrates the application of **Principal Component Analysis (PCA)** as a dimensionality reduction technique in both supervised (classification) and unsupervised (clustering) machine learning workflows.

### Key Objectives

1. **PCA as preprocessing for classification** — Compare the accuracy of Gradient Boosting and Logistic Regression models with and without PCA across varying numbers of components.
2. **K-Means clustering** — Apply unsupervised clustering and evaluate cluster quality using the Silhouette Score.
3. **Visualization** — Use PCA to project 561-dimensional data into 1D, 2D, and 3D for visual inspection of cluster structure.

---

## Dataset

**Human Activity Recognition Using Smartphones**
- **Source:** [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones) (originally from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones))
- **Samples:** 10,299 (7,352 train + 2,947 test)
- **Features:** 561 numeric features derived from smartphone sensor signals
- **Subjects:** 30 volunteers (ages 19–48)
- **Device:** Samsung Galaxy S II (waist-mounted)

### Target Classes (6 Activities)

| Label | Activity |
|-------|----------|
| 1 | WALKING |
| 2 | WALKING_UPSTAIRS |
| 3 | WALKING_DOWNSTAIRS |
| 4 | SITTING |
| 5 | STANDING |
| 6 | LAYING |

### Feature Description

Features are derived from tri-axial accelerometer and gyroscope signals in both time and frequency domains. Statistical measures include: mean, standard deviation, median absolute deviation, max, min, signal magnitude area, energy, interquartile range, entropy, autoregression coefficients, correlation, skewness, kurtosis, band energy, and angle between vectors.

---

## Project Structure

```
├── 04_Unsupervised_FinalAssignment.ipynb   # Enhanced Jupyter notebook (main deliverable)
├── 04_Unsupervised_FinalAssignment.pdf     # PDF report with results and visualizations
├── README.md                                # This file
├── requirements.txt                         # Python dependencies
└── UCI HAR Dataset/                         # Dataset (download from Kaggle — see below)
    ├── activity_labels.txt
    ├── features.txt
    ├── features_info.txt
    ├── train/
    │   ├── X_train.txt
    │   ├── y_train.txt
    │   └── subject_train.txt
    └── test/
        ├── X_test.txt
        ├── y_test.txt
        └── subject_test.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone or download** this repository.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones):
   - Click "Download" on the Kaggle page
   - Extract the zip file — it will create a `UCI HAR Dataset` folder
   - Place this folder in the same directory as the notebook

4. **Launch the notebook:**
   ```bash
   jupyter notebook 04_Unsupervised_FinalAssignment.ipynb
   ```

---

## Methodology

### 1. Data Loading & Assembly
- Load split train/test files (X, y, subject) from the Kaggle dataset structure
- Combine into a single DataFrame with 10,299 rows × 561 features
- Map numeric activity IDs to descriptive labels

### 2. Exploratory Data Analysis
- Dataset shape, types, missing values
- Feature value range verification (normalized to [-1, 1])
- Correlation analysis to identify multicollinearity

### 3. Preprocessing
- Label encoding of activity names
- Feature standardization with `StandardScaler`
- New stratified train/test split (70/30)

### 4. Classification with PCA
- **Gradient Boosting Classifier** (`max_features=4`, `n_estimators=400`, `subsample=0.5`)
- **Logistic Regression** (`C=0.1`, `solver='liblinear'`, `penalty='l2'`)
- PCA sweep: n_components ∈ {10, 20, 50, 100, 150, 200, 300, 400}
- Accuracy comparison with and without PCA

### 5. K-Means Clustering
- K-Means with `n_clusters=6`, `init="k-means++"`, `n_init=12`
- Evaluation via Silhouette Score and inertia
- PCA-based visualization in 1D, 2D, and 3D projections

---

## Results Summary

### Classification

| Model | Accuracy (no PCA) | Best PCA Accuracy | Optimal n_components |
|---|---|---|---|
| Gradient Boosting | **98.9%** | 96.3% | ~200 |
| Logistic Regression | **98.0%** | 98.0% | 300–400 |

- PCA introduces a small accuracy trade-off due to information loss.
- Logistic Regression is more resilient to dimensionality reduction than Gradient Boosting.

### Clustering

- **Silhouette Score: ~0.11** — indicates poor cluster separation.
- PCA visualizations confirm significant overlap between activity groups, especially among static activities (Sitting, Standing, Laying).

---

## Key Takeaways

- PCA is effective at reducing dimensionality with minimal accuracy loss for classification tasks on this dataset.
- K-Means is not well-suited for this dataset due to overlapping, non-spherical cluster shapes.
- Alternative approaches like DBSCAN, Gaussian Mixture Models, t-SNE, or UMAP may yield better clustering and visualization results.
- Hyperparameter tuning via GridSearchCV is recommended for optimizing classifier performance.

---

## References

1. Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2012). *Human Activity Recognition Using Smartphones.* UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K
2. Kaggle Dataset: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

---

## License

This project is part of the IBM Machine Learning Professional Certificate coursework and is intended for educational purposes.
