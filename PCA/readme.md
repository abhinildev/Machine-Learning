# Principal Component Analysis (PCA) — From Scratch (NumPy)

This folder contains a **scratch implementation** of **Principal Component Analysis (PCA)** using only:

- **NumPy** (for linear algebra)
- **Matplotlib** (for visualization)

✅ **Scikit-learn** is used only for:
- Loading datasets  
- Splitting datasets when needed  

---

## What is PCA?

**Principal Component Analysis (PCA)** is an **unsupervised learning** technique mainly used for:

- **Dimensionality Reduction**
- **Data Visualization (nD → 2D / 3D)**
- **Noise Reduction**
- **Feature Compression**
- Speeding up ML algorithms by reducing input dimensions

PCA finds a new set of axes (directions) called **Principal Components** such that:

- The 1st component captures the **maximum variance**
- The 2nd component captures the **next maximum variance**
- Each component is **perpendicular (orthogonal)** to the previous one

---

## Why PCA?

Real-world datasets often contain many features (dimensions), such as:

- 30 features
- 100 features
- 1000+ features

These high-dimensional datasets are hard to visualize and slow to process.

PCA helps reduce dimensions while keeping most of the useful information.

Example:
- 50D → 2D for plotting
- 100D → 10D for faster training

---

## Step-by-Step PCA Algorithm

### 1) Mean Center the Data
PCA works best when data is centered around the origin.

X_centered = X - mean(X)

---

### 2) Compute the Covariance Matrix
The covariance matrix represents how features vary with respect to each other.

cov = (1 / (m - 1)) * (X_centered.T @ X_centered)

Where:
- m = number of samples

---

### 3) Compute Eigenvalues and Eigenvectors
Solve:

cov * v = lambda * v

Where:
- v = eigenvector (direction of principal component)
- lambda = eigenvalue (amount of variance in that direction)

✅ Larger eigenvalue = more important principal component

---

### 4) Sort and Select Top K Components
Sort eigenvalues in descending order and choose the top `k` eigenvectors.

These eigenvectors form the projection matrix.

---

### 5) Project the Data to Lower Dimensions
Z = X_centered @ W

Where:
- W = matrix of top k eigenvectors
- Z = reduced-dimension dataset

---

## Explained Variance

Explained variance tells how much information is retained after reduction.

explained_variance_ratio = (sum of top k eigenvalues) / (sum of all eigenvalues)

Higher ratio = better representation in fewer dimensions.

---

## Files in this Folder

- `pca.py` -> PCA implementation from scratch  
- `train.py` -> Example script to reduce dimensions and visualize results  

---

## How to Run

From inside this folder:

```bash
python train.py
