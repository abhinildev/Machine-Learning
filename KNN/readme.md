# K-Nearest Neighbors (KNN) — From Scratch (NumPy)

This folder contains a **scratch implementation** of the **K-Nearest Neighbors (KNN)** algorithm using only:

- **NumPy** (for computations)
- **Matplotlib** (for visualization)

✅ **Scikit-learn** is used only for:
- Loading datasets  
- Splitting data into training and testing sets  

---

## 📌 What is KNN?

**KNN** is a simple, non-parametric, instance-based learning algorithm used for:

- **Classification** (most common)
- **Regression** (less common)

The idea is straightforward:

> To predict the label of a new point, find the **K closest points** from the training set and use their labels to decide the output.

---

## ✅ Why is it called “Lazy Learning”?

KNN does **not train** a model like Linear Regression or Logistic Regression.

- It stores the training dataset
- At prediction time, it computes distances to all points and decides the class

So:
- Training time ✅ fast
- Prediction time ❌ slow (depends on dataset size)

---

## 🧠 How KNN Classification Works

Given:
- Training data: \((X_{train}, y_{train})\)
- A query/test point: \(x\)
- A chosen value of **K**

Steps:

1. Compute distance between \(x\) and every training point  
2. Sort training points by distance  
3. Pick the top **K nearest neighbors**  
4. Take **majority vote** of their labels  
5. Assign the most frequent label as prediction

---

## 📏 Distance Metric (Euclidean Distance)

Most commonly, KNN uses **Euclidean distance**:

\[
d(x, x_i) = \sqrt{\sum_{j=1}^{n} (x_j - x_{ij})^2}
\]

Where:
- \(x\) is the test point  
- \(x_i\) is a training point  
- \(n\) is the number of features  

✅ In code we usually compute squared distance (without sqrt) since sqrt doesn’t change ordering.

---

## 🗳 Majority Voting

After finding K nearest neighbors, we pick the label that appears the most:

Example (K = 5):

Neighbors' labels:
\[
[1, 0, 1, 1, 0]
\]

Count:
- Class `1` → 3 times  
- Class `0` → 2 times  

✅ Prediction = `1`

---

## ⚙️ Time Complexity

Let:
- \(m\) = number of training samples
- \(n\) = number of features

### Prediction complexity:
- Distance computation: \(O(m \cdot n)\)
- Sorting distances: \(O(m \log m)\)

So overall:

\[
O(mn + m\log m)
\]

📌 KNN becomes slow when dataset size is large.

---

## 📂 Files in this Folder

- `knn.py` → Contains the KNN class implemented from scratch  
- `train.py` → Demonstration script to train/test KNN on a dataset  

---

## ▶️ How to Run

From inside this folder:

```bash
python train.py
