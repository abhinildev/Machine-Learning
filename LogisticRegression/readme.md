# Logistic Regression — From Scratch (NumPy)

This folder contains a **scratch implementation** of **Logistic Regression** using only:

- **NumPy** (for computations and linear algebra)
- **Matplotlib** (for visualization)

✅ **Scikit-learn** is used only for:
- Loading datasets  
- Splitting the dataset into training and testing sets  

---

## What is Logistic Regression?

Logistic Regression is a **supervised learning algorithm** mainly used for **binary classification**.

Instead of predicting a continuous value like Linear Regression, it predicts a **probability**:

- Output is between **0 and 1**
- Final class is decided using a **threshold** (usually 0.5)

---

## Model Equation

Logistic Regression uses a linear combination of inputs:

z = X · w + b

Then it passes `z` through the **Sigmoid function** to get probability:

sigmoid(z) = 1 / (1 + exp(-z))

So prediction probability:

y_pred = sigmoid(X · w + b)

---

## Decision Rule

After computing probabilities, we convert them to class labels:

- If y_pred >= 0.5 -> class 1  
- If y_pred < 0.5  -> class 0  

(Threshold can be changed depending on the problem)

---

## Loss Function (Binary Cross-Entropy)

This implementation uses **Binary Cross-Entropy Loss**:

loss = -(1/m) * sum( y*log(y_pred) + (1-y)*log(1 - y_pred) )

Where:
- m = number of samples
- y = actual labels (0 or 1)
- y_pred = predicted probabilities

---

## Optimization (Gradient Descent)

Gradients:

dw = (1/m) * X^T · (y_pred - y)  
db = (1/m) * sum(y_pred - y)

Update rules:

w = w - learning_rate * dw  
b = b - learning_rate * db

---

## Files in this Folder

- `logistic_regression.py` -> Logistic Regression class implemented from scratch  
- `train.py` -> Example script to train and test the model  

---

## How to Run

From inside this folder:

```bash
python train.py
