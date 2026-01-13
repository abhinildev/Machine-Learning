# Linear Regression — From Scratch (NumPy)

This folder contains a **scratch implementation** of **Linear Regression** using only:

- **NumPy** (for matrix operations and calculations)
- **Matplotlib** (for plotting results)

✅ **Scikit-learn** is used only for:
- Loading datasets  
- Splitting the dataset into training and testing sets  

---

## What is Linear Regression?

Linear Regression is a **supervised learning algorithm** used to model the relationship between:

- Input features (X)
- Target output (y)

It assumes a linear relationship:

y = w1*x1 + w2*x2 + ... + wn*xn + b

Where:
- w = weights (coefficients)
- b = bias (intercept)

---

## Goal of Linear Regression

The goal is to find the best values of weights (w) and bias (b) such that the predicted output is as close as possible to the actual output.

Prediction:

y_pred = X · w + b

---

## Loss Function (Mean Squared Error)

This implementation uses **Mean Squared Error (MSE)** as the loss function:

MSE = (1/m) * sum( (y - y_pred)^2 )

Where:
- m = number of samples
- y = actual values
- y_pred = predicted values

---

## Optimization (Gradient Descent)

To minimize the loss, we use **Gradient Descent**.

Update rules:

dw = (1/m) * X^T · (y_pred - y)  
db = (1/m) * sum(y_pred - y)

Weight update:

w = w - learning_rate * dw  
b = b - learning_rate * db

---

## Files in this Folder

- `linear_regression.py` -> Linear Regression class implemented from scratch  
- `train.py` -> Example script to train and test the model  

---

## How to Run

From inside this folder:

```bash
python train.py
