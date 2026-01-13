from math import e
import numpy as np
import matplotlib.pyplot as plt



class LogisticRegression:
    def __init__(self,lr=0.01,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    
    def fit(self,x,y): 
        n_samples,n_features=x.shape
        self.weights=np.zeros(n_features)
        self.bias=0

        for _ in range(self.n_iters):
            predicts=np.dot(x,self.weights)+self.bias
            predictions=(1/(1+np.exp(-predicts)))
            dw=(1/n_samples) * np.dot(x.T,(predictions-y))
            db=(1/n_samples) * np.sum(predictions-y)
            self.weights=self.weights -self.lr * dw
            self.bias=self.bias-self.lr*db
    def predict_proba(self, x):
        linear = np.dot(x.T, self.weights) + self.bias
        probs = 1 / (1 + np.exp(-linear))
        return np.column_stack((1 - probs, probs))
    def predict(self,x):
        predicts=np.dot(x,self.weights)+self.bias
        predictions=(1/(1+np.exp(-predicts)))
        class_pred=[0 if y<=0.5 else 1 for y in predictions]
        return class_pred