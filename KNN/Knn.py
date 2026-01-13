from typing import Counter
import numpy as np
#from LinearRegression.train import X_train

def euclidean_distance(x1,x2):
   return np.sqrt(np.sum((x1-x2)**2))
class Knn:
    def __init__(self,k):
        self.k=k

    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    
    def predict(self,X):
        predictions=[self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        #compute the distances
        distance=[euclidean_distance(x,X_train) for X_train in self.x_train]
        
        k_indices=np.argsort(distance)[:self.k]
        k_nearest_label=[self.y_train[i] for i in k_indices]

        most_common=Counter(k_nearest_label).most_common()

        return most_common[0][0]