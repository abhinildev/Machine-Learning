from cProfile import label
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

bc=datasets.load_breast_cancer()
x,y=bc.data,bc.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)


lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_test)

acc=accuracy(y_pred,y_test)
print(acc)



plt.show()