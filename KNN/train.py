from matplotlib.colors import ListedColormap
from networkx import predecessor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from Knn import Knn
cmap=ListedColormap(['#FF0000','#00FF00','#0000FF'])
bc=datasets.load_iris()
x,y=bc.data,bc.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)

plt.figure()
plt.scatter(x[:,2],x[:,3],c=y,cmap=cmap,edgecolors='k',s=20)
plt.show()

clf=Knn(k=5)
clf.fit(x_train,y_train)
pred=clf.predict(x_test)
#predict=np.array(predict)
print(pred)

acc=np.sum(pred==y_test)/len(y_test)
print(acc)