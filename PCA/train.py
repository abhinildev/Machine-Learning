import matplotlib.pyplot as plt
from sklearn import datasets

from pca import PCA

data=datasets.load_iris()
x=data.data
y=data.target
 
pca =PCA(2)
pca.fit(x)
x_projected=pca.transform(x)
print(x.shape)
print(x_projected.shape)
x1=x_projected[:,0]
x2=x_projected[:,1]

plt.scatter(
    x1,x2,c=y,edgecolors="none",alpha=0.8,cmap=plt.get_cmap("viridis",3)
)
plt.title("Principal component Analysis")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()
plt.show()