# %%
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
mnist = pd.read_csv("data/MNIST_100.csv")
mnist.head()
y = mnist['label']
x = mnist.drop('label', axis=1)
x5 = mnist.iloc[500:602]
y5 = y.iloc[500:602]
x6 = mnist.iloc[600:702]
y6 = y.iloc[600:702]
xfs = pd.concat([x5,x6], axis=0)
yfs = pd.concat([y5,y6], axis=0)
xfs.shape

# %%
pca = PCA(n_components=2)
pca.fit(xfs)
pca = pca.transform(xfs)
plt.plot(pca[:,0],pca[:,1], 'wo',)
for i in range(len(yfs)):
    plt.text(pca[i:i+1,0], pca[i:i+1,1],yfs.iloc[i])
plt.show()

# %%
pca2 = PCA(n_components=2)
pca2.fit(x)
pca2 = pca2.transform(x)
plt.plot(pca2[:,0],pca2[:,1], 'wo',)
for i in range(len(y)):
    plt.text(pca2[i:i+1,0], pca2[i:i+1,1],y[i])
plt.show()

# %%



