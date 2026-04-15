"""  T-distributed Stochastic Neighbor Embedding (t-SNE) is a non linear dimensionality reduction technique used for visualizing high-dimensional data in a lower-dimensional space mainly in 2D or 3D. Unlike linear methods such as Principal Component Analysis (PCA), t-SNE focus on preserving the local structure and pattern of the data.  """

"""  Implementation of t-SNE on MNIST Dataset
Now let's use the sklearn implementation of the t-SNE algorithm on the MNIST dataset which contains 10 classes that are for the 10 different digits in the mathematics.  """

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

"Now let's load the MNIST dataset into pandas dataframe."
mnist = fetch_openml('mnist_784', version=1)

d = mnist.data  
l = mnist.target  

df = pd.DataFrame(d)
df['label'] = l  

print(df.head(4))

"""  Before applying the t-SNE algorithm on the dataset we must standardize the data. As we know that the t-SNE algorithm is a complex algorithm which utilizes some complex non-linear methods.  """
from sklearn.preprocessing import StandardScaler

X = df.drop('label', axis=1)
standardized_data = StandardScaler().fit_transform(X)

print(standardized_data.shape)

"  Now let's reduce the 784 columns data to 2 dimensions so that we can create a scatter plot to visualize the same.  "
data_1000 = standardized_data[0:1000, :]
labels_1000 = l[0:1000]

model = TSNE(n_components = 2, random_state = 0)

tsne_data = model.fit_transform(data_1000)

tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data = tsne_data,
     columns =("Dim_1", "Dim_2", "label"))

sn.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2',
               hue='label', palette="bright")
plt.show()

