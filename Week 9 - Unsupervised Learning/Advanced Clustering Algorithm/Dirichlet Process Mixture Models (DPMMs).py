"""  Dirichlet Process Mixture Models (DPMMs) is a flexible clustering method that can automatically decide the number of clusters based on the data. Unlike traditional methods like K-means which require you to specify the number of clusters.

It offers a probabilistic and nonparametric approach to clustering which allows the model to figure out number of groups on its own based complexity of the data.  """


"""  Implementing Dirichlet Process Mixture Models using Sklearn
Now let us implement DPMM process in scikit learn and we'll use the Mall Customers Segmentation Data. Let's understand this step-by-step:

Step 1: Import Libraries and Load Dataset
In this step we will import all the necessary libraries. This dataset contains customer information, including age, income and spending score.  USE DATASET - Mall_Customer.csv"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA

data = pd.read_csv('/content/Mall_Customers (1).csv')
print(data.head())


"""  Step 2: Feature Selection
In this step we select features that are likely to influence customer clusters.  """
X = data[['Age', 'Annual Income (k\$)', 'Spending Score (1-100)']].values

"""  Step 3: Dimensionality Reduction
We will use PCA algorithm to reduces the data's dimensions to 2 for easy visualization.  """
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

"""  Step 4: Fit Bayesian Gaussian Mixture Model
The model automatically determines the optimal number of clusters based on the data.  """
dpmm = BayesianGaussianMixture(
    n_components=10,          
    covariance_type='full',
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=1e-2,  
    random_state=42
)

dpmm.fit(X)
labels = dpmm.predict(X)

"""  Step 5: Visualization
Clusters are visualized with different colors making patterns easier to interpret.  """
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=plt.cm.Paired, edgecolors='k', s=100, linewidth=1.5)
plt.title('Dirichlet Process Mixture Model Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.show()






  

