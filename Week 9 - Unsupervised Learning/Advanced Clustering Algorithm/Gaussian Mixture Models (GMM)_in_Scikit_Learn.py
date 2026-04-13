"""  Gaussian Mixture Model (GMM) is a flexible clustering technique that models data as a mixture of multiple Gaussian distributions. Unlike k-means which assumes spherical clusters GMM allows clusters to take various shapes making it more effective for complex datasets.  """

"""  Implementation of GMM Covariances
To work with GMM covariances in scikit-Learn, we will use the built-in wine dataset.

Step 1: Importing Required Libraries
Before using Gaussian Mixture Models (GMM), we need to import the necessary libraries.

Scikit-Learn: This is the main library that provides the GaussianMixture class for GMM.
NumPy: Used for handling and manipulating numerical data efficiently.  """
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture

"""  Step 2: Data Preparation
We will prepare the data by selecting the first two features of the Wine dataset to simplify visualization.

datasets.load_wine(): Loads the Wine dataset.
X = wine.data[:, :2]: Selects only the first two features for simplicity.  """
wine = datasets.load_wine()
X = wine.data[:, :2]

"""  Step 3: Initializing Gaussian Mixture Model
We will initialize GMM with different covariance types to see how it affects clustering.

n_components: Number of Gaussian distributions (clusters).
covariance_type: Shape of covariance matrix, can be full, tied, diag or spherical.  """

n_components = 2  # Number of clusters
covariance_types = ['full', 'tied', 'diag', 'spherical']

"""  Step 4: Fitting the GMM Model
We will fit the GMM to the data for each covariance type to learn cluster distributions.

GaussianMixture.fit(): Fits the GMM to the data.
Dictionary comprehension: To store models for each covariance type.  """
gmm_models = {cov_type: GaussianMixture(n_components=n_components, covariance_type=cov_type)
              for cov_type in covariance_types}

for cov_type, gmm_model in gmm_models.items():
    gmm_model.fit(X)

"""  Step 5: Accessing Covariances
We will access the covariance matrices of each component to understand cluster shapes. Shape depends on covariance type.

covariances_: Attribute of fitted GMM, returns covariance matrices of components.    """
covariances = {cov_type: gmm_model.covariances_
               for cov_type, gmm_model in gmm_models.items()}

"""  Step 6: Using GMM Model for Clustering or Predictions
We will predict cluster assignments for each point using the trained GMM models. We will store predictions for all covariance types in a dictionary

GaussianMixture.predict(): Predicts cluster labels for the data.  """
predictions = {cov_type: gmm_model.predict(X)
               for cov_type, gmm_model in gmm_models.items()}

"""  Step 7: Visualization of Clusters
We will visualize the clusters for each covariance type to compare how GMM splits the data.

plt.scatter(): Plots each point colored by predicted cluster.
plt.colorbar(): Adds a color scale for clusters.
plt.subplot(): Creates multiple plots for comparison.  """
plt.figure(figsize=(12, 8))

for i, (cov_type, gmm_model) in enumerate(gmm_models.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(X[:, 0], X[:, 1], c=predictions[cov_type], cmap='viridis', edgecolors='k', s=40)
    plt.title(f'GMM Clustering with {cov_type} Covariance')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    
    print(f'Covariance Matrix ({cov_type} - Component):\n{covariances[cov_type][0]}')

plt.tight_layout()
plt.show()




