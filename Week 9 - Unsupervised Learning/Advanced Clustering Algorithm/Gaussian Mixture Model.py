"""  A Gaussian Mixture Model (GMM) is a probabilistic model that assumes data points are generated from a mixture of several Gaussian (normal) distributions with unknown parameters. Unlike hard clustering methods such as K-Means which assign each point to a single cluster based on the closest centroid, GMM performs soft clustering by assigning each point a probability of belonging to multiple clusters.   """


"""  Implementing Gaussian Mixture Model (GMM)
Import required libraries. make_blobs creates a simple synthetic dataset for demo.  """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs


"""  Step 1: Generate synthetic data
creates 500 points in 2D grouped around 3 centers. cluster_std controls how tight or spread each cluster is. y is the true label (only for reference).  """
X, y = make_blobs(
    n_samples=500,
    centers=3,
    random_state=42,
    cluster_std=[1.0, 1.5, 0.8]   # spread for each cluster
)

"""  Step 2: Fit the Gaussian Mixture Model
fit(X) runs the EM algorithm to learn means, covariances and mixing weights.
labels gives the cluster index for each point (the component with highest posterior probability).  """
gmm = GaussianMixture(
    n_components=3,        # number of Gaussian components
    covariance_type='full',
    random_state=42
)

gmm.fit(X)               
labels = gmm.predict(X)

"""  Step 3: Plot clusters and component centers
Points colored by assigned cluster and red X marks showing the learned Gaussian centers.  """
plt.figure(figsize=(8, 6))

# scatter points colored by hard labels
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')

# plot Gaussian centers
plt.scatter(
    gmm.means_[:, 0],
    gmm.means_[:, 1],
    s=300,
    c='red',
    marker='X',
    label='Centers'
)

plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.legend()
plt.show()




  
