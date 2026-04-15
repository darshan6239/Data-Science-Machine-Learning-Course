"""  OPTICS (Ordering Points To Identify the Clustering Structure) is a density-based clustering algorithm similar to DBSCAN clustering. Unlike DBSCAN which struggles with varying densities. OPTICS does not directly assign clusters but instead creates a reachability plot which visually represents clusters. The key concepts in OPTICS are:

Core Distance: The minimum distance needed for a point to be classified as a core point. If a point does not have enough nearby neighbours, its core distance is undefined.
Reachability Distance: It is a measure of how difficult it is to reach from one point to another. It is calculated as the larger core distance of the starting point and the actual point.  """



"""  Implementing OPTICS in Python
Below is the Python implementation using scikit-learn to demonstrate OPTICS on a synthetic dataset of varying densities:

OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05): Configures the OPTICS algorithm.
labels=clustering.labels_: Retrieves cluster labels.
plt.scatter(): Plots the clustering results.  """
from sklearn.cluster import OPTICS
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

clustering = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
clustering.fit(X)

labels = clustering.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', edgecolor='k')
plt.title("OPTICS Clustering on Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()


