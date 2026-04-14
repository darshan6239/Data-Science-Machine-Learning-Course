"""  Agglomerative and divisive clustering are two main types of hierarchical clustering methods. Agglomerative clustering is a bottom-up approach where each data point starts as its own cluster and similar ones are merged step by step.
Divisive clustering is top-down, starting with all data in one cluster and splitting it into smaller groups based on differences.  """

#  It can be implemented using Scikit learn and SciPy library of python. Here’s a simple implementation of agglomerative clustering using randomly generated data in Python with Scipy:

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

data = np.random.randn(50, 2)

Z = linkage(data, 'ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Agglomerative Clustering Dendrogram")
plt.show()


