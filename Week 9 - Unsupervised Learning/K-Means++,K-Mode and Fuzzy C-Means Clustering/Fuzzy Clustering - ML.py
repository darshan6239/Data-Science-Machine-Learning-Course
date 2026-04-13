"""  Fuzzy clustering allows each data point to belong to multiple clusters with different membership values. Instead of assigning a point to just one group, it captures how strongly a point relates to each cluster.

Uses membership scores between 0 and 1
Handles overlapping or unclear cluster boundaries
More flexible than hard clustering methods
Useful when data points don’t fit neatly into a single group
"""


"""  Implementation of Fuzzy Clustering
The fuzzy scikit learn library has a pre-defined function for fuzzy c-means which can be used in Python. For using fuzzy c-means we need to install the skfuzzy library.  """
pip install scikit-fuzzy

"""  Step 1: Importing Libraries
We will use numpy for numerical operations, skfuzzy for the Fuzzy C-Means clustering algorithm and matplotlib for plotting the results.  """
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

"""  Step 2: Generating Sample Data
We will creates 100 two-dimensional points clustered using Gaussian noise.

Set random seed (np.random.seed(0)): Ensures results are reproducible every time you run the code.
Define center = 0.5 and spread = 0.1: The cluster points will be centered around 0.5 with some variation.
Generate data (np.random.randn(2, 100)): Creates 100 random points in 2D space using Gaussian (normal) distribution.
Clip values (np.clip(data, 0, 1)): Ensures all points lie within the [0,1] range (keeps data bounded).  """
np.random.seed(0)
center = 0.5
spread = 0.1

data = center + spread * np.random.randn(2, 100)

data = np.clip(data, 0, 1)

"""  Step 3: Setting Fuzzy C-Means Parameters
Parameters control clustering behavior: number of clusters, fuzziness degree, stop tolerance and max iterations for convergence.

n_clusters = 3: We want to divide data into 3 clusters.
m = 1.7: The fuzziness parameter; higher values make cluster memberships softer (points can belong to multiple clusters).
error = 1e-5: The stopping tolerance; algorithm stops if changes are smaller than this threshold.
maxiter = 2000: The maximum number of iterations allowed to reach convergence.  """
n_clusters = 3
m = 1.7
error = 1e-5
maxiter = 2000

"""  Step 4: Performing Fuzzy C-Means Clustering and Assign Each Point to a Hard Cluster
Converts fuzzy memberships to hard cluster assignments by taking the cluster with highest membership for each point.

cntr: Final cluster centers
u: Membership matrix indicating degree of belonging for each point to each cluster
fpc: Fuzzy partition coefficient (quality metric)
This runs the clustering algorithm on the data.  """
cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None
)

hard_clusters = np.argmax(u, axis=0)

"""  Step 5: Printing Cluster Centers and Membership Matrix
Outputs coordinates of cluster centers and the membership values for the first 5 data points to provide insight into clustering results.  """
print("Cluster Centers:\n", cntr)
print("\nFuzzy Membership Matrix (first 5 data points):")
print(u[:, :5])

"""  Step 6: Visualizing Fuzzy Memberships and Hard Clusters
Plots membership levels as soft-colored points and overlays crisp cluster assignments with distinct markers to visualize both fuzzy and hard clustering. Cluster centers are highlighted with red X marks.  """
fig, ax = plt.subplots(figsize=(8, 6))

for i in range(n_clusters):
    ax.scatter(data[0], data[1], c=u[i], cmap='coolwarm',
               alpha=0.5, label=f'Fuzzy Cluster {i+1}')

markers = ['o', 's', '^']
colors = ['blue', 'green', 'orange']
for i in range(n_clusters):
    cluster_points = data[:, hard_clusters == i]
    ax.scatter(cluster_points[0], cluster_points[1], c=colors[i],
               marker=markers[i], edgecolor='k', s=80, label=f'Hard Cluster {i+1}')

ax.scatter(cntr[:, 0], cntr[:, 1], c='red',
           marker='X', s=200, label='Cluster Centers')

ax.set_title('Fuzzy C-Means')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(loc='upper left')
plt.grid(True)
plt.show()




