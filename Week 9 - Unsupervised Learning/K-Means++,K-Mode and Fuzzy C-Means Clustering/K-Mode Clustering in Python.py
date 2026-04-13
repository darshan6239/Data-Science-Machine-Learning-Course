"""  K-mode clustering is an unsupervised machine-learning used to group categorical data into k clusters (groups). The K-Modes clustering partitions the data into K mutually exclusive clusters. Unlike K-Means which uses distances between numbers K-Modes uses the number of mismatches between categorical values to decide how similar two data points are. For example:

Data point 1: ["red", "small", "round"]
Data point 2: ["blue", "small", "square"]   """


"""  Implementation of the k-mode clustering algorithm
K-Modes is a way to group categorical data into clusters. Here's how you can do it step-by-step in Python using just NumPy and Pandas.

Step 1: Prepare Your Data
Start by defining your dataset. Each row is a data point and each column contains categorical values like letters or labels.  """
import numpy as np
import pandas as pd

data = np.array([
    ['A', 'B', 'C'],
    ['B', 'C', 'A'],
    ['C', 'A', 'B'],
    ['A', 'C', 'B'],
    ['A', 'A', 'B']
])

"""  Step 2: Set Number of Clusters
Decide how many groups you want to divide your data into.  """
k = 2

"""  Step 3: Pick Starting Points (Modes)
Randomly choose k rows from the data to be the starting cluster centers.  """
np.random.seed(0)
modes = data[np.random.choice(data.shape[0], k, replace=False)]

"""  Step 4: Assign Data to Clusters
For each data point, count how many features are different from each mode. Assign the point to the most similar cluster.  """

clusters = np.zeros(data.shape[0], dtype=int)

for _ in range(10):  
    for i, point in enumerate(data):
        distances = [np.sum(point != mode) for mode in modes]
        clusters[i] = np.argmin(distances)

"""  Step 5: Update Cluster Modes
After assigning all points update each cluster’s mode to the most common values in that cluster.
"""
for j in range(k):
        if np.any(clusters == j):
            modes[j] = pd.DataFrame(data[clusters == j]).mode().iloc[0].values

"""  Step 6: View Final Results
Print out which cluster each data point belongs to and what the final cluster centers (modes) are.  """
print("Cluster assignments:", clusters)
print("Cluster modes:", modes)


"""  Optimal number of clusters in the K-Mode algorithm
Elbow method is used to find the optimal number of clusters  """
import pandas as pd
import numpy as np
# !pip install kmodes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
%matplotlib inline

cost = []
K = range(1,5)
for k in list(K):
    kmode = KModes(n_clusters=k, init = "random", n_init = 5, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)
    
plt.plot(K, cost, 'x-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Curve')
plt.show()


"""  As we can see from the graph there is an elbow-like shape at 2.0 and 3.0 Now it we can consider either 2.0 or 3.0 cluster. Let's consider Number of cluster =2.0  """
kmode = KModes(n_clusters=2, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(data)
clusters





  







"""  Step 3: Pick Starting Points (Modes)
Randomly choose k rows from the data to be the starting cluster centers.  """
np.random.seed(0)
modes = data[np.random.choice(data.shape[0], k, replace=False)]

