"""  Implementation of K-Means Clustering
We will be using blobs datasets and show how clusters are made using Python programming language.

Step 1: Importing the necessary libraries
We will be importing the following libraries.

Numpy: for numerical operations (e.g., distance calculation).
Matplotlib: for plotting data and results.
Scikit learn: to create a synthetic dataset using make_blobs  """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

"""  Step 2: Creating Custom Dataset
We will generate a synthetic dataset with make_blobs.

make_blobs(n_samples=500, n_features=2, centers=3): Generates 500 data points in a 2D space, grouped into 3 clusters.
plt.scatter(X[:, 0], X[:, 1]): Plots the dataset in 2D, showing all the points.
plt.show(): Displays the plot  """
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()

"""  Step 3: Initializing Random Centroids
We will randomly initialize the centroids for K-Means clustering

np.random.seed(23): Ensures reproducibility by fixing the random seed.
The for loop initializes k random centroids, with values between -2 and 2, for a 2D dataset.  """
k = 3

clusters = {}
np.random.seed(23)

for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }
    
    clusters[idx] = cluster
    
clusters

"""  Step 4: Plotting Random Initialized Center with Data Points
We will now plot the data points and the initial centroids.

plt.grid(): Plots a grid.
plt.scatter(center[0], center[1], marker='*', c='red'): Plots the cluster center as a red star (* marker).  """
plt.scatter(X[:,0],X[:,1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '*',c = 'red')
plt.show()

"""  Step 5: Defining Euclidean Distance
To assign data points to the nearest centroid, we define a distance function:

np.sqrt(): Computes the square root of a number or array element-wise.
np.sum(): Sums all elements in an array or along a specified axis  """
def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))

"""  Step 6: Creating Assign and Update Functions
Next, we define functions to assign points to the nearest centroid and update the centroids based on the average of the points assigned to each cluster.

dist.append(dis): Appends the calculated distance to the list dist.
curr_cluster = np.argmin(dist): Finds the index of the closest cluster by selecting the minimum distance.
new_center = points.mean(axis=0): Calculates the new centroid by taking the mean of the points in the cluster.  """
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []
        
        curr_x = X[idx]
        
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters

def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            clusters[i]['center'] = new_center
            
            clusters[i]['points'] = []
    return clusters

"""  Step 7: Predicting the Cluster for the Data Points
We create a function to predict the cluster for each data point based on the final centroids.

pred.append(np.argmin(dist)): Appends the index of the closest cluster (the one with the minimum distance) to pred.  """
def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

"""  Step 8: Assigning, Updating and Predicting the Cluster Centers
We assign points to clusters, update the centroids and predict the final cluster labels.

assign_clusters(X, clusters): Assigns data points to the nearest centroids.
update_clusters(X, clusters): Recalculates the centroids.
pred_cluster(X, clusters): Predicts the final clusters for all data points.  """
clusters = assign_clusters(X,clusters)
clusters = update_clusters(X,clusters)
pred = pred_cluster(X,clusters)

"""  Step 9: Plotting Data Points with Predicted Cluster Centers
Finally, we plot the data points, colored by their predicted clusters, along with the updated centroids.

center = clusters[i]['center']: Retrieves the center (centroid) of the current cluster.
plt.scatter(center[0], center[1], marker='^', c='red'): Plots the cluster center as a red triangle (^ marker).  """
plt.scatter(X[:,0],X[:,1],c = pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.show()


