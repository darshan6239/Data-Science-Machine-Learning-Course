"""  Agglomerative Clustering is one of the most common hierarchical clustering technique where each data point starts in its own group (cluster) and step by step the closest clusters are joined together until we reach one big cluster. It's a bottom-up approach meaning:

Each data point starts in its own cluster.
Pairs of clusters are merged step-by-step based on a linkage criterion like shortest distance until all points are merged into a single cluster or until a desired number of clusters is formed  """

"""  Step 1: Importing the required libraries 
First we will import all the necessary libraries like numpy , pandas, matplotlib and scikit learn.  """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc


"""  Step 2: Loading and Cleaning the data 
We will now read the .csv file and clean it.

Remove the CUST_ID column since it's just an ID and not useful
Handle missing values using forward fill.  
USE DATASET -- CC-GENERAL.csv  """
data = pd.read_csv('CC_GENERAL.csv')

data.drop('CUST_ID', axis=1, inplace=True)

data.fillna(method='ffill', inplace=True)

print(data.head())


"""  Step 3: Preprocessing the data 
We prepare the data so that all features are on the same scale.

Scaling makes features comparable It is important because clustering depends on distance.
Normalization helps the clustering algorithm work better.  """
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_normalized = normalize(X_scaled)

X_normalized = pd.DataFrame(X_normalized)

"""  Step 4: Reducing the dimensionality of the Data 
We use PCA to reduce many columns features to just 2 so we can easily visualize the data.  """
pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']


"""  Step 5: Make the Dendrograms 
A dendrogram helps us decide how many clusters to choose. We will use the matplotlib to plot it.  """
plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))

"""  Step 6: Apply Agglomerative Clustering for Different Values of k
Now let’s apply clustering for different values of k (number of clusters). For each value of k we created a clustering model and plot the two PCA components colored by cluster.  """
for k in range(2, 7):  # Try values from 2 to 6
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_pca)

    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca['P1'], X_pca['P2'], c=labels, cmap='rainbow')
    plt.title(f'Agglomerative Clustering (k={k})')
    plt.xlabel('P1')
    plt.ylabel('P2')
    plt.show()

"""  Step 7: Evaluate models and Visualizing results
Silhouette score tells us how well the data has been grouped. The Higher the score the better is model.  """
k = [2, 3, 4, 5, 6]

silhouette_scores = []
silhouette_scores.append(
        silhouette_score(X_principal, ac2.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac3.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac4.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac5.fit_predict(X_principal)))
silhouette_scores.append(
        silhouette_score(X_principal, ac6.fit_predict(X_principal)))

plt.bar(k, silhouette_scores)
plt.xlabel('Number of clusters', fontsize = 20)
plt.ylabel('S(i)', fontsize = 20)
plt.show()

