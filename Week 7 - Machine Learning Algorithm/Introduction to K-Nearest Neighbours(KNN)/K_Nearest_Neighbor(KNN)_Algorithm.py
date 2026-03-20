""" K‑Nearest Neighbor (KNN) is a simple and widely used machine learning technique for classification and regression tasks. It works by identifying the K closest data points to a given input and making predictions based on the majority class or average value of those neighbors.

K-Nearest Neighbors is also called as a lazy learner algorithm because it does not learn from the training set immediately instead it stores the entire dataset and performs computations only at the time of classification.   

What is 'K' in K Nearest Neighbour?
In the k-Nearest Neighbours algorithm k is just a number that tells the algorithm how many nearby points or neighbors to look at when it makes a decision.

Example: Imagine you're deciding which fruit it is based on its shape and size. You compare it to fruits you already know.

    If k = 3, the algorithm looks at the 3 closest fruits to the new one.
    If 2 of those 3 fruits are apples and 1 is a banana, the algorithm says the new fruit is an apple because most of its neighbors are apples."""

#    Implementing KNN from Scratch in Python
#    1. Importing Libraries
#    Counter is used to count the occurrences of elements in a list or iterable. In KNN after finding the k nearest neighbor labels Counter helps count how many times each label appears.
import numpy as np
from collections import Counter

#    2. Defining the Euclidean Distance Function
#    euclidean_distance is to calculate euclidean distance between points.
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

#    3. KNN Prediction Function
    """     1) distances.append saves how far each training point is from the test point, along                 with its label.
            2) distances.sort is used to sorts the list so the nearest points come first.
            3) k_nearest_labels picks the labels of the k closest points.
            4) Uses Counter to find which label appears most among those k labels that becomes the prediction. """"
def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]   
    
#    4. Training Data, Labels and Test Point
training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

#    5. Prediction
prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)













