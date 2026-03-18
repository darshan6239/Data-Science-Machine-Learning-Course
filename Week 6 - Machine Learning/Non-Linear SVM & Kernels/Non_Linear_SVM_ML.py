""" Support Vector Machines (SVM) are algorithms for classification and regression tasks. However, the standard (linear) SVM can only classify data that is linearly separable, meaning a straight line can separate the classes (in 2D) or a hyperplane (in higher dimensions). Non-linear SVM extends SVM to handle complex, non-linearly separable data using kernels."""


""" Example 1: Non linear SVM in Circular Decision Boundary
Below is the Python implementation for Non linear SVM in circular decision boundary.
1. Importing Libraries
We begin by importing the necessary libraries for data generation, model training, evaluation and visualization. """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

""" 2. Creating and Splitting the Dataset
We generate a synthetic dataset of concentric circles and split it into training and testing sets. """
X, y = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

""" 3. Creating and Training the Non-Linear SVM Model
We create an SVM classifier using the RBF kernel to handle non-linear patterns and train it on the data.  """ 
svm = SVC(kernel='rbf', C=1, gamma=0.5)  # RBF kernel allows learning circular boundaries
svm.fit(X_train, y_train)

""" 4. Making Predictions and Evaluating the Model
We predict the labels for the test set and compute the accuracy of the model."""
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

""" 5. Visualizing the Decision Boundary
We define a function to visualize the decision boundary of the trained non-linear SVM on the dataset. """
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title("Non-linear SVM with RBF Kernel")
    plt.show()
# Plot the decision boundary
plot_decision_boundary(X, y, svm)
 






