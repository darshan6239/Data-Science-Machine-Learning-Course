""" Implementation of Gradient Descent in Linear Regression
Let’s implement linear regression step by step. To understand how gradient descent improves the model, we will first build a simple linear regression without using gradient descent and observe its results.

Here we will be using Numpy, Pandas, Matplotlib and Sckit learn libraries for this.

X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42): Generating 100 data points with one feature and some noise for realism.
X_b = np.c_[np.ones((m, 1)), X]: Adding a column of ones to X to account for the intercept term in the model.
theta = np.array([[2.0], [3.0]]): Initializing model parameters (intercept and slope) with starting values. """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
y = y.reshape(-1, 1)
m = X.shape[0]

X_b = np.c_[np.ones((m, 1)), X]

theta = np.array([[2.0], [3.0]])

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, X_b.dot(theta), color="green", label="Initial Line (No GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression Without Gradient Descent")
plt.legend()
plt.show()''

""" Here the model’s predictions are not accurate and the line does not fit the data well. This happens because the initial parameters are not optimized which prevents the model from finding the best-fit line.

Now we will apply gradient descent to improve the model and optimize these parameters.

learning_rate = 0.1, n_iterations = 100: Set the learning rate and number of iterations for gradient descent to run respectively.
gradients = (2 / m) * X_b.T.dot(y_pred - y): Finding gradients of the cost function with respect to parameters.
theta -= learning_rate * gradients: Updating parameters by moving opposite to the gradient direction. """
learning_rate = 0.1
n_iterations = 100

for _ in range(n_iterations):

    y_pred = X_b.dot(theta)

    gradients = (2 / m) * X_b.T.dot(y_pred - y)

    theta -= learning_rate * gradients

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, X_b.dot(theta), color="red", label="Optimized Line (With GD)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.title("Linear Regression With Gradient Descent")
plt.legend()
plt.show()



