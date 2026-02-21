""" Python Implementation of Linear Regression """
#    1. Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#    Generating Random Dataset
#    Fetches the California Housing dataset and separates features (X) and target (y).
np.random.seed(42)
X = np.random.rand(50, 1) * 100  
Y = 3.5 * X + np.random.randn(50, 1) * 20

#    3. Creating and Training Linear Regression Model
model = LinearRegression()
model.fit(X, Y)
