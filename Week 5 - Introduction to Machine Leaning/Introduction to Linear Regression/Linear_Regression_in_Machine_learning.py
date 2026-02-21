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

#    4. Predicting Y Values
Y_pred = model.predict(X)

#    5. Visualizing the Regression Line
plt.figure(figsize=(8,6)) 
plt.scatter(X, Y, color='blue', label='Data Points') 
plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
plt.title('Linear Regression on Random Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

#    6. Slope and Intercept
print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
