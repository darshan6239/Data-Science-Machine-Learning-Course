""" Predicting rainfall is a vital aspect of weather forecasting, agriculture planning and water resource management. In this article we will use Linear regression algorithm that help establish relationship between two variables i.e one dependent (rainfall) and one or more independent variables (temperature, humidity). It tells us how many inches of rainfall we can expect. """ 

#  Step 1: Importing the required libraries
#  Here we will use pandas, numpy, matplotlib and scikit learn.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#  Step 2: Data Collection and Loading
#  Gather historical weather data, including rainfall, temperature, humidity and other relevant factors. Reliable data ensures better model accuracy and load it.
""" DOWNLOAD THE DATA SET NAME - Austin-2019-01-01-to-2023-07-22 """
data = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")

#  Step 3: Data Preprocessing
""" Clean and preprocess the data by handling missing values, removing outliers and scaling variables. Split the dataset into training and testing sets. Preprocessing ensures the model isn’t biased or skewed due to incomplete or inconsistent data leading to reliable predictions.
data.dropna(): This function is used to remove rows containing missing (NaN) values in the specified columns (features and target). It's important to handle missing data to avoid errors in model training. """
features = ['tempmax', 'tempmin', 'humidity', 'dew']
target = 'precip'
data = data.dropna(subset=features + [target])

#  Step 4: Feature Selection
""" Identify which weather variables i.e features are most correlated with rainfall. For example humidity might have a stronger correlation than temperature. Selecting relevant features improves model performance and reduces computational complexity by focusing on important variables. """
X = data[features]
y = data[target]

#  Step 5: Model Training
""" Use the training dataset to fit a linear regression model. Model learns the relationship between the independent variables (humidity, temperature) and rainfall.
train_test_split(): This function splits the dataset into training and testing sets.
test_size=0.2 indicates that 20% of the data will be used for testing and the remaining 80% will be used for training.
random_state=42 ensures that the split is reproducible.
model.fit(): Trains the linear regression model on the training data. The model learns the relationship between the features and the target variable. """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

#  Step 6: Model Evaluation
""" Test the model using the testing dataset and evaluate its performance using metrics like Mean Squared Error (MSE) or R-squared.

model.predict(): Uses the trained model to predict the target variable (y_pred) for the test data (X_test). The predicted values of rainfall are stored in y_pred. """
y_pred = model.predict(X_test)

#  Step 7: Prediction and Visualziing Results
""" Input new data into the trained model to predict rainfall. For instance, given a specific temperature and humidity, the model forecasts rainfall levels. Prediction is the ultimate goal, enabling actionable insights, such as preparing for heavy rainfall or managing agricultural schedules.

mean_squared_error(): Calculates the Mean Squared Error (MSE), which measures the average squared differences between actual and predicted values. A lower MSE indicates better model performance.
np.sqrt(): Computes the Root Mean Squared Error (RMSE), which is the square root of MSE. It gives an error metric in the same unit as the target variable (rainfall).
r2_score(): Calculates the R-squared value, which indicates how well the model explains the variance in the data. Value ranges from 0 to 1, with higher values indicating a better fit. """
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Rainfall')
plt.xlabel('Actual Rainfall')
plt.ylabel('Predicted Rainfall')
plt.grid()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Rainfall')
plt.ylabel('Residuals')
plt.grid()
plt.show()
