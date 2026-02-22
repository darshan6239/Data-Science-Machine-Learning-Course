""" Linear regression is a statistical method used for predictive analysis. It models the relationship between a dependent variable and a single independent variable by fitting a linear equation to the data.

Multiple Linear Regression extends this concept by modelling the relationship between a dependent variable and two or more independent variables. This technique allows us to understand how multiple features collectively affect the outcomes.

Implementation of Multiple Linear Regression Model
We will use the California Housing dataset which includes features such as median income, average rooms and the target variable, house prices."""

#    Step 1: Importing Libraries
#    We will be using numpy, pandas, matplotlib and scikit learn for this.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

#    Step 2: Loading Dataset
#    Load the California Housing dataset from sklearn.datasets.
#    Dataset contains features such as median income, average rooms stored in X and the target i.e house prices is stored in y.
california_housing = fetch_california_housing()

X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
y = pd.Series(california_housing.target)

#    Step 3: Selecting Features for Visualization
#    Choose two features MedInc (median income) and AveRooms (average rooms) to simplify visualization in two dimensions.
X = X[['MedInc', 'AveRooms']]

#    Step 4: Train-Test Split
#    We will use 80% data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#    Step 5: Initializing and Training Model
#    Create a multiple linear regression model using LinearRegression from scikit-learn and train it on the training data.
model = LinearRegression()

model.fit(X_train, y_train)


#    Step 6: Making Predictions
#    Using the trained model to predict house prices on the test data.
y_pred = model.predict(X_test)

