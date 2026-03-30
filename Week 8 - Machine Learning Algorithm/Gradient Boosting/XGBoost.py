"""  Traditional models like decision trees and random forests are easy to interpret but may lack accuracy on complex data. XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting algorithm that combines multiple weak models into a stronger, high-performance model.

  It uses decision trees as base learners, building them sequentially so each tree corrects errors from the previous one and it is known as boosting.
  It features parallel processing for faster training on large datasets and allows parameter customization to optimize performance for specific problems.  """

#    Implementation
"""  Here we implement XGBoost using Python and the Scikit-learn compatible API to train, predict and evaluate a classification model.

Step 1: Import Required Libraries
Import required libraries like:

  Pandas and NumPy for data manipulation
  Matplotlib and Seaborn for visualization
  XGBoost with Scikit-learn utilities are used to build and evaluate the classification model  """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

%matplotlib inline
sns.set_style("whitegrid")

#    Step 2: Load and View the Dataset
#    Here, we load the dataset using Pandas and display the first 5 rows to understand its structure, features and sample values.
df = pd.read_csv("/content/Wholesale customers data.csv")

df.head()

#    Step 3: Explore Statistical Summary of the Data
#    In this step, we use describe() to view key statistics of the dataset which helps in understanding data distribution and spotting anomalies.
print("\nStatistical Summary")
display(df.describe())

#    Step 4: Prepare Features and Target, Split Data
#    Here, we separate the dataset into features (X) and target labels (y), convert the target into binary format and split the data into training and testing sets for model training and evaluation.
X = df.drop('Channel', axis=1)
y = df['Channel'].map({1:1, 2:0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


#    Step 5: Build and Train the XGBoost Model
#    Here we initialize the XGBoost classifier with specified hyperparameters, train it on the training data and make predictions on the test set.

  """  Defines the learning objective, tree depth, learning rate, number of trees and regularization to control overfitting.
  Fits the XGBoost model on the training data (X_train, y_train).
  Uses the trained model to predict target labels on the test set (X_test).  """
params = {
    'objective':'binary:logistic',
    'max_depth':4,
    'learning_rate':0.1,
    'n_estimators':100,
    'alpha':10
}

model = XGBClassifier(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


#    Step 6: Evaluate Model Accuracy and Performance
#    In this step, we measure how well the XGBoost model performs on the test set using accuracy and a detailed classification report.
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

#    Step 7: Plot Confusion Matrix Heatmap
#    Visualizes the model’s confusion matrix using a heatmap, helping to quickly identify correct and incorrect predictions.
plt.figure(figsize=(5,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#    Step 8: Plot Feature Importance
#    Here we visualize the importance of each feature in the XGBoost model to understand which variables contribute most to predictions.
plt.figure(figsize=(8,6))
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()


#    Step 9: Visualize XGBoost Decision Tree
#    Plots one of the trained XGBoost decision trees to help understand how the model makes predictions based on feature splits.
plt.figure(figsize=(20,10))
xgb.plot_tree(model, num_trees=0)
plt.show()

