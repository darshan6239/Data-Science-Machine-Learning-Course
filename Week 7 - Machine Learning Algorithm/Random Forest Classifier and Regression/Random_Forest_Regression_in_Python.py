"""  Random Forest is an ensemble learning method that combines multiple decision trees to produce more accurate and stable predictions. It can be used for both classification and regression tasks, where regression predictions are obtained by averaging the outputs of several trees.

      Multiple decision trees: Builds many trees and combines their predictions.
      Ensemble approach: Reduces errors compared to a single decision tree.
      Regression prediction: Produces continuous values by averaging predictions from all trees.  """

#    Implementation
"We will be implementing random forest regression on salaries data."

#    1. Importing Libraries
"""    Here we are importing numpy, pandas, matplotlib and scikit learn.

RandomForestRegressor: This is the regression model that is based upon the Random Forest model.
LabelEncoder: This class is used to encode categorical data into numerical values.
train_test_split: This function is used to split a dataset into training and testing sets.
mean_squared_error, r2_score: Used to evaluate regression performance.  """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

#    2. Importing Dataset
"    Now let's load the dataset in the panda's data frame. For better data handling and leveraging the handy functions to perform complex tasks in one go.  """
df= pd.read_csv('/content/Position_Salaries.csv')
print(df)

df.info()

#    3. Data Preparation
"""  Here the code will extracts two subsets of data from the Dataset and stores them in separate variables.

    Extracting Features: It extracts the features from the DataFrame and stores them in a variable named X.
    Extracting Target Variable: It extracts the target variable from the DataFrame and stores it in a variable named y.  """
X = df.iloc[:,1:2].values
y = df.iloc[:,2].values

#    4. Encoding categorical columns
"""  If the dataset contains object type columns, they are converted into numeric form using Label Encoding so that the machine learning model can process them.  """
label_encoder = LabelEncoder()

for col in df.select_dtypes(include=['object']).columns:
    df[col] = label_encoder.fit_transform(df[col])

#    5. Splitting Dataset
"    The dataset is divided into training and testing sets so that the model is trained on one portion and evaluated on unseen data. This prevents overly optimistic performance results.  "
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#    6. Random Forest Regressor Model
"""   The model is trained using the training dataset.

      n_estimators=100: Number of decision trees in the forest.
      random_state=42: Ensures reproducible results.
      oob_score=True: Uses out-of-bag samples to estimate model performance.  """
regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    oob_score=True
)

regressor.fit(X_train, y_train)

#    6. Making predictions and Evaluating
"""  The code evaluates the trained Random Forest Regression model:

oob_score_: Retrive out-of-bag (OOB) score which estimates the model's generalization performance.
Predictions are made on the test set.
Evaluates the model's performance using the Mean Squared Error (MSE) and R-squared (R2) metrics.  """
print("Out-of-Bag Score:", regressor.oob_score_)

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

#    7. Visualizing
"""  Now let's visualize the results obtained by using the RandomForest Regression model on our salaries dataset.

Creates a grid of prediction points covering the range of the feature values.
Plots the real data points as blue scatter points.
Plots the predicted values for the prediction grid as a green line.
Adds labels and a title to the plot for better understanding.  """
import numpy as np
import matplotlib.pyplot as plt

X_grid = np.arange(min(X), max(X), 0.01).reshape(-1,1)

plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X_grid, regressor.predict(X_grid), color='green', label="Random Forest Prediction")

plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()


#    8. Visualizing a Single Decision Tree
"    The code visualizes one of the decision trees from the trained Random Forest model. Plots the selected decision tree, displaying the decision making process of a single tree within the ensemble.  "
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

tree_to_plot = regressor.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree from Random Forest")
plt.show()










