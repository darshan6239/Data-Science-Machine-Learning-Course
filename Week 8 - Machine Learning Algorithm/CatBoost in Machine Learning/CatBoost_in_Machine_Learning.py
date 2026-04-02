"""   When working with machine learning we often deal with datasets that include categorical data. We use techniques like One-Hot Encoding or Label Encoding to convert these categorical features into numerical values. However One-Hot Encoding can lead to sparse matrix and cause overfitting.  """

"""  CatBoost Installation
CatBoost is an open-source library that does not comes pre-installed with Python so before using CatBoost we must install it in our local system.

For installing CatBoost in Python 
pip install catboost

For Installing CatBoost In R
install.packages("catboost")
Implementation of CatBoost
We will see its implementation in step by step process:  """

#    Step 1: Importing Libraries
#      We will import catboost and scikit learn.
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#    Step 2: Loading and splitting the dataset
#        We will load iris dataset and divide it into 80% training dataset and 20% testing.
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

#    Step 3: Initializing and Training Model
model = CatBoostClassifier(
    iterations=100,      
    learning_rate=0.1,   
    depth=6,              
    verbose=0             
)

model.fit(X_train, y_train)

#    Step 4: Making Predictions and Evaluating
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")


