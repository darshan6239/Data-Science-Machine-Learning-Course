""" Installing and Using Scikit-learn
Before we start building models we need to install Scikit-learn. It requires Python 3.8 or newer and depends on two important libraries: NumPy and SciPy. Make sure these are installed first.

To install Scikit-learn run the following command: """
pip install -U scikit-learn

#    Step 1: Loading a Dataset
""" A dataset consists of:

Features (X): Input variables that describe the data
Target (y): The value we want to predict
Scikit-learn provides built-in datasets like Iris, Digits and Boston Housing. Using the Iris dataset:

load_iris() loads the data
X stores feature data
y stores target labels
feature_names and target_names give descriptive names
We can inspect the first few rows to understand the structure. For custom datasets, Pandas is commonly used to load external files such as CSVs. """
from sklearn.datasets import load_iris 
iris = load_iris() 

X = iris.data 
y = iris.target 
  
feature_names = iris.feature_names 
target_names = iris.target_names 
  
print("Feature names:", feature_names) 
print("Target names:", target_names) 

print("\nType of X is:", type(X)) 
print("\nFirst 5 rows of X:\n", X[:5])

#    Step 2: Splitting the Dataset
""" To evaluate a model fairly, we split data into:

Training set: Used to train the model
Testing set: Used to evaluate how well the model generalizes
Using train_test_split, we split the Iris dataset so that 60% is for training and 40% for testing (test_size=0.4). random_state=1 ensures reproducibility.

After splitting, we get:
X_train, y_train -> Training data
X_test, y_test -> Testing data
Checking the shapes ensures the data is split correctly."""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

""" Now lets check the Shapes of the Splitted Data to ensures that both sets have correct proportions of data avoiding any potential errors in model evaluation or training. """
print("X_train Shape:",  X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)

#    Step 3: Handling Categorical Data
""" Machine learning algorithms work with numerical inputs, so categorical (text) data must be converted into numbers. If not encoded properly, models can misinterpret categories. Scikit-learn provides multiple encoding methods:

1. Label Encoding: It converts each category into a unique integer. For example in a column with categories like 'cat', 'dog' and 'bird', it would convert them to 0, 1 and 2 respectively. This method works well when the categories have a meaningful order such as “Low”, “Medium” and “High”.

LabelEncoder(): It is initialized to create an encoder object that will convert categorical values into numerical labels.
fit_transform(): This method first fits the encoder to the categorical data and then transforms the categories into corresponding numeric labels. """
from sklearn.preprocessing import LabelEncoder
categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird']
encoder = LabelEncoder()
encoded_feature = encoder.fit_transform(categorical_feature)
print("Encoded feature:", encoded_feature)

""" 2. One-Hot Encoding: One-Hot Encoding creates separate binary columns for each category. This is useful when categories do not have any natural ordering. Example: cat, dog, bird -> 3 new columns (cat/dog/bird) with 1s and 0s.

Input must be reshaped into a 2D array
OneHotEncoder(sparse_output=False) generates binary columns"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np
categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird']
categorical_feature = np.array(categorical_feature).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded_feature = encoder.fit_transform(categorical_feature)
print("OneHotEncoded feature:\n", encoded_feature)

#    Step 4: Training the Model
""" Now that our data is ready, it’s time to train a machine learning model. Scikit-learn has many algorithms with a consistent interface for training, prediction and evaluation. Here we’ll use Logistic Regression as an example.

log_reg = LogisticRegression(max_iter=200): Creating a logistic regression classifier object.
log_reg.fit(X_train, y_train): Using this the logistic regression model adjusts the model’s parameters to best fit the data. """
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)







