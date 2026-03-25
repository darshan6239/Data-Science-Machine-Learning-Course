"""  Ensemble learning is a method where multiple models are combined instead of using just one. Even if individual models are weak, combining their results gives more accurate and reliable predictions.

Multiple Models: Uses many small models together
Better Accuracy: Combined results improve performance
Reduced Errors: Mistakes of one model are balanced by those of others
Simple Idea: Like taking advice from a group instead of one person  """


#  Implementation
#    1. Importing Libraries and Loading Data
"""  We will import scikit learn for:

BaggingClassifier: for creating an ensemble of classifiers trained on different subsets of data.
DecisionTreeClassifier: the base classifier used in the bagging ensemble.
load_iris: to load the Iris dataset for classification.
train_test_split: to split the dataset into training and testing subsets.
accuracy_score: to evaluate the model’s prediction accuracy.    """
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#  2. Loading and Splitting the Iris Dataset
"""  data = load_iris(): loads the Iris dataset, which includes features and target labels.
     X = data.data: extracts the feature matrix (input variables).
     y = data.target: extracts the target vector (class labels).
     train_test_split(...): splits the data into training (80%) and testing (20%) sets, with          random_state=42 to ensure reproducibility.  """
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  3. Creating a Base Classifier
""""  Decision tree is chosen as the base model. They are prone to overfitting when trained on small datasets making them good candidates for bagging.
      base_classifier = DecisionTreeClassifier(): initializes a Decision Tree classifier, which will serve as the base estimator in the Bagging ensemble.  """
base_classifier = DecisionTreeClassifier()

#  4. Creating and Training the Bagging Classifier
"""    A BaggingClassifier is created using the decision tree as the base classifier.
n_estimators = 10 specifies that 10 decision trees will be trained on different bootstrapped subsets of the training data.    """
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)

#  5. Making Predictions and Evaluating Accuracy
"""  The trained bagging model predicts labels for test data.
The accuracy of the predictions is calculated by comparing the predicted labels (y_pred) to the actual labels (y_test).    """

y_pred = bagging_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



"""  2. Boosting Algorithm
Boosting is an ensemble technique where multiple weak models are trained one after another, and each new model focuses on correcting the errors of the previous one to build a strong model. The process works as follows:

Initialize Weights : Start with equal weights for all training data
Train Weak Learner : Train a simple model on the dataset
Sequential Learning : Each new model learns from previous errors
Weight Adjustment : Misclassified samples get higher weights so future models focus more on them
Implementaion """
#    1. Importing Libraries and Modules
"""  AdaBoostClassifier: for building the AdaBoost ensemble model.
DecisionTreeClassifier: as the base weak learner for AdaBoost.
load_iris: to load the Iris dataset.
train_test_split f: to split the dataset into training and testing sets.
accuracy_score: to evaluate the model’s accuracy.   """
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#    2. Loading and Splitting the Dataset
"""  data = load_iris(): loads the Iris dataset, which includes features and target labels.
X = data.data: extracts the feature matrix (input variables).
y = data.target: extracts the target vector (class labels).
train_test_split(...): splits the data into training (80%) and testing (20%) sets, with random_state=42 to ensure reproducibility.  """
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#    3. Defining the Weak Learner
"""  We are creating the base classifier as a decision tree with maximum depth 1 (a decision stump). This simple tree will act as a weak learner for the AdaBoost algorithm, which iteratively improves by combining many such weak learners.  """
base_classifier = DecisionTreeClassifier(max_depth=1)

#    4. Creating and Training the AdaBoost Classifier
"""  base_classifier: The weak learner used in boosting.
     n_estimators = 50: Number of weak learners to train sequentially.
     learning_rate = 1.0: Controls the contribution of each weak learner to the final model.
     random_state = 42: Ensures reproducibility.  """
adaboost_classifier = AdaBoostClassifier(
    base_classifier, n_estimators=50, learning_rate=1.0, random_state=42
)
adaboost_classifier.fit(X_train, y_train)

#    5. Making Predictions and Calculating Accuracy
""""  We are calculating the accuracy of the model by comparing the true labels y_test with the predicted labels y_pred. The accuracy_score function returns the proportion of correctly predicted samples. Then, we print the accuracy value.  """
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




