"""  Random Forest is an ensemble machine learning algorithm that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. In Scikit‑learn, the Random Forest Classifier is widely used for classification tasks because it handles large datasets and handles nonlinear relationships well.  """

#    Implementing Random Forest Classification in Python
"""  Before implementing random forest classifier in Python let's first understand it's parameters.

n_estimators: Number of trees in the forest.
max_depth: Maximum depth of each tree.
max_features: Number of features considered for splitting at each node.
criterion: Function used to measure split quality ('gini' or 'entropy').
min_samples_split: Minimum samples required to split a node.
min_samples_leaf: Minimum samples required to be at a leaf node.
bootstrap: Whether to use bootstrap sampling when building trees (True or False).
Now that we know it's parameters we can start building it in python.  """

#    1. Import Required Libraries
#        We will be importing Pandas, matplotlib, seaborn and sklearn to build the model.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

#    2. Import Dataset
#      For this we'll use the Iris Dataset which is available within scikit learn. This dataset contains information about three types of Iris flowers and their respective features (sepal length, sepal width, petal length and petal width).
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df

#    3. Data Preparation
#      Here we will separate the features (X) and the target variable (y).
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#    4. Splitting the Dataset
#        We'll split the dataset into training and testing sets so we can train the model on one part and evaluate it on another.

"""    X_train, y_train: 80% of the data used to train the model.
       X_test, y_test: 20% of the data used to test the model.
       test_size=0.2: means 20% of data goes to testing.
       random_state=42: ensures you get the same split every time  """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#    5. Feature Scaling
#          Feature scaling ensures that all the features are on a similar scale which is important for some machine learning models. However Random Forest is not highly sensitive to feature scaling. But it is a good practice to scale when combining models.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#    6. Building Random Forest Classifier
#        We will create the Random Forest Classifier model, train it on the training data and make predictions on the test data.

"""      RandomForestClassifier(n_estimators=100, random_state=42) creates 100 trees (100 trees           balance accuracy and training time).
        classifier.fit(X_train, y_train) trains on training data.
        classifier.predict(X_test) predicts on test data.
        random_state=42 ensures reproducible results.    """
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#    7. Evaluation of the Model
#      We will evaluate the model using the accuracy score and confusion matrix.
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, 
            xticklabels=iris.target_names, yticklabels=iris.target_names)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#    8. Feature Importance
#        Random Forest Classifiers also provide insight into which features were the most important in making predictions. We can plot the feature importance.
feature_importances = classifier.feature_importances_

plt.barh(iris.feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()







