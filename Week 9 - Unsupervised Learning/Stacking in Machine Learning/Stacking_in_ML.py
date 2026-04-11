"""  Stacking is a ensemble learning technique where the final model known as the “stacked model" combines the predictions from multiple base models. The goal is to create a stronger model by using different models and combining them.  """


"""  Implementation of Stacking
Lets see its implementation step by step:  """

#  Step 1: Importing the required Libraries 
"We will import pandas, matplotlib and scikit learn for data handling, visualization and modeling."
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score


#  Step 2: Loading the Dataset 
# We will load the dataset into a pandas DataFrame and separate features from the target variable.
"""
pd.read_csv(): Reads the dataset from a CSV file.
drop(): Removes the target column from features.
df['target']: Selects the target column for prediction.  """
df = pd.read_csv('heart.csv')    
                 
X = df.drop('target', axis = 1)
y = df['target']

df.head()

#  Step 3: Splitting the Data into Training and Testing Sets
"""  We will split the dataset into training and testing sets so we can train models and evaluate their performance.

train_test_split(): Splits data into train and test sets.
test_size = 0.2: Specifies that 20% of the data should be used for testing, leaving 80% for training.
random_state = 42: Ensures reproducibility by setting a fixed seed for random number generation.  """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#   Step 4: Standardizing the Data
"  We will standardize numerical features so they have a mean of 0 and standard deviation of 1. This helps some models perform better.  "

  """  StandardScaler(): Standardizes features.
  fit_transform(): Learns scaling parameters from training data and applies them.
  transform(): Applies learned scaling to test data.
  var_transform: Specifies the list of feature columns that need to be standardized.
  X_train[var_transform]: Applies the fit_transform method to standardize the selected columns in the training data.
  X_test[var_transform]: Applies the transform method to standardize the corresponding columns in the test data using the scaling parameters from the training data.  """
sc = StandardScaler()  

var_transform = ['thalach', 'age', 'trestbps', 'oldpeak', 'chol']
X_train[var_transform] = sc.fit_transform(X_train[var_transform])   
X_test[var_transform] = sc.transform(X_test[var_transform])           

X_train.head()

#  Step 5: Building First Layer Estimators 
#  We will create base models that will form the first layer of our stacking model. For this example we’ll use K-Nearest Neighbors classifier and Naive Bayes classifier.

"""  KNeighborsClassifier(): A model based on nearest neighbors.
GaussianNB(): A Naive Bayes classifier assuming Gaussian distribution.  """
KNC = KNeighborsClassifier()   
NB = GaussianNB()

#    Step 6: Training and Evaluating KNeighborsClassifier
#    We will Train the KNN model and check its accuracy on the test set.
"""  
fit(): Trains the model.
predict(): Makes predictions on test data.
accuracy_score(): Calculates accuracy  """
model_kNeighborsClassifier = KNC.fit(X_train, y_train)  
pred_knc = model_kNeighborsClassifier.predict(X_test)

acc_knc = accuracy_score(y_test, pred_knc)  
print('Accuracy Score of KNeighbors Classifier:', acc_knc * 100)

#  Step 7: Training and Evaluating Naive Bayes Classifier
"  Similarly, we will train the Naive Bayes model and check its accuracy.  "
model_NaiveBayes = NB.fit(X_train, y_train)
pred_nb = model_NaiveBayes.predict(X_test)

acc_nb = accuracy_score(y_test, pred_nb)
print('Accuracy of Naive Bayes Classifier:', acc_nb * 100)

#  Step 8: Implementing the Stacking Classifier 
"  Now, we will combine the base models using a Stacking Classifier. The meta-model will be a logistic regression model which will take the predictions of KNN and Naive Bayes as input.  "

  """  StackingClassifier(): Combines base models and a meta-model.
  classifiers: List of base learners.
  meta_classifier: Model that learns from base learners’ predictions.
  use_probas=True: Passes probability outputs to the meta-model instead of class labels.  """
base_learners = [
    KNeighborsClassifier(),
    GaussianNB()
]
meta_model = LogisticRegression()

stacking_model = StackingClassifier(classifiers=base_learners, meta_classifier=meta_model, use_probas=True)

#  Step 9: Training Stacking Classifier  
"  Next we will rain the stacking classifier and evaluate its accuracy.  "
model_stack = stacking_model.fit(X_train, y_train)   
pred_stack = model_stack.predict(X_test)

acc_stack = accuracy_score(y_test, pred_stack)
print('Accuracy Score of Stacked Model:', acc_stack * 100)








