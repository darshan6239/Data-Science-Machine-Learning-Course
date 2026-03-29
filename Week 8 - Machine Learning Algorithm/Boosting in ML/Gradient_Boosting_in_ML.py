"""    Gradient Boosting is a boosting algorithm and here each new model is trained to minimize the loss function such as mean squared error or cross-entropy of the previous model using gradient descent. In each iteration the algorithm computes the gradient of the loss function with respect to predictions and then trains a new weak model to predict this gradient. Predictions of the new model are then added to the ensemble (all models prediction) and the process is repeated until a stopping criterion is met.    """

#    Implementing Gradient Boosting for Classification and Regression
""""  Here are two examples to demonstrate how Gradient Boosting works for both classification and regression. But before that let's understand gradient boosting parameters.

  n_estimators: This specifies the number of trees (estimators) to be built. A higher value typically improves model performance but increases computation time.
  learning_rate: This is the shrinkage parameter. It scales the contribution of each tree.
  random_state: It ensures reproducibility of results. Setting a fixed value for random_state ensure that you get the same results every time you run the model.
  max_features: This parameter limits the number of features each tree can use for splitting. It helps prevent overfitting by limiting the complexity of each tree and promoting diversity in the model.
Now we start building our models with Gradient Boosting.  """

#    Example 1: Classification
"""  We use Gradient Boosting Classifier to predict digits from Digits dataset.

  Import the necessary libraries
  Setting SEED for reproducibility
  Load the digit dataset and split it into train and test.
  Instantiate Gradient Boosting classifier and fit the model.
  Predict the test set and compute the accuracy score.  """
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

SEED = 23

X, y = load_digits(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = SEED)

gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=100,
                                 max_features=5 )
                                 
gbc.fit(train_X, train_y)

pred_y = gbc.predict(test_X)

acc = accuracy_score(test_y, pred_y)
print("Gradient Boosting Classifier accuracy is : {:.2f}".format(acc))

#    Example 2: Regression
"""  We use Gradient Boosting Regressor on the Diabetes dataset to predict continuous values:

  Import the necessary libraries
  Setting SEED for reproducibility
  Load the diabetes dataset and split it into train and test.
  Instantiate Gradient Boosting Regressor and fit the model.
  Predict on the test set and compute RMSE.  """

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

SEED = 23

X, y = load_diabetes(return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = SEED)

gbr = GradientBoostingRegressor(loss='absolute_error',
                                learning_rate=0.1,
                                n_estimators=300,
                                max_depth = 1, 
                                random_state = SEED,
                                max_features = 5)

gbr.fit(train_X, train_y)

pred_y = gbr.predict(test_X)

test_rmse = mean_squared_error(test_y, pred_y) ** (1 / 2)

print('Root mean Square error: {:.2f}'.format(test_rmse))
