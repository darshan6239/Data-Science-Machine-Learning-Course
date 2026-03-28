"""  Random Forest is one of the most popular machine learning algorithms used for both classification and regression tasks. It works by building multiple decision trees and combining their outputs to improve accuracy and control overfitting. While Random Forest is a robust model, fine-tuning its hyperparameters such as the number of trees, maximum depth and feature selection can improve its prediction and performance.  """


"""  Random Forest Hyperparameter Tuning using Sklearn
    Scikit-learn offers tools for hyperparameter tuning which can help improve the performance of machine learning models. Hyperparameter tuning involves selecting the best set of parameters for a given model to maximize its efficiency and accuracy. We will explore two commonly used techniques for hyperparameter tuning: GridSearchCV and RandomizedSearchCV.

    Both methods are essential for automating the process of fine-tuning machine learning models and we will examine how each works and when to use them. Below is the code with random forest working on heart disease prediction.  """
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = pd.read_csv("heart.csv")
data.head(7)

X = data.drop("target", axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(
    n_estimators=100, 
    max_features="sqrt", 
    max_depth=6, 
    max_leaf_nodes=6
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_pred, y_test))

#    1. Hyperparameter Tuning using GridSearchCV
"""  First let's use GridSearchCV to obtain the best parameters for the model. It is a hyperparameter tuning method in Scikit-learn that exhaustively searches through all possible combinations of parameters provided in the param_grid. For that we will pass RandomForestClassifier() instance to the model and then fit the GridSearchCV using the training data to find the best parameters.

    param_grid: A dictionary containing hyperparameters and their possible values. GridSearchCV will try every combination of these values to find the best-performing set of hyperparameters.
    grid_search.fit(X_train, y_train): This trains the model on the training data (X_train, y_train) for every combination of hyperparameters defined in param_grid.
    grid_search.best_estimator_: After completing the grid search, this will print the RandomForest model that has the best combination of hyperparameters from the search.  """
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)


#    Updating the Model
#    Now we will update the parameters of the model by those which are obtained by using GridSearchCV.
model_grid = RandomForestClassifier(max_depth=3,
                                    max_features="log2",
                                    max_leaf_nodes=3,
                                    n_estimators=50)
model_grid.fit(X_train, y_train)
y_pred_grid = model.predict(X_test)
print(classification_report(y_pred_grid, y_test))


#    2. Hyperparameter Tuning using RandomizedSearchCV
"""  RandomizedSearchCV performs a random search over a specified parameter grid. It randomly selects combinations and evaluates the model often leading to faster results especially when there are many hyperparameters.

Now let's use RandomizedSearchCV to obtain the best parameters for the model. For that we will pass RandomFoestClassifier() instance to the model and then fit the RandomizedSearchCV using the training data to find the best parameters.

    param_grid specifies the hyperparameters that you want to tune similar to the grid in GridSearchCV.
    fit(X_train, y_train) trains the model using the training data.
    best_estimator_ shows the model with the best combination of hyperparameters found by the search process.  """
random_search = RandomizedSearchCV(RandomForestClassifier(),
                                   param_grid)
random_search.fit(X_train, y_train)
print(random_search.best_estimator_)


#    Updating the model
#    Now we will update the parameters of the model by those which are obtained by using RandomizedSearchCV.
model_random = RandomForestClassifier(max_depth=3,
                                      max_features='log2',
                                      max_leaf_nodes=6,
                                      n_estimators=100)
model_random.fit(X_train, y_train)
y_pred_rand = model.predict(X_test)
print(classification_report(y_pred_rand, y_test))

