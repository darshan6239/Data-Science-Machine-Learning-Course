"""  Gradient Boosting is an ensemble learning method and it works by sequentially adding decision trees where each tree tries to improve the model's performance by focusing on the errors made by the previous trees and reducing those with help of gradient descent. While Gradient Boosting performs well for improving model accuracy fine-tuning its hyperparameters can significantly improve its performance and prevent overfitting.  """


"""    Gradient Boosting Hyperparameter Tuning in Python
Scikit-learn is a popular python library that provides useful tools for hyperparameter tuning that can help improve the performance of Gradient Boosting models. Hyperparameter tuning is the process of selecting the best parameters to maximize the efficiency and accuracy of the model. We'll explore three common techniques: GridSearchCV, RandomizedSearchCV and Optuna.  """

#    We will use Titanic dataset for demonstration.

"""  Classification Model without Tuning
In this implementation, a Gradient Boosting Classifier is applied to the Titanic dataset to predict passenger survival. The process involves data preprocessing, splitting the dataset into training and testing sets, and training the model. To focus on demonstrating the model’s behavior with its default configuration, hyperparameter tuning is not included.  """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic_data = pd.read_csv("train.csv")

titanic_data.fillna(0, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingClassifier()

gb_model.fit(X_train, y_train)

y_pred = gb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")



#    1. Hyperparameter Tuning using GridSeachCV
"""  GridSearchCV systematically tests all possible combinations of hyperparameters from a predefined grid to identify the best configuration for a model. It is most effective when the number of combinations is relatively small. With GradientBoostingClassifier(), we can pass the model into GridSearchCV and fit it on the training data to obtain the optimal parameters.

param_grid: A dictionary containing hyperparameters and their possible values. GridSearchCV will try every combination of these values to find the best-performing set of hyperparameters.
grid_search.fit(X_train, y_train): This line trains the Gradient Boosting model using all combinations of the hyperparameters defined in param_grid.
grid_search.best_estimator_: After completing the grid search this will return the Gradient Boosting model that has the best combination of hyperparameters from the search.
best_params: This stores the best combination of hyperparameters found during the grid search.  """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

titanic_data = pd.read_csv("train.csv")

titanic_data.fillna(0, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
}

gb_model = GradientBoostingClassifier()

grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)

print("Best Parameters:", best_params)
print(f"Best Model Accuracy: {accuracy_best}")

#    2. Hyperparameter Tuning using RandomizedSearchCV
"""  RandomizedSearchCV is used to tune hyperparameters by sampling random combinations from a predefined grid. Unlike GridSearchCV, it does not search all possibilities, making it faster and more practical when the parameter space is large. With GradientBoostingClassifier(), we can pass the model to RandomizedSearchCV and fit it on the training data to identify a strong set of parameters without evaluating every option.

  param_dist: it will randomly sample from this distribution to find the best-performing combination of hyperparameters.
  random_search.fit(X_train, y_train): This line trains the GradientBoostingClassifier model using random combinations of hyperparameters defined in param_dist.
  random_search.best_estimator_: This retrieves the model that has the best combination of hyperparameters found during the random search.
  best_params: This stores the best combination of hyperparameters found during the search.  """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

titanic_data = pd.read_csv("train.csv")

titanic_data.fillna(0, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'learning_rate': np.arange(0.01, 0.2, 0.01), 
    'n_estimators': [100, 200, 300, 400],  
    'max_depth': [3, 5, 7, 9],  
}

gb_model = GradientBoostingClassifier()

random_search = RandomizedSearchCV(estimator=gb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)

print("Best Parameters:", best_params)
print(f"Best Model Accuracy: {accuracy_best}")

#    3. Hyperparameter Tuning using Optuna
"""  Optuna is an efficient framework for hyperparameter tuning that searches for settings to improve model performance. The process begins with defining an objective function, which Optuna then attempts to minimize or maximize through iterative trials. Its flexibility allows it to work well across different models and tasks, including Gradient Boosting, where it can be used to identify the most effective hyperparameters.

  param_space: Defines the hyperparameter search space where Optuna samples values for n_estimators, learning_rate and max_depth within the specified ranges.
  objective(trial): The objective function that it tries to minimize. It trains the GradientBoostingClassifier with different hyperparameters, calculates the accuracy and returns the inverse of accuracy.
  study.optimize(objective, n_trials=50): This runs the optimization process for 50 trials exploring the hyperparameter space and finding the best-performing combination of parameters.
  study.best_params: Returns the best combination of hyperparameters found during the optimization process.
  best_model_optuna.fit(X_train, y_train): Fits the GradientBoostingClassifier model using the best hyperparameters found by it.  """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import optuna

titanic_data = pd.read_csv("train.csv")

titanic_data.fillna(0, inplace=True)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = titanic_data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    param_space = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
    }

    gb_model = GradientBoostingClassifier(**param_space, validation_fraction=0.1, n_iter_no_change=5, random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred = gb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return 1.0 - accuracy 

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

best_params_optuna = study.best_params
best_model_optuna = GradientBoostingClassifier(**best_params_optuna, validation_fraction=0.1, n_iter_no_change=5, random_state=42)
best_model_optuna.fit(X_train, y_train)

y_pred_best_optuna = best_model_optuna.predict(X_test)

accuracy_best_optuna = accuracy_score(y_test, y_pred_best_optuna)
print(f"Best Model Accuracy (Optuna): {accuracy_best_optuna}")



