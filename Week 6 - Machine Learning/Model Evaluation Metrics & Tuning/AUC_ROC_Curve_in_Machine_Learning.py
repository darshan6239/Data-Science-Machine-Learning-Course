""" AUC-ROC curve is a graph used to check how well a binary classification model works. It helps us to understand how well the model separates the positive cases like people with a disease from the negative cases like people without the disease at different threshold level. It shows how good the model is at telling the difference between the two classes by plotting:

    True Positive Rate (TPR): how often the model correctly predicts the positive cases also known as Sensitivity or Recall.
    False Positive Rate (FPR): how often the model incorrectly predicts a negative case as positive.
    Specificity: measures the proportion of actual negatives that the model correctly identifies. It is calculated as 1 - FPR.

When to Use AUC-ROC
AUC-ROC is effective when:

    The dataset is balanced and the model needs to be evaluated across all thresholds.
    False positives and false negatives are of similar importance. """ 
    
#    Implementation using two different models
#    1. Installing Libraries
#    We will be importing numpy, pandas, matplotlib and scikit learn.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

#    2. Generating data and splitting data
#    Using an 80-20 split ratio, the algorithm creates artificial binary classification data with 20 features, divides it into training and testing sets and assigns a random seed to ensure reproducibility.
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#    3. Training the different models
#    To train the Random Forest and Logistic Regression models we use a fixed random seed to get the same results every time we run the code. First we train a logistic regression model using the training data. Then use the same training data and random seed we train a Random Forest model with 100 trees.
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

#    4. Predictions
#    Using the test data and a trained Logistic Regression model the code predicts the positive class's probability. In a similar manner, using the test data, it uses the trained Random Forest model to produce projected probabilities for the positive class.
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]
y_pred_rf = random_forest_model.predict_proba(X_test)[:, 1]

#    5. Creating a dataframe
#    Using the test data the code creates a DataFrame called test_df with columns labeled "True," "Logistic" and "RandomForest," add true labels and predicted probabilities from Random Forest and Logistic Regression models.
test_df = pd.DataFrame(
    {'True': y_test, 'Logistic': y_pred_logistic, 'RandomForest': y_pred_rf})

#    6. Plotting ROC Curve for models
#    Plot the ROC curve and compute the AUC for both Logistic Regression and Random Forest. The ROC curve compares models based on True Positive Rate vs False Positive Rate, while the red dashed line shows random guessing.
plt.figure(figsize=(7, 5))

for model in ['Logistic', 'RandomForest']:
    fpr, tpr, _ = roc_curve(test_df['True'], test_df[model])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Two Models')
plt.legend()
plt.show()







