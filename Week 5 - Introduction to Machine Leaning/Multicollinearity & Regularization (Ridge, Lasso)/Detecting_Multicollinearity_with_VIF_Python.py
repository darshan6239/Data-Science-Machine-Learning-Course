"""Multicollinearity occurs when two or more independent variables are highly correlated which leads to unstable coefficient estimates and reduces model reliability. This makes it difficult to identify the individual effect of each predictor on the dependent variable. The Variance Inflation Factor (VIF) is used to detect multicollinearity in regression analysis. In this article, we’ll see VIF and how to use it in Python to identify multicollinearity."""

""" To detect multicollinearity in regression analysis we can implement the Variance Inflation Factor (VIF) using the statsmodels library. This function calculates the VIF value for each feature in the dataset helping us identify multicollinearity.

Syntax :  statsmodels.stats.outliers_influence.variance_inflation_factor(exog, exog_idx) """

#    USE THE DATASET NAMED -BMI, OUTSIDE THE FOLDER 

import pandas as pd 
data = pd.read_csv('/content/BMI.csv')
print(data.head())

#  ---------------------------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

data['Gender'] = data['Gender'].map({'Male':0, 'Female':1})

X = data[['Gender', 'Height', 'Weight']]

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
print(vif_data)

#  ---------------------------------------------------------------------------------

""" High VIF values for Height and Weight shows strong multicollinearity between these two variables which makes sense because a person’s height influences their weight. Detecting such relationships helps us to understand and improve the stability of our regression models.

What to do if VIF is High?
Here are several effective strategies to address high VIF values and improve model performance:

1. Removing Highly Correlated Features: Drop one of the correlated features, the one which is less important or with a higher VIF. Removing such features reduces redundancy and improves model interpretability and stability.

2. Combining Variables or Using Dimensionality Reduction Techniques

Create new variables such as BMI from height and weight.
Use Principal Component Analysis (PCA) to convert correlated features into uncorrelated components while keeping most of the data information. """
