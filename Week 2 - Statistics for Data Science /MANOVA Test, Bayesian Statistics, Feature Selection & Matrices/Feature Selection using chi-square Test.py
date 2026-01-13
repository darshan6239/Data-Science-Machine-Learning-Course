###Real-World Example: Customer Purchase Prediction We will use a real-world dataset from: Predict Customer Purchase Behavior Dataset. This dataset contains demographic information, purchasing habits and other important features to predict customer purchase behavior.###

#Step 1: Loading and Preparing the Dataset 
#We'll be using Pandas library for loading the dataset into a pandas DataFrame and Scikit Learn library to do Feature Selection.
#(Data Sheet is Attached with it outside) 

import pandas as pd
import sklearn

df = pd.read_csv('customer_purchase_behavior.csv')

print(df.head())
#        This step helps us get an initial idea of the data and understand the structure of the dataset.

#        Step 2: Data Summary
#        We summarize the dataset to understand its properties.

print(df.info())

print(df.describe(include='all'))

#       This provides an overview of the data types, missing values and statistical distributions.

#       Step 3: Data Cleaning
#       Cleaning the data ensures that it is suitable for analysis. We check and remove missing values. This ensures our dataset is complete and ready for processing.

print(df.isnull().sum())

df = df.dropna()

print(df.isnull().sum())

#               Step 4: Feature Encoding
#               Since the Chi-Square test requires numerical input so we convert categorical variables into numbers using label encoding. This step ensures that categorical variables can be used in statistical tests and machine learning models.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_features = ['Gender', 'Age', 'AnnualIncome', 'ProductCategory']
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

df['Purchase'] = le.fit_transform(df['PurchaseStatus'])

#                 Step 5: Applying Chi-Square Test 
#                 Now we will apply the Chi-Square test to determine which features are most relevant to the target variable. This step helps identify the features with the strongest relationship to customer purchases.
from sklearn.feature_selection import chi2, SelectKBest

X = df[categorical_features]
y = df['Purchase']

selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

feature_scores = selector.scores_
selected_features = X.columns[selector.get_support()]

print("Feature Scores:", feature_scores)
print("Selected Features:", selected_features)
