"""    Customer Default Prediction is used by many banks and loan lenders to determine whether a person will be able to return the money they lend them or not. For this we be using AdaBoost which is an ensemble learning technique that combines multiple weak classifiers to create a strong classifier. The algorithm works by iteratively training sequence of classifiers each focusing on correcting the errors made by the previous one by assigning weights to the misclassified instances.  """

"""  1. Importing Libraries
We will import NumPy, pandas, seaborn, matplotlib and Scikit learn library in python.  """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

"""  2. Loading the Dataset
You can download dataset from here.  """
df = pd.read_csv('/content/LoanDataset---LoansDatasest.csv')
df.head()

"""  3. Handling Missing Values
Here we are handle missing values for numerical columns by filling them with the mean.

Numeric columns (like income, age, etc.) are filled with their mean value.
Categorical columns (like job title, loan intent) are filled with the most common value (mode).
This prevents errors during model training due to missing data.  """
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

"""  4. Clean Numeric Columns with Comma Values
Sometimes numeric columns are stored as strings with commas (e.g., "25,000"). This step removes commas and converts them to numeric format. If conversion fails, the value becomes NaN (and is handled again below).  """
# Remove commas and convert strings to numbers
columns_to_clean = ['customer_income', 'loan_amnt']
for col in columns_to_clean:
    df[col] = df[col].replace({',': ''}, regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

"""  5. Re-check for Any New Missing Numeric Values
After cleaning, we double-check that any new NaN values (from failed conversion) are replaced with the column mean. This ensures no gaps in numeric data.  """
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

"""  6. Encode Categorical Columns to Numeric Labels
Machine learning models can’t understand text labels. This step uses Label Encoding to convert categories (e.g., 'RENT', 'OWN', 'MORTGAGE') into numbers (e.g., 0, 1, 2) so models can use them for training.    """
label_encoder = LabelEncoder()

df['home_ownership'] = label_encoder.fit_transform(df['home_ownership'])
df['loan_intent'] = label_encoder.fit_transform(df['loan_intent'])
df['loan_grade'] = label_encoder.fit_transform(df['loan_grade'])
df['historical_default'] = label_encoder.fit_transform(df['historical_default'])
df['Current_loan_status'] = label_encoder.fit_transform(df['Current_loan_status'])

"""  7. Feature Selection
We will perform feature selection by removing columns like customer_id that do not contribute to prediction and separating the target variable Current_loan_status. This keeps only useful features in X while y holds the outcome for prediction.  """
X = df.drop(['customer_id', 'Current_loan_status'], axis=1)
y = df['Current_loan_status']


"""  8. Splitting and Imputing the Data
We will split dataset for training and testing and after that we impute dataset. When cleaning data like customer_income and loan_amnt removing commas might cause some values to turn into NaN if there are any invalid characters. Imputation fills these gaps with meaningful values like mean to make sure the data is complete and ready for the model.  """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

"""  9. Model Training with AdaBoost
AdaBoostClassifier Initializes the AdaBoost model with 50 estimators (weak classifiers). We can adjust this parameter based on the complexity of the problem.  """
model = AdaBoostClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

"""  10. Model Evaluation
We will be evaluating model using accuracy and confusion matrix.  """
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()
print(f'Accuracy: {accuracy* 100: 4f}%')

