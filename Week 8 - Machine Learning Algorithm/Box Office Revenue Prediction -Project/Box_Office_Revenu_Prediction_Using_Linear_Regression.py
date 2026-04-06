"""  The objective of this project is to develop a machine learning model using Linear Regression to accurately predict the box office revenue of movies based on various available features. The model will be trained on a dataset containing historical movie data and will aim to identify key factors that impact revenue. By implementing data preprocessing, feature engineering, visualization and model evaluation techniques, this project seeks to:

Build a predictive model that can estimate the expected revenue of a movie prior to its release.
Provide insights into which features most influence box office success.
Compare linear regression performance with more advanced models (e.g., XGBoost) to assess predictive accuracy.  """


#    1. Importing Libraries and Dataset
"""  Core Libraries
    Pandas: For loading and exploring the dataset.
    NumPy:For working with numerical arrays and math operations.
Visualization
    Matplotlib and Seaborn: Used to plot data distributions, trends and model performance.
Preprocessing and Modeling
    train_test_split: Splits the data into training and validation sets.
    LabelEncoder: Converts categories like genres into numeric format.
    StandardScaler: Scales features for better model performance.
    CountVectorizer: Converts text data (e.g., genres) into numeric vectors.
    metrics: Offers tools for evaluating model accuracy.
Advanced Modeling
    XGBoost: A high-performance gradient boosting algorithm used for better predictions.
Utility
warnings.filterwarnings('ignore'): Hides unnecessary warning messages for cleaner output."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

#    2. Loading the dataset into a pandas DataFrame
"  We now load the dataset into a pandas DataFrame to begin analysis. You can download the dataset from here.  "
df = pd.read_csv('boxoffice.csv',
                 encoding='latin-1')
df.head()

#    2.1 Checking Dataset Size
"  Let's see how many rows and columns we have.  "
df.shape

#  2.2 Checking Data Types
"  We check the data types of each column and look for issues.  "
df.info()

#  3. Exploring the Dataset
"""  We take a quick look at statistical metrics (like mean, min, max) for each numeric column to understand the data distribution.

df.describe() gives a summary of the numeric columns (count, mean, standard deviation, min, max, etc.).
.T transposes the output for better readability ( rows become columns and vice versa ). """
df.describe().T

#  Since we are predicting only domestic revenue in this project, we are dropping world_revenue and opening_revenue columns from the dataframe.
to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)

#    3.1 Checking Missing Values
#    We calculate what percentage of values is missing in each column. isnull().sum() functions helps us identify columns with many missing entries.
df.isnull().sum() * 100 / df.shape[0]


#    4. Handling Missing Values
"""  We clean the data by removing or filling missing values appropriately.

We drop the budget column entirely, likely due to too many missing values.
Fill missing values in MPAA and genres columns using their most frequent values (mode).
Remove any remaining rows with missing values.
Finally, check if any null values remain; the result should be 0.  """
df.drop('budget', axis=1, inplace=True)

for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.dropna(inplace=True)

df.isnull().sum().sum()

#    4.1 Cleaning Numeric Columns Stored as Strings
"""  Some numeric columns might be stored as strings with special characters (like $ or ,). We need to remove these characters and convert the columns back to numeric format.

Remove the first character from 'domestic_revenue' (likely a $ sign).
Remove commas from numeric values (e.g., 1,000 to 1000).
Ensure the columns are properly converted to float types.
Use pd.to_numeric to handle any remaining non-numeric values gracefully to turn them into NaNs.  """
df['domestic_revenue'] = df['domestic_revenue'].astype(str).str[1:]

for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].astype(str).str.replace(',', '') 
    
    temp = (~df[col].isnull()) 
    df[temp][col] = df[temp][col].convert_dtypes(float) 

    df[col] = pd.to_numeric(df[col], errors='coerce')

#    5. Visualizing MPAA Rating Distribution
"""  We want to see how many movies fall under each MPAA rating category like PG, R, PG-13, etc. We will create a horizontal bar chart showing the count of movies in each MPAA rating.

plt.figure(figsize=(10, 5)) sets the size of the plot.
sb.countplot() from Seaborn automatically counts and plots the frequency of each category in the 'MPAA' column.
plt.show() displays the plot.  """
plt.figure(figsize=(10, 5))
sb.countplot(df['MPAA'])
plt.show()


#    5.1 Average Domestic Revenue by MPAA Rating
#    We group the dataset by the 'MPAA' rating category and calculate the mean (average) of the 'domestic_revenue' for each rating group.
df.groupby('MPAA')['domestic_revenue'].mean()

