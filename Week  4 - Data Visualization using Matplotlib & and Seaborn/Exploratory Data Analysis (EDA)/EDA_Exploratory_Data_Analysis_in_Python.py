""" Exploratory Data Analysis (EDA) is a important step in data analysis which focuses on understanding patterns, trends and relationships through statistical tools and visualizations. Python offers various libraries like pandas, numPy, matplotlib, seaborn and plotly which enables effective exploration and insights generation to help in further modeling and analysis. Some common EDA techniques are:

Data Inspection: Check the size of the dataset, how it is organized, the types of data it contains and basic summary values.
Handling Missing and Duplicate Data: Find and fix empty values or repeated rows to keep the data clean.

Univariate Analysis: Study one variable at a time to understand its distribution, trend and outliers.
Bivariate Analysis: Compare two variables to see how they are related.
Multivariate Analysis: Analyze three or more variables together to understand deeper relationships.

Key Steps for Exploratory Data Analysis (EDA)
Lets see various steps involved in Exploratory Data Analysis """

#   Step 1: Importing Required Libraries
#   We need to install Pandas, NumPy, Matplotlib and Seaborn libraries in python to proceed further. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#   Step 2: Reading Dataset
#   Lets read the dataset using pandas.
#   DATASET IS IN FOLDER USE IT 
df = pd.read_csv("/content/WineQT.csv")
print(df.head())

#   Step 3: Analyzing the Data
#   1. df.shape(): This function is used to understand the number of rows (observations) and columns (features) in the dataset. This gives an overview of the dataset's size and structure.
df.shape

#   2. df.info(): This function helps us to understand the dataset by showing the number of records in each column, type of data, whether any values are missing and how much memory the dataset uses.
df.info()

#  3. df.describe().T: This method gives a statistical summary of the DataFrame (Transpose) showing values like count, mean, standard deviation, minimum and quartiles for each numerical column. It helps in summarizing the central tendency and spread of the data.
df.describe().T

#   4. df.columns.tolist(): This converts the column names of the DataFrame into a Python list making it easy to access and manipulate the column names.
df.columns.tolist()

#   Step 4 : Checking Missing Values
#   df.isnull().sum(): This checks for missing values in each column and returns the total number of null values per column helping us to identify any gaps in our data.
df.isnull().sum()

#   Step 5 : Checking for the duplicate values
#   df.nunique(): This function tells us how many unique values exist in each column which provides insight into the variety of data in each feature.
df.nunique()


#   Step 6: Univariate Analysis
#   In Univariate analysis plotting the right charts can help us to better understand the data making the data visualization so important.

#   1. Bar Plot for evaluating the count of the wine with its quality rate.
quality_counts = df['quality'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts, color='deeppink')
plt.title('Count Plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

#   2. Kernel density plot for understanding variance in the dataset
sns.set_style("darkgrid")

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

plt.tight_layout()
plt.show()

