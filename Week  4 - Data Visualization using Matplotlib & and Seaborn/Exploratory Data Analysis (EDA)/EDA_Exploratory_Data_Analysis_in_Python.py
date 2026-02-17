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
