#   Installation
#   Before using Pandas, make sure it is installed:
pip install pandas

#   After the Pandas have been installed in the system we need to import the library. This module is imported using:
import pandas as pd

#   Data Structures in Pandas
#   Pandas provides two data structures for manipulating data which are as follows:

#   1. Pandas Series
#   A Pandas Series is one-dimensional labeled array capable of holding data of any type (integer, string, float, Python objects etc.). The axis labels are collectively called indexes. Series is created by loading the datasets from existing storage which can be a SQL database, a CSV file or an Excel file.

import pandas as pd 
import numpy as np

s = pd.Series() 
print("Pandas Series: ", s) 
data = np.array(['g', 'e', 'e', 'k', 's']) 
  
s = pd.Series(data) 
print("Pandas Series:\n", s)

#   2. Pandas DataFrame
#   Pandas DataFrame is a two-dimensional data structure with labeled axes (rows and columns). It is created by loading the datasets from existing storage which can be a SQL database, a CSV file or an Excel file. It can be created from lists, dictionaries, a list of dictionaries etc.

import pandas as pd 
   
df = pd.DataFrame() 
print(df)
lst = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks'] 
  
df = pd.DataFrame(lst) 
print(df)

#   Operations in Pandas
#   1. Loading Data: This operation reads data from files such as CSV, Excel or JSON into a DataFrame. (You can find the file within the folder)

import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())

#   2. Viewing and Exploring Data: After loading data, it is important to understand its structure and content. This methods allow you to inspect rows, summary statistics and metadata.

print(df.info())

#   3. Handling Missing Data: Datasets often contain empty or missing values. Pandas provides functions to detect, remove or replace these values.

print(df.isnull().sum())
df = df.fillna(0)

#   4. Selecting and Filtering Data: This operation retrieves specific columns, rows or records that match a condition. It allows precise extraction of required information.

ages = df[df['age'] > 25]
print(ages)

#   5. Adding and Removing Columns: You can create new columns based on existing ones or delete unwanted columns from the DataFrame.

df['total'] = df['a'] + df['b']
print(df.head())

#   6. Grouping Data (GroupBy): Grouping allows you to organize data into categories and compute values for each group for example, sums, counts or averages.

res = df.groupby('category')['sales'].sum()
print(res)


