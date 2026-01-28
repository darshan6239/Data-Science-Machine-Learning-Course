""" Creating a Pandas DataFrame """
#   Pandas allows us to create a DataFrame from many data sources. We can create DataFrames directly from Python objects like lists and dictionaries or by reading data from external files like CSV, Excel or SQL databases.

#  Here are some ways by which we create a dataframe:


""" 1. Creating DataFrame using a List """
#   If we have a simple list of data, we can easily create a DataFrame by passing that list to the pd.DataFrame() function.

import pandas as pd
lst = ['Geeks', 'For', 'Geeks', 'is', 
            'portal', 'for', 'Geeks']

df = pd.DataFrame(lst)
print(df)


""" 2. Creating DataFrame from dict of ndarray/lists """"
#   We can create a DataFrame from a dictionary where the keys are column names and the values are lists or arrays.

#   All arrays/lists must have the same length.
#   If an index is provided, it must match the length of the arrays.
#   If no index is provided, Pandas will use a default range index (0, 1, 2, …).

import pandas as pd
 
data = {'Name':['Tom', 'nick', 'krish', 'jack'],
        'Age':[20, 21, 19, 18]}
 
df = pd.DataFrame(data)
 
print(df)

""" Working With Rows and Columns in Pandas DataFrame """
#  We can perform basic operations on rows/columns like selecting, deleting, adding and renaming.

""" 1. Column Selection """
#   In Order to select a column in Pandas DataFrame, we can either access the columns by calling them by their columns name.

import pandas as pd
 
data = {'Name':['Jai', 'Princi', 'Gaurav', 'Anuj'],
        'Age':[27, 24, 22, 32],
        'Address':['Delhi', 'Kanpur', 'Allahabad', 'Kannauj'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd']}
 
df = pd.DataFrame(data)
 
print(df[['Name', 'Qualification']])

""" 2. Row Selection """
#   Pandas provide unique methods for selecting rows from a Data frame. DataFrame.loc[] method is used for label-based selection

#   Here we’ll be using nba.csv dataset in below examples for better understanding.

import pandas as pd
 
data = pd.read_csv("/content/nba.csv", index_col ="Name")

first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]
 
print(first, "\n\n\n", second)

""" Indexing and Selecting Data in Pandas DataFrame """
#   Indexing in pandas means simply selecting particular rows and columns of data from a DataFrame. It allows us to access subsets of data such as:

""" 1) Selecting all rows and some columns.
    2) Selecting some rows and all columns.
    3) Selecting a specific subset of rows and columns.

Indexing can also be known as Subset Selection."""

""" 1. Indexing a Dataframe using indexing operator [] """ 
#   The indexing operator [] is the basic way to select data in Pandas. We can use this operator to access columns from a DataFrame. This method allows us to retrieve one or more columns. The .loc and .iloc indexers also use the indexing operator to make selections.

#   In order to select a single column, we simply put the name of the column in-between the brackets.

import pandas as pd
 
data = pd.read_csv("/content/nba.csv", index_col ="Name")

first = data["Age"]
 
print(first)

""" 2. Indexing a DataFrame using .loc[ ] """
#   The .loc method is used to select data by label. This means it uses the row and column labels to access specific data points. .loc[] is versatile because it can select both rows and columns simultaneously based on labels.

#   In order to select a single row using .loc[], we put a single row label in a .loc function.

import pandas as pd
 
data = pd.read_csv("/content/nba.csv", index_col ="Name")

first = data.loc["Avery Bradley"]
second = data.loc["R.J. Hunter"]
 
print(first, "\n\n\n", second)

""" 3. Indexing a DataFrame using .iloc[ ] """ 
#   The .iloc() method allows us to select data based on integer position. Unlike .loc[] (which uses labels) .iloc[] requires us to specify row and column positions as integers (0-based indexing).

#   In order to select a single row using .iloc[], we can pass a single integer to .iloc[] function.

import pandas as pd
 
data = pd.read_csv("/content/nba.csv", index_col ="Name")
 
row2 = data.iloc[3] 
 
print(row2)

""" Working with Missing Data """
#   Missing Data can occur when no information is available for one or more items or for an entire row/column. In Pandas missing data is represented as NaN (Not a Number). Missing data can be problematic in real-world datasets where data is incomplete. Pandas provides several methods to handle such missing data effectively:

""" 1. Checking for Missing Values using isnull() and notnull() """
#   To check for missing values (NaN) we can use two useful functions:

"""   isnull(): It returns True for NaN (missing) values and False otherwise.
      notnull(): It returns the opposite, True for non-missing values and False for NaN values. """"

import pandas as pd
 
import numpy as np
 
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}

df = pd.DataFrame(dict)
 
print(df.isnull())

""" 2. Filling Missing Values using fillna(), replace() and interpolate() """
#   In order to fill null values in a datasets, we use fillna(), replace() and interpolate() function these function replace NaN values with some value of their own. All these function help in filling a null values in datasets of a DataFrame. Interpolate() function is used to fill NA values in the dataframe but it uses various interpolation technique to fill the missing values rather than hard-coding the value.

import pandas as pd
 
import numpy as np
 
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, 45, 56, np.nan],
        'Third Score':[np.nan, 40, 80, 98]}
df = pd.DataFrame(dict)
 
print(df.fillna(0))

""" 3. Dropping Missing Values using dropna() """
#   If we want to remove rows or columns with missing data we can use the dropna() method. This method is flexible which allows us to drop rows or columns depending on the configuration.

import pandas as pd
 
import numpy as np
 
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score':[52, 40, 80, 98],
        'Fourth Score':[np.nan, np.nan, np.nan, 65]}
 
df = pd.DataFrame(dict)
   
print(df)

# Now we drop rows with at least one Nan value (Null value).
import pandas as pd
 
import numpy as np
 
dict = {'First Score':[100, 90, np.nan, 95],
        'Second Score': [30, np.nan, 45, 56],
        'Third Score':[52, 40, 80, 98],
        'Fourth Score':[np.nan, np.nan, np.nan, 65]}
df = pd.DataFrame(dict)
 
print(df.dropna())


""" Iterating over rows and columns """
#   Iteration refers to the process of accessing each item one at a time. In Pandas, it means iterating through rows or columns in a DataFrame to access or manipulate the data. We can iterate over rows and columns to extract values or perform operations on each item.

""" 1. Iterating Over Rows """
#   There are several ways to iterate over the rows of a Pandas DataFrame and three common methods are:

"""   1) iteritems()
      2) iterrows()
      3) itertuples()
      
Each method provides different ways to iterate over the rows which depends on our specific needs. """

import pandas as pd
  
dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score':[90, 40, 80, 98]}
 

df = pd.DataFrame(dict)
 
print(df)

""" 2. Iterating Over Columns """
#   In order to iterate over columns, we need to create a list of dataframe columns and then iterating through that list to pull out the dataframe columns.

import pandas as pd
   
dict = {'name':["aparna", "pankaj", "sudhir", "Geeku"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score':[90, 40, 80, 98]}
  
df = pd.DataFrame(dict)
 
print(df)

#   Now here we iterate through columns in order to iterate through columns we first create a list of dataframe columns and then iterate through list.

columns = list(df)
 
for i in columns:
    print (df[i][2])


""" 
DataFrame Methods for Working with Data
Pandas has a variety of methods for manipulating data in a DataFrame. Here's are some useful DataFrame methods:

---  FUNCTION	DESCRIPTION  ---
index()	- Method returns index (row labels) of the DataFrame
insert() - 	Method inserts a column into a DataFrame
add() -	Method returns addition of dataframe and other, element-wise (binary operator add)
sub() - 	Method returns subtraction of dataframe and other element-wise (binary operator sub)
mul() -	Method returns multiplication of dataframe and other, element-wise (binary operator mul)
div() - Method returns floating division of dataframe and other element-wise (binary operator truediv)
unique() -	Method extracts the unique values in the dataframe
nunique() -	Method returns count of the unique values in the dataframe
value_counts() - Method counts the number of times each unique value occurs within the Series
columns() -	Method returns the column labels of the DataFrame
axes() - Method returns a list representing the axes of the DataFrame
isnull()	- Method creates a Boolean Series for extracting rows with null values
notnull() -	Method creates a Boolean Series for extracting rows with non-null values
isin() -	Method extracts rows from a DataFrame where a column value exists in a predefined collection
dtypes() - Method returns a Series with the data type of each column. The result’s index is the original DataFrame’s columns
astype() -	Method converts the data types in a Series
values() - Method returns a Numpy representation of the DataFrame i.e only the values in the DataFrame will be returned, the axes labels will be removed
sort_values() -	Method sorts a data frame in Ascending or Descending order of passed Column
sort_index() -	Method sorts the values in a DataFrame based on their index positions or labels instead of their values but sometimes a data frame is made out of two or more data frames and hence later index can be changed using this method
loc[] -	Method retrieves rows based on index label
iloc[] -	Method retrieves rows based on index position
ix[] -	Method retrieves DataFrame rows based on either index label or index position. This method combines the best features of the .loc[] and .iloc[] methods
rename() -	Method is called on a DataFrame to change the names of the index labels or column names
drop() -	Method is used to delete rows or columns from a DataFrame
pop() -	Method is used to delete rows or columns from a DataFrame
sample() -	Method pulls out a random sample of rows or columns from a DataFrame
nsmallest() -	Method pulls out the rows with the smallest values in a column
nlargest() -	Method pulls out the rows with the largest values in a column
shape() -	Method returns a tuple representing the dimensionality of the DataFrame
ndim() - Method returns an ‘int’ representing the number of axes / array dimensions.
Returns 1 if Series, otherwise returns 2 if DataFrame
dropna() - Method allows the user to analyze and drop Rows/Columns with Null values in different ways
fillna() - Method manages and let the user replace NaN values with some value of their own
rank() -	Values in a Series can be ranked in order with this method
query()	- Method is an alternate string-based syntax for extracting a subset from a DataFrame
copy() -	Method creates an independent copy of a pandas object
duplicated() - Method creates a Boolean Series and uses it to extract rows that have duplicate values
drop_duplicates() -	Method is an alternative option to identifying duplicate rows and removing them through filtering
set_index() - Method sets the DataFrame index (row labels) using one or more existing columns
reset_index() - Method resets index of a Data Frame. This method sets a list of integer ranging from 0 to length of data as index
where() - Method is used to check a Data Frame for one or more condition and return the result accordingly. By default, the rows not satisfying the condition are filled with NaN value """



 """ USE THE SAME FILE nba.csv FOR PROCESSING OF CODES """





