""" C - Create 
    R - Read
    U - Update
    D - Delete """

""" 1. Create: Creating Dataframe """
#   Creating a dataset in Pandas means building a DataFrame which is the main data structure in Pandas. We can create a DataFrame using various methods like reading from a file or directly creating one from Python objects like dictionaries, lists or arrays

""" 1. Creating a DataFrame from a Dictationary """
#   This is one of the easiest and most commonly used methods to create a dataset in Pandas
import pandas as pd
data = {
    "Name": ["Ansh", "Sahil", "Ram"],
    "Age": [21, 20, 41],
    "City": ["Moradabad", "New Delhi", "Chennai"]
}
df = pd.DataFrame(data)
print(df)

""" 2. Creating a DataFrame from Lists """
#   We can also create a DataFrame by combining lists.

import pandas as pd

names = ["Akshit", "Uday", "Sam"]
ages = [25, 30, 35]
cities = ["Gurugram", "New Delhi", "Chicago"]

df = pd.DataFrame({
    "Name": names,
    "Age": ages,
    "City": cities
})

print(df)

""" 3. Creating a DataFrame from a CSV File """
#   We can also create a DataFrame by reading an external file like a CSV. Here we used the random car.csv data.

import pandas as pd

df = pd.read_csv("/content/CAR.csv")
print(df.head())

""" 2. Read: Reading Dataframe """
#   Now that we’ve created a dataset using the Create operation, lets see by using the Read operation. This step is all about accessing and understanding our data. Pandas provides simple methods to view our dataset, check its structure and analyze its contents.

""" 1. Viewing Rows in a DataFrame
       head(n): Displaying the First Few Rows
       tail(n): Displaying the Last Few Rows """"

import pandas as pd

data = {"Name": ["Eve", "Jack", "Charlie", "Henry", "John"],
        "Age": [25, 30, 35, 40, 45],
        "City": ["NY", "LA", "SF", "Houston", "Seattle"]}
df = pd.DataFrame(data)

print(df.head(3))

print()

print(df.tail(2))

""" 2. Exploring Columns of the dataset """
print(df.columns)

""" 3. Checking Data Types with dtype """
#   We use df.types to check the particular data type of the columns we have for further operations
print(df.dtypes)

""" 4. Generating Descriptive Statistics with describe() """
#   This is a very important command used in pandas to check the overall statistics for the numerical data so that we can make predictions and move easily in our data exploration.
print(df.describe())

""" 5. Filtering Columns """
#   Accessing a single Column.
print(df["Name"])

""" 6. Accessing Multiple columns """
print(df[["Name", "City"]])

""" 5. Finding Unique Values in a Column """
#   Finding unique tends to provide the non-duplicates values in our columns
print(df["city"].unique())

""" 6. Filtering Rows (Conditional Filtering) """
#   Single Condition Filtering.
print(df[df["Age"] > 30])

""" 7. Filtering with Multiple Conditions (AND/OR Logic) """
print(df[(df["Age"] > 30) & (df["City"] == "SF")])

print(df[(df["Age"] > 30) | (df["City"] == "LA")])

""" 8. Indexing in Pandas """
#   Integer-Based Indexing with iloc.
print(df.iloc[0])

""" 9. Accessing Rows and Cells """
print(df.iloc[0, 2])

""" 10. Slicing Rows """
print(df.iloc[1:3])

""" 11. Label-Based Indexing """
df.set_index("Name", inplace=True)

""" 12. Setting an Index and Accessing Rows by Labels """
print(df.loc["Alice"])

""" 3. Update: Modifying Data in Pandas """
#   Update operation allows us to modify existing data within a DataFrame. Whether we're changing specific values, updating entire columns or applying conditions to update data, Pandas makes it simple.
#   We will use the following dataset for the update operations.

import pandas as pd

data = {'Name': ['Eve', 'Jack', 'Charlie', 'Henry', 'John'],
        'Age': [25, 30, 35, 40, 45],
        'City': ['NY', 'LA', 'SF', 'Houston', 'Seattle']}
df = pd.DataFrame(data)
df

"""1. Updating a Single Value: 
   We can update a single value in a specific row and column using loc or iloc."""
df.loc[df['Name'] == 'Jack', 'Age'] = 42

print(df)

""" 2. Updating an Entire Column: 
    We can update an entire column by assigning a new list, series or value."""
df['City'] = ['Boston', 'Chicago', 'LA', 'Austin', 'Miami']
df

""" 3. Updating Based on a Condition: 
    We can apply conditions to update values in a DataFrame."""
df.loc[df['City'] == 'LA', 'Age'] = 31
df

""" 4. Delete: Removing Data in Pandas """
# Delete operation allows us to remove data from a DataFrame. We can drop rows, columns or specific values providing flexibility in cleaning and manipulating datasets. For the delete operations we will use the dataset below.

import pandas as pd

data = {'Name': ['Eve', 'Jack', 'Charlie', 'Henry', 'John'],
        'Age': [25, 30, 35, 40, 45],
        'City': ['NY', 'LA', 'SF', 'Houston', 'Seattle']}
df = pd.DataFrame(data)

print(df)

#   1. Delete a Column: We can delete a column using the drop() method.
df = df.drop('City', axis=1)
print(df)

#   2. Delete a Row: Similarly we can delete rows by specifying the index.
df = df.drop(2, axis=0) 
print(df)

#   3. Delete Rows Based on Condition: We can delete rows based on conditions.
df = df[df['Age'] != 35]
print(df)

#   4. Delete entire dataset: To delete the entire DataFrame, we can use the del statement or reassign it to an empty DataFrame.
df = pd.DataFrame(data)
del df

""" USE THE FILE FOR THIS - CAR.csv """







