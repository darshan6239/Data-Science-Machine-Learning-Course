""" Duplicates are a common issues in real-world datasets that can negatively impact our analysis. They occur when identical rows or entries appear multiple times     in a dataset. Although they may seem harmless but they can cause problems in analysis if not fixed. Duplicates could happen due to:

    Data entry errors: When the same information is recorded more than once by mistake.
    Merging datasets: When combining data from different sources can lead to overlapping of data that can create duplicates.

    Why Duplicates Can Cause Problems?
    
    Skewed Analysis: Duplicates can affect our analysis results which leads to misleading conclusions such as an wrong average salary.
    Inaccurate Models: It can cause machine learning models to overfit which reduces their ability to perform well on new data.
    Increased Computational Costs: It consume extra computational power which slows down analysis and impacts workflow.
    Data Redundancy and Complexity: It make it harder to maintain accurate records and organize data and adds unnecessary complexity.

    Identifying Duplicates
    To manage duplicates the first step is identifying them in the dataset. Pandas offers various functions which are helpful to spot and remove duplicate rows.       Now we will see how to identify and remove duplicates using Python.

We will be using Pandas library for its implementation and will use a sample dataset below. """

import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob', 'David'],
    'Age': [25, 30, 25, 35, 30, 40],
    'City': ['New York', 'Los Angeles', 'New York', 'Chicago', 'Los Angeles', 'San Francisco']
}

df = pd.DataFrame(data)
df

#   1. Using duplicated() Method
#   The duplicated() method helps to identify duplicate rows in a dataset. It returns a boolean Series indicating whether a row is a duplicate of a previous row.

duplicates = df.duplicated()

duplicates

#   Removing Duplicates
#   Duplicates may appear in one or two columns instead of the entire dataset. In such cases, we can choose specific columns to check for duplicates.

#   1. Based on Specific Columns
#   Here we will specify columns i.e name and city to remove duplicates using drop_duplicates() .

df_no_duplicates_columns = df.drop_duplicates(subset=['Name', 'City'])
(df_no_duplicates_columns)

#   2. Keeping the First or Last Occurrence
#   By default drop_duplicates() keeps the first occurrence of each duplicate row. However, we can adjust it to keep the last occurrence instead.

df_keep_last = df.drop_duplicates(keep='last')
(df_keep_last)

""" Cleaning duplicates is an important step in ensuring data accuracy which improves model performance and optimizing analysis efficiency. """

""" USE EMPLOYEE.csv FILE TO GET OUTPUTS """
