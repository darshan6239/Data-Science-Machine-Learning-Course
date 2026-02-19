""" Implementation for Zomato Data Analysis using Python.
    Below steps are followed for its implementation. """

#    Step 1: Importing necessary Python libraries.
#    We will be using Pandas, Numpy, Matplotlib and Seaborn libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# USE THE DATA SET NAME - zomato-data.csv
#    Step 2: Creating the data frame.
dataframe = pd.read_csv("/content/Zomato-data-.csv")
print(dataframe.head())

#    Step 3: Data Cleaning and Preparation
#    Before moving further we need to clean and process the data.

#    1. Convert the rate column to a float by removing denominator characters.
#    dataframe['rate']=dataframe['rate'].apply(handleRate): Applies the handleRate function to clean and convert each rating value in the 'rate' column.
def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)

dataframe['rate']=dataframe['rate'].apply(handleRate)
print(dataframe.head())




