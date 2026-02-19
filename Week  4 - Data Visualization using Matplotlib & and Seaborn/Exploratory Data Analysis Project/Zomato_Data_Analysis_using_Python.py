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
#        Before moving further we need to clean and process the data.

#    1. Convert the rate column to a float by removing denominator characters.
#        dataframe['rate']=dataframe['rate'].apply(handleRate): Applies the handleRate function to clean and convert each rating value in the 'rate' column.
def handleRate(value):
    value=str(value).split('/')
    value=value[0];
    return float(value)

dataframe['rate']=dataframe['rate'].apply(handleRate)
print(dataframe.head())

#      2. Getting summary of the dataframe use df.info().
dataframe.info()

#      3. Checking for missing or null values to identify any data gaps.
print(dataframe.isnull().sum())

#        There is no NULL value in dataframe.

#    Step 4: Exploring Restaurant Types
#        1. Let's see the listed_in (type) column to identify popular restaurant categories.
sns.countplot(x=dataframe['listed_in(type)'])
plt.xlabel("Type of restaurant")

#    2. Votes by Restaurant Type
#        Here we get the count of votes for each category.
grouped_data = dataframe.groupby('listed_in(type)')['votes'].sum()
result = pd.DataFrame({'votes': grouped_data})
plt.plot(result, c='green', marker='o')
plt.xlabel('Type of restaurant')
plt.ylabel('Votes')

#    Step 5: Identify the Most Voted Restaurant
#        Find the restaurant with the highest number of votes.
max_votes = dataframe['votes'].max()
restaurant_with_max_votes = dataframe.loc[dataframe['votes'] == max_votes, 'name']

print('Restaurant(s) with the maximum votes:')
print(restaurant_with_max_votes)

#    Step 6: Online Order Availability
#        Exploring the online_order column to see how many restaurants accept online orders.
sns.countplot(x=dataframe['online_order'])

#    Step 7: Analyze Ratings
#        Checking the distribution of ratings from the rate column.
plt.hist(dataframe['rate'],bins=5)
plt.title('Ratings Distribution')
plt.show()

#    Step 8: Approximate Cost for Couples
#        Analyze the approx_cost(for two people) column to find the preferred price range.
couple_data=dataframe['approx_cost(for two people)']
sns.countplot(x=couple_data)

#    Step 9: Ratings Comparison - Online vs Offline Orders
#        Compare ratings between restaurants that accept online orders and those that don't.
plt.figure(figsize = (6,6))
sns.boxplot(x = 'online_order', y = 'rate', data = dataframe)


#    Step 10: Order Mode Preferences by Restaurant Type
#        Find the relationship between order mode (online_order) and restaurant type (listed_in(type)).
#        pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0): Creates a pivot table counting restaurants by type and online order availability.
pivot_table = dataframe.pivot_table(index='listed_in(type)', columns='online_order', aggfunc='size', fill_value=0)
sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Heatmap')
plt.xlabel('Online Order')
plt.ylabel('Listed In (Type)')
plt.show()





