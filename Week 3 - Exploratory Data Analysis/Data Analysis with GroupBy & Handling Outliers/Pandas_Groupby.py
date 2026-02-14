"""  The groupby() function in Pandas is important for data analysis as it allows us to group data by one or more categories and then apply different functions to those groups. This technique is used for handling large datasets efficiently and performing operations like aggregation, transformation and filtration on grouped data. 

The groupby() operation is divided into three main steps: """

#   Step 1: Splitting Data into Groups
#   The splitting process refers to dividing the dataset into groups based on a particular condition or key. This can be done using the groupby() function by passing one or more columns as keys.

#   Methods for splitting data are as follows:
#   1. Group data by a single key: In order to group data with one key we pass only one key as an argument in groupby function. Here we will group the data by the Name column.

import pandas as pd
data1 = {'Name':['Jai', 'Anuj', 'Jai', 'Princi',
                 'Gaurav', 'Anuj', 'Princi', 'Abhi'],
        'Age':[27, 24, 22, 32,
               33, 36, 27, 32],
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj',
                   'Jaunpur', 'Kanpur', 'Allahabad', 'Aligarh'],
        'Qualification':['Msc', 'MA', 'MCA', 'Phd',
                         'B.Tech', 'B.com', 'Msc', 'MA']}
df = pd.DataFrame(data1)
print(df)


df.groupby('Name')
print(df.groupby('Name').groups)

#   Now we print the first entries in all the groups formed. 
gk = df.groupby('Name') 
gk.first()

#   2. Grouping data with multiple keys : In order to group data with multiple keys, we pass multiple keys in groupby function. Here we group a data of "Name" and "Qualification" together using multiple keys in groupby function. 
df.groupby(['Name', 'Qualification'])
print(df.groupby(['Name', 'Qualification']).groups)

#   3. Grouping data by sorting keys: Group keys are sorted by default using the groupby operation. Now we apply groupby() using sort.
df.groupby('Name')['Age'].sum()

#   Now we apply groupby() without using sort(we pass sort=False).
df.groupby(['Name'], sort=False)['Age'].sum()

#   4. Grouping data with object attributes: Grouping data with object attributes in Pandas allows us to treat the groups attribute as a dictionary where the keys are unique group values and values are the indices (row labels) corresponding to each group.
df.groupby('Name').groups

#   5. Iterating through groups: In order to iterate an element of groups, we can iterate through the object.
grp = df.groupby('Name')
for name, group in grp:
    print(name)
    print(group)
    print()


#   Now we iterate an element of group containing multiple keys.
grp = df.groupby(['Name', 'Qualification'])
for name, group in grp:
    print(name)
    print(group)
    print()

#   6. Selecting a group: In order to select a group, we can select group using GroupBy.get_group(). We can select a group by applying a function GroupBy.get_group this function select a single group. 
grp = df.groupby('Name')
grp.get_group('Jai')

#   Now we select an object grouped on multiple columns.
grp = df.groupby(['Name', 'Qualification'])
grp.get_group(('Jai', 'Msc'))

"""  Step 2: Applying Functions to Groups
After splitting a data into a group we apply a function to each group, in order to do that we have to perform some operations which are as follows: """

#   1. Aggregation: It allows us to find a summary statistic for each group like summing or averaging values.
grp1 = df.groupby('Name')
grp1['Age'].aggregate(np.sum)

#   Now we perform aggregation on a group containing multiple keys. 
grp1 = df.groupby(['Name', 'Qualification'])
grp1['Age'].aggregate(np.sum)

#   When we can apply a multiple functions at once by passing a list or dictionary of functions.
grp = df.groupby('Name')
grp['Age'].agg([np.sum, np.mean, np.std])

#   2. Transformation: It is a process in which we perform some group-specific computations and return a result with the same index as the original data.
#   Here we are using some different datasets which is created randomly below.
import pandas as pd
data2 = {'Name':['Jai', 'Anuj', 'Jai', 'Princi', 
                 'Gaurav', 'Anuj', 'Princi', 'Abhi'], 
        'Age':[27, 24, 22, 32, 
               33, 36, 27, 32], 
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj',
                   'Jaunpur', 'Kanpur', 'Allahabad', 'Aligarh'], 
        'Qualification':['Msc', 'MA', 'MCA', 'Phd',
                         'B.Tech', 'B.com', 'Msc', 'MA'],
        'Score':[23, 34, 35, 45, 47, 50, 52, 53]} 
grp2 = df2.groupby('Name')
sc = lambda x: (x - x.mean()) / x.std() 
grp2['Age'].transform(sc)


#   3. Filtration: It is a process which is used to discard groups based on a condition. For example we can filter out groups where the number of records is less than a certain threshold.
grp2 = df2.groupby('Name')
grp2.filter(lambda x: len(x) >= 2)

""" Step 3: Combining Results
After applying the functions, results are combined back into a data structure like a DataFrame or Series. We can further analyze or visualize the data in its new form. """
df.groupby('Name').agg({'Age': 'sum'})






















