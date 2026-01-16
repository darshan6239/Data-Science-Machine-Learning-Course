#   Creating a Pandas Series
#   A Pandas Series can be created from different data structures such as lists, NumPy arrays, dictionaries or scalar value.

import pandas as pd

data = [1, 2, 3, 4]
 
ser = pd.Series(data)
print(ser)

#   Accessing elements of Series

""" 1. Position-based Indexing """
#   In order to access the series element refers to the index number. Use the index operator []to access an element in a series. The index must be an integer. In order to access multiple elements from a series we use Slice operation.

import pandas as pd
import numpy as np

data = np.array(['g','e','e','k','s','f', 'o','r','g','e','e','k','s'])
ser = pd.Series(data)

print(ser[:5])

""" 2. Label-based Indexing """
#   In order to access an element from series, we have to set values by index label. A Series is like a fixed-size dictionary in that we can get and set values by index label. let's see a example to understand this

import pandas as pd
import numpy as np

data = np.array(['g','e','e','k','s','f', 'o','r','g','e','e','k','s'])
ser = pd.Series(data,index=[10,11,12,13,14,15,16,17,18,19,20,21,22])

print(ser[16])

#   Indexing and Selecting Data in Series
""" 1. Indexing a Series using .loc[] """
#   This function selects data by refering the explicit index . The df.loc indexer selects data in a different way than just the indexing operator. It can select subsets of data. You can download dataset from here.

import pandas as pd  
 
df = pd.read_csv("nba.csv")  
   
ser = pd.Series(df['Name']) 
data = ser.head(10)
data

#   Now we access the element of series using .loc[] function.
data.loc[3:6]

#-----------------------------------------------------------------------------------------------------------------------------------------

""" 2. Indexing a Series using .iloc[] """
#   .iloc[] function allows us to retrieve data by position. In order to do that we’ll need to specify the positions of the data that we want. The df.iloc indexer is very similar to df.loc but only uses integer locations to make its selections.

import pandas as pd  
  
df = pd.read_csv("nba.csv")  
   
ser = pd.Series(df['Name']) 
data = ser.head(10)
data

#   Now we access the element of Series using .iloc[] function.
data.iloc[3:6]

#-----------------------------------------------------------------------------------------------------------------------------------------

""" Binary Operations on Pandas Series """
#   Pandas allows performing binary operations on Series, such as addition, subtraction, multiplication and division. These operations can be performed using functions like .add() , .sub(), .mul() and .div().

import pandas as pd

ser1 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
ser2 = pd.Series([4, 5, 6], index=['A', 'B', 'C'])

df_sum = ser1.add(ser2)
print(df_sum)

""" Common Binary Operations """
#   sub()	Method is used to subtract series or list like objects with same length from the caller series 
#   mul()	Method is used to multiply series or list like objects with same length with the caller series
#   div()	Method is used to divide series or list like objects with same length by the caller series
#   sum()	Returns the sum of the values for the requested axis
#   prod()	Returns the product of the values for the requested axis
#   mean()	Returns the mean of the values for the requested axis
#   pow()	Method is used to put each element of passed series as exponential power of caller series and returned the results
#   abs()	Method is used to get the absolute numeric value of each element in Series/DataFrame
#   cov()	Method is used to find covariance of two series 

""" Conversion Operation on Series """
#   Conversion operations allow transforming data types within a Series. This can be useful for ensuring consistency in data types. In order to perform conversion operation we have various function which help in conversion like .astype(), .tolist() etc
import pandas as pd

ser = pd.Series([1, 2, 3, 4])
ser = ser.astype(float)
print(ser)


"""     DOWNLOAD THE NBA FILE OUTSIDE   """


