#  Example 1: Broadcasting a Scalar to a 1D Array
#  It creates a NumPy array arr with values [1, 2, 3] and adds a scalar value 1 to each element of the array using broadcasting.

import numpy as np
arr = np.array([1, 2, 3])
res = arr + 1  
print(res)

#  Example 2: Broadcasting a 1D Array to a 2D Array
#  This example shows how a 1D array a1 is added to a 2D array a2. NumPy automatically expands    the 1D array along the rows of the 2D array to perform element-wise addition.

import numpy as np

a = np.array([2, 4, 6])
b = np.array([[1, 3, 5], [7, 9, 11]])
res = a + b
print(res)


#   Example 3: Broadcasting in Conditional Operations
#   This example checks each age in the array and assigns "Adult" or "Minor" using np.where().

import numpy as np

a = np.array([12, 24, 35, 45, 60, 72])
b = np.array(["Adult", "Minor"])
res = np.where(a > 18, b[0], b[1])
print(res)

#   Example 4: Using Broadcasting for Matrix Multiplication
#   In this example, each element of a 2D matrix is multiplied by the corresponding element in a broadcasted vector.

import numpy as np
m = np.array([[1, 2], [3, 4]])
v = np.array([10, 20])
res = m * v
print(res)

#  Example 5: Scaling Data with Broadcasting
#  Consider a real-world scenario where we need to calculate the total calories in foods based on the amount of fats, proteins and carbohydrates. Each nutrient has a specific caloric value per gram.

#  Fats: 9 calories per gram (CPG)
#  Proteins: 4 CPG
#  Carbohydrates: 4 CPG

import numpy as np

fd = np.array([ [0.8, 2.9, 3.9],
                [52.4, 23.6, 36.5],
                [55.2, 31.7, 23.9],
                [14.4, 11.0, 4.9] ])

cpg = np.array([9, 4, 4])
res = fd * cpg
print(res)

#  Example 6: Adjusting Temperature Data Across Multiple Locations
#  Suppose you have a 2D array representing daily temperature readings across multiple cities and you want to apply a correction factor to each city’s temperature data.

import numpy as np

temp = np.array([ [30, 32, 34, 33, 31],
                  [25, 27, 29, 28, 26],
                  [20, 22, 24, 23, 21] ])

corr = np.array([1.5, -0.5, 2.0])
res = temp + corr[:, None]
print(res)

#  Example 7: Normalizing Image Data
#  Normalization is important in many real-world scenarios like image processing and machine learning because it:

#  Centers data by subtracting the mean by ensuring features have zero mean.
#  Scales data by dividing by the standard deviation by ensuring features have unit variance.
#  Improves numerical stability and performance of algorithms like gradient descent.
#  Let's see how broadcasting simplifies normalization:

import numpy as np

img = np.array([ [100, 120, 130],
                 [90, 110, 140],
                 [80, 100, 120] ])

m = img.mean(axis=0)
s = img.std(axis=0)
res = (img - m) / s
print(res)

#  Example 8: Centering Data in Machine Learning
#  Centering data is an important step in many machine learning workflows. Broadcasting helps center the data efficiently by subtracting the mean from each feature. This example centers each feature by subtracting its mean using NumPy broadcasting.

import numpy as np

data = np.array([ [10, 20],
                  [15, 25],
                  [20, 30] ])

m = data.mean(axis=0)
res = data - m
print(res)







