""" Visualizing Data with Pyplot using Matplotlib
   Pyplot is a module in Matplotlib that provides a simple interface for creating plots. It allows users to generate charts like line graphs, bar charts and histograms with minimal code. Let’s explore some examples     with simple code to understand how to use it effectively. """

#   1. Line Chart
#   Line chart is one of the basic plots and can be created using plot() function. It is used to represent a relationship between two data X and Y on a different axis.

""" Syntax: matplotlib.pyplot.plot(x, y)
    Parameter: x, y Coordinates for data points.
    Example: This code plots a simple line chart with labeled axes and a title using Matplotlib. """

import matplotlib.pyplot as plt

x = [10, 20, 30, 40]
y = [20, 25, 35, 55]

plt.plot(x, y)
plt.title("Line Chart")
plt.ylabel('Y-Axis')
plt.xlabel('X-Axis')
plt.show()

#   2.. Bar Chart
#   Bar chart displays categorical data using rectangular bars whose lengths are proportional to the values they represent. It can be plotted vertically or horizontally to compare different categories.

""""Syntax: matplotlib.pyplot.bar(x, height)
    Parameter:
    x: Categories or positions on x-axis.
    height: Heights of the bars (y-axis values).
    Example: This code creates a simple bar chart to show total bills for different days. X-axis represents the days and Y-axis shows total bill amount. """

import matplotlib.pyplot as plt

x = ['Thur', 'Fri', 'Sat', 'Sun']
y = [170, 120, 250, 190]

plt.bar(x, y)
plt.title("Bar Chart")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show()


#   3.Histogram
#   Histogram shows the distribution of data by grouping values into bins. The hist() function is used to create it, with X-axis showing bins and Y-axis showing frequencies.

"""Syntax: matplotlib.pyplot.hist(x, bins=None)
   Parameter:
   x: Input data.
   bins: Number of bins (intervals) to group data.
   Example: This code plots a histogram to show frequency distribution of total bill values from the list x. It uses 10 bins and adds axis labels and a title for clarity. """
import matplotlib.pyplot as plt

x = [7, 8, 9, 10, 10, 12, 12, 12, 13, 14, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20,
     21, 22, 23, 24, 25, 25, 26, 28, 30, 32, 35, 36, 38, 40, 42, 44, 48, 50]

plt.hist(x, bins=10, color='steelblue')
plt.title("Histogram")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")
plt.show()

#   4. Scatter Plot
#   Scatter plots are used to observe relationships between variables. The scatter() method in the matplotlib library is used to draw a scatter plot.

"""Syntax: matplotlib.pyplot.scatter(x, y)
   Parameter: x, y Coordinates of the points.

   Example: This code creates a scatter plot to visualize the relationship between days and total bill amounts using scatter()."""

import matplotlib.pyplot as plt

x = ['Thur', 'Fri', 'Sat', 'Sun', 'Thur', 'Fri', 'Sat', 'Sun']
y = [170, 120, 250, 190, 160, 130, 240, 200]

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("Day")
plt.ylabel("Total Bill")
plt.show()

#   5. Pie Chart
#   Pie chart is a circular chart used to show data as proportions or percentages. It is created using the pie(), where each slice (wedge) represents a part of the whole.

"""Syntax: matplotlib.pyplot.pie(x, labels=None, autopct=None)
   Parameter:
   x: Data values for pie slices.
   labels: Names for each slice.
   autopct: Format to display percentage (e.g., '%1.1f%%').
   Example: This code creates a simple pie chart to visualize distribution of different car brands. Each slice of pie represents the proportion of cars for each brand in the dataset."""
import matplotlib.pyplot as plt
import pandas as pd

cars = ['AUDI', 'BMW', 'FORD','TESLA', 'JAGUAR',]
data = [23, 10, 35, 15, 12]

plt.pie(data, labels=cars)
plt.title(" Pie Chart")
plt.show()

#   6. Box Plot
""" Box plot is a simple graph that shows how data is spread out. It displays the minimum, maximum, median and quartiles and also helps to spot outliers easily.

   Syntax: matplotlib.pyplot.boxplot(x, notch=False, vert=True)
   Parameter:
   x: Data for which box plot is to be drawn (usually a list or array).
   notch: If True, draws a notch to show the confidence interval around the median.
   vert: If True, boxes are vertical. If False, they are horizontal.
   Example: This code creates a box plot to show the data distribution and compare three groups using matplotlib"""

import matplotlib.pyplot as plt

data = [ [10, 12, 14, 15, 18, 20, 22],
         [8, 9, 11, 13, 17, 19, 21],
         [14, 16, 18, 20, 23, 25, 27] ]

plt.boxplot(data)
plt.xlabel("Groups")
plt.ylabel("Values")
plt.title("Box Plot")
plt.show()

#   7. Heatmap
"""Heatmap is a graphical representation of data where values are shown as colors. It helps visualize patterns, correlations or intensity in a matrix-like format. It is created using imshow() method in Matplotlib.

   Syntax: matplotlib.pyplot.imshow(X, cmap='viridis')
   Parameter:
   X: 2D array (data to display as an image or heatmap).
   cmap: Sets the color map.
   Example: This code creates a heatmap of random 10×10 data using imshow(). It uses 'viridis' color map and colorbar() adds a color scale."""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
data = np.random.rand(10, 10)

plt.imshow(data, cmap='viridis', interpolation='nearest')

plt.colorbar()
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Heatmap')
plt.show()

""" Explanation:

   np.random.seed(0): Ensures same random values every time (reproducibility).
   np.random.rand(10, 10): Generates a 10×10 array of random numbers between 0 and 1. """
