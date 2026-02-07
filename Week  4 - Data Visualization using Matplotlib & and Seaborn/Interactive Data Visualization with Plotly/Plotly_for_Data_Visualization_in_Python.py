""" Plotly is an open-source Python library designed to create interactive, visually appealing charts and graphs. It helps users to explore data through features like zooming, additional details and clicking for deeper insights. It handles the interactivity with JavaScript behind the scenes so that we can focus on writing Python code to build the charts. In this article, we will see plotting in Plotly and covers how to create basic charts and enhance them with interactive features.

To get started with Plotly simply install it using the following command:
----> pip install plotly 

Understanding Plotly Modules
Plotly consists of two key modules:

1) plotly.graph_objects: This module is used to define and create plots. It contains objects such as Figure, layout and data which are responsible for plotting.
2) plotly.express: This is a high-level interface for creating a wide variety of plots with minimal code. It simplifies the process of creating complex visualizations and allows users to create figures with just one line of code.

Example """
import plotly.express as px

fig = px.line(x=[1, 2], y=[3, 4])

print(fig)

""" Basic Charts in Plotly : 
Here we will see how to generate basic charts using Plotly and apply various customizations to enhance their appearance and functionality. We will learn how to visualize different graph like line charts, scatter plots, bar charts, histograms and pie charts. We will cover the following customizations:

Adjusting chart layout
Adding annotations
Customizing markers and lines

---> 1. Line chart
Plotly line chart is one of the simple plots where a line is drawn to show relation between the X-axis and Y-axis. It can be created using the px.line() method with each data position is represented as a vertex of a polyline mark in 2D space.

Syntax: plotly.express.line(data_frame=None, x=None, y=None, color=None, title=None)

Parameters:
data_frame: Dataset to plot.
x: Column name for the X-axis.
y: Column name for the Y-axis.
color: Color the lines based on this column.
title: Title of the plot.
Return: A plotly.express.Figure object.

Example:
We will be using Iris dataset and it is directly available as part of scikit-learn. df = px.data.iris() from the plotly.express library loads it into a Pandas DataFrame. This dataset contains measurements of sepal length, sepal width, petal length and petal width for 150 iris flowers and categorized into three species: setosa, versicolor and virginica. """
import plotly.express as px

df = px.data.iris()

fig = px.line(df, y="sepal_width",)

fig.show()
""" In the above example, we can see that:

Plotly automatically assigns labels to the X and Y axes.
The data points for both axes are displayed.
We can zoom in, zoom out or select specific parts of the data.
It provides interactive tools in the top-right corner for chart manipulation.
We can also save the chart locally as a static image.
Now let's try to customize our graph a little. 

Example 1: In this example we will use the line dash parameter which is used to group the lines according to the dataframe column passed. """
import plotly.express as px

df = px.data.iris()

fig = px.line(df, y="sepal_width", line_group='species')

fig.show()

"""Example 2: In this example, we will group and color the data according to the species. We will also change the line format. For this we will use two attributes such line_dash and color."""

import plotly.express as px
df = px.data.iris()
fig = px.line(df, y="sepal_width", line_dash='species',
              color='species')
fig.show()

""" 2. Bar Chart
A bar chart is a pictorial representation of data that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. These data sets contain the numerical values of variables that represent the length or height. It can be created using the px.bar() method.

Syntax:  plotly.express.bar(data_frame=None, x=None, y=None, color=None, title=None)

Parameters:

data_frame: Dataset to plot.
x: The column name for the X-axis.
y: The column name for the Y-axis.
color: Color the bars based on this column.
title: Title of the plot.
Return: A plotly.express.Figure object.

Example:
We will be using tips dataset and df = px.data.tips() from the plotly.express library loads a sample dataset about tips into a Pandas DataFrame. This dataset contains 244 rows and 7 columns with each row representing a single restaurant bill and associated information. """

import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='day', y="total_bill")
fig.show()

""" Let's try to customize this plot.

Example: Customizations that we will use -

color: Used to color the bars.
facet_row: Divides the graph into rows according to the data passed
facet_col: Divides the graph into columns according to the data passed """
import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x='day', y="total_bill", color='sex',
             facet_row='time', facet_col='sex')
fig.show()

""" 3. Scatter Plot
A scatter plot is a set of dotted points to represent individual pieces of data in the horizontal and vertical axis. A graph in which the values of two variables are plotted along X-axis and Y-axis, the pattern of the resulting points reveals a correlation between them and it can be created using the px.scatter() method.
 
Syntax:  plotly.express.scatter(data_frame=None, x=None, y=None, color=None, title=None)

Parameters:
data_frame: Dataset to plot.
x: The column name for the X-axis.
y: The column name for the Y-axis.
color: Color the bars based on this column.
title: Title of the plot.
Return: A plotly.express.Figure object.

Example:"""
import plotly.express as px
df = px.data.tips()
fig = px.scatter(df, x='total_bill', y="tip")
fig.show()

""" Example: Let's see various customizations available for this chart that we will use - 

color: Color the points.
symbol: Gives a symbol to each point according to the data passed.
size: Size for each point."""

import plotly.express as px
df = px.data.tips()
fig = px.scatter(df, x='total_bill', y="tip", color='time',
                 symbol='sex', size='size', facet_row='day',
                 facet_col='time')
fig.show()

""" 4. Histogram
A histogram is used to represent data in the form of some groups. It is a type of bar plot where the X-axis represents the bin ranges while the Y-axis gives information about frequency. It can be created using the px.histogram() method.

Syntax:  plotly.express.histogram(data_frame=None, x=None, y=None, color=None, nbins=None, histnorm=None, title=None, width=None, height=None) 

Parameters:
data_frame: Dataset to plot.
x: The column name for the X-axis (values to be binned).
color: Color the bars based on this column.
nbins: Set the number of bins.
histnorm: Normalize the histogram (e.g"percent", "density").
Return: A plotly.express.Figure object.

Example: """
import plotly.express as px
df = px.data.tips()
fig = px.histogram(df, x="total_bill")
fig.show()

