""" Seaborn is a popular Python library for creating attractive statistical visualizations. Built on Matplotlib and integrated with Pandas, it simplifies complex plots like line charts, heatmaps and violin plots with minimal code. """

#   Creating Plots with Seaborn
#   Seaborn makes it easy to create clear and informative statistical plots with just a few lines of code. It offers built-in themes, color palettes, and functions tailored for different types of data.

""" 1. LINE PLOT """
""" Syntax: sns.lineplot(x=None, y=None, data=None) """
#   Example
import pandas as pd
import matplotlib.pyplot as plt
data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

plt.plot(df.index, df['Age'])
plt.xlabel('Index')
plt.ylabel('Age')
plt.title('Age Line Plot')
plt.show()

""" 2. SCATTER PLOT """
""" Syntax: sns.scatterplot(x=None, y=None, data=None) """
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.scatterplot(x=df.index, y='Age', data=df)
plt.show()


""" 3. BOX PLOT """
""" Syntax: sns.boxplot(x=None, y=None, hue=None, data=None) """
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 45]} 
df = pd.DataFrame(data)

sns.boxplot(y='Age', data=df)
plt.show()


""" 4. VIOLIN PLOT """
""" Syntax: sns.violinplot(x=None, y=None, hue=None, data=None)"""
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.violinplot(y='Age', data=df)
plt.show()


""" 5. SWARM PLOT """
""" Syntax:  sns.swarmplot(x=None, y=None, hue=None, data=None) """
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.swarmplot(x=df.index, y='Age', data=df)
plt.show()


""" 6. BAR PLOT """
""" Syntax: sns.barplot(x=None, y=None, hue=None, data=None) """
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'], 'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.barplot(x='Name', y='Age', data=df)
plt.show()

""" 7. POINT PLOT """
""" Sytnax: sns.pointplot(x=None, y=None, hue=None, data=None)"""
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'JAYAN', 'ANURAG'],'Age': [21, 23, 20, 24]}
df = pd.DataFrame(data)

sns.pointplot(x='Name', y='Age', data=df)
plt.show()

""" 8. COUNT PLOT """
""" Syntax: sns.countplot(x=None, y=None, hue=None, data=None) """
#   Example 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {'Name': ['ANSH', 'SAHIL', 'ANSH', 'JAYAN', 'ANURAG', 'ANURAG', 'ANURAG', 'SAHIL']}
df = pd.DataFrame(data)

sns.countplot(x='Name', data=df)
plt.title("Frequency of Names")
plt.show()

""" 9. KDE PLOT """
""" Syntax: sns.kdeplot(x=None, *, y=None, vertical=False, palette=None, data=None, **kwargs)"""
#   Example 
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].map({ 0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

sns.kdeplot(data=df[df['Species'] == 'Virginica'], x='sepal length (cm)', fill=True, label='Virginica')
plt.legend()
plt.show()


#   How to Customize Seaborn Plots with Python?
""" Customizing Seaborn plots increases their readability and visual appeal which makes the data insights clearer and more informative. Here are several ways we can customize our plots in Seaborn:

1. Adding Titles and Axis Labels
Adding descriptive titles and axis labels makes our plots more understandable and informative. Using Matplotlib's plt.title(), plt.xlabel() and plt.ylabel() to set titles and axis labels. """
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
sns.scatterplot(x='sepal_length', y='sepal_width', data=iris)

# Add plot title and axis labels
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

""" 2. Built-in Styles and Grids in Seaborn
Seaborn provides built-in styles that control the background and grid of your plots. These styles improve readability and can be chosen based on your presentation needs.

Available Styles:

darkgrid – Dark background with light gridlines. Great for clear contrast.
whitegrid – White background with light gridlines. Ideal for statistical plots.
dark – Dark background without gridlines. Clean and modern look.
white – Plain white background without gridlines. Good for simple visuals.
ticks – White background with axis ticks styled sharply. Suitable for publications. """
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

sns.boxplot(x='species', y='petal_length', data=sns.load_dataset('iris'))
plt.title('Petal Length Distribution by Species')
plt.show()

""" 3. Customizing Color Palettes
Seaborn makes it easy to enhance the appearance of plots using color palettes. You can choose from built-in palettes like "deep", "muted", or "bright" or define your own using sns.color_palette(). Customizing colors improves clarity and helps match your data’s theme or purpose.

a) Using a Built-in Palette: """
sns.set_palette("pastel") 

sns.violinplot(x='species', y='petal_length', data=sns.load_dataset('iris'))
plt.title('Petal Length Distribution by Species')
plt.show()

""" b) Using a Custom Palette: """
custom_colors = ['#FF5733', '#33FFBD', '#335BFF']
sns.set_palette(custom_colors)

sns.violinplot(x='species', y='petal_length', data=sns.load_dataset('iris'))
plt.title('Custom Colored Petal Length Distribution')
plt.show()

""" 4. Adjusting Figure Size and Aspect Ratio
We can adjust the figure size using plt.figure(figsize=(width,height)) to control the plot's dimensions. This allows for better customization to fit different presentation or reports. """
plt.figure(figsize=(10, 6))

sns.lineplot(x='year', y='passengers', data=sns.load_dataset('flights'))
plt.title('Number of Passengers Over Time')
plt.show()


""" 5. Adding Markers to Line Plots
Markers can be added to Seaborn line plots using the marker argument to highlight data points. For example adding circular markers to the line plot using sns.lineplot(x='x', y='y' ,marker='o') """
sns.lineplot(x='year', y='passengers', data=sns.load_dataset('flights'), marker='o')
plt.title('Number of Passengers Over Time')
plt.show()

""" Visualizing Relationships and Patterns with Seaborn
We’ll see various plots in Seaborn for visualizing relationships, distributions and trends across our dataset. These visualizations help to find hidden patterns and correlations in datasets with multiple variables.

1. Pair Plots
Pair plots are used explore relationships between several variables by generating scatter plots for every pair of variables in a dataset along with univariate distributions on the diagonal. This is useful for exploring datasets with multiple variables and seeing potential correlations.

Syntax: sns.pairplot(data, hue=None) """
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
custom_palette = sns.color_palette("husl", 8)
sns.set_palette(custom_palette)

data = sns.load_dataset("iris")
sns.pairplot(data, hue="species")
plt.show()

""" 2. Joint Plots
Joint plots combine a scatter plot with the distributions of the individual variables. This allows for a quick visual representation of how the variables are distributed individually and how they relate to one another.

Syntax: sns.jointplot(x, y, data, kind='scatter')"""
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("tips")
sns.jointplot(x="total_bill", y="tip", data=data, kind="scatter", color="#008B8B")
plt.show()

""" 3. Grid Plot
Grid plots in Seaborn are used to create multiple subplots in a grid layout. Using Seaborn's FacetGrid we can visualize how variables interact across different categories which makesit easier to compare groups or conditions within our dataset.

Syntax: g = sns.FacetGrid(data, col='column_name', row='row_name')
g.map(sns.scatterplot, 'x', 'y') """
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

plot=sns.FacetGrid(tips, col="time", row="sex")
plot.map(sns.scatterplot, "total_bill", "tip")
plt.show()

""" Regression Plots: Visualizing Linear Relationships
Seaborn simplifies the process of performing and visualizing regressions specifically linear regressions which is important for identifying relationships between variables, detecting trends and making predictions. It supports two primary functions for regression visualization:

regplot(): This function plots a scatter plot along with a linear regression model fit.
lmplot(): This function also plots linear models but provides more flexibility in handling multiple facets and datasets.
Example: Let’s use a simple dataset to visualize a linear regression between two variables: x (independent variable) and y (dependent variable). """
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

sns.regplot(x='total_bill', y='tip', data=tips, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.show()
