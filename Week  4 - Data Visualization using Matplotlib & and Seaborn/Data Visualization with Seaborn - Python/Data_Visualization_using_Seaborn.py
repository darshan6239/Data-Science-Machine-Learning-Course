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
