""" Time series data is information collected in sequence over time. It shows how things change at different points, like stock prices every day or temperature every hour.

   It is used in industries such as finance, pharmaceuticals, social media, and research.
   Analyzing and visualizing this data helps us find trends, seasonal patterns, and behaviors.
   These insights support forecasting and guide better decision-making.
   The main goal is to study data in time order to extract meaningful patterns and predictions.
   
Concepts in Time Series Analysis
1)Trend: It represents the general direction in which a time series is moving over an extended period. It checks whether the values are increasing, decreasing or staying relatively constant.
2)Seasonality: Seasonality refers to repetitive patterns or cycles that occur at regular intervals within a time series corresponding to specific time units like days, weeks, months or seasons.
3)Moving average: It is used to smooth out short-term fluctuations and highlight longer-term trends or patterns in the data.
4)Noise: It represents the irregular and unpredictable components in a time series that do not follow a pattern.
5)Differencing: It is used to make the difference in values of a specified interval. By default it’s 1 but we can specify different values for plots.
6)Stationarity: A stationary time series is statistical properties such as mean, variance and autocorrelation remain constant over time.
7)Order: The order of differencing refers to the number of times the time series data needs to be differenced to achieve stationarity.
8)Autocorrelation: Autocorrelation is a statistical method used in time series analysis to quantify the degree of similarity between a time series and a lagged version of itself.
9)Resampling: Resampling is a technique in time series analysis that is used for changing the frequency of the data observations.

Types of Time Series Data
   Time series data can be classified into two sections:

1) Continuous Time Series: Data recorded at regular intervals with a continuous range of values like temperature, stock prices, Sensor Data, etc.
2) Discrete Time Series: Data with distinct values or categories recorded at specific time points like counts of events, categorical statuses, etc.
Visualization Approaches
     Use line plots or area charts for continuous data to highlight trends and fluctuations. 
     Use bar charts or histograms for discrete data to show frequency or distribution across categories. """

# Practical Time Series Visualization with Python
# Step 1: Installing and Importing Libraries
# We will be using Numpy, Pandas, seaborn and Matplotlib libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

# Step 2: Loading the Dataset
# Here we will load the dataset and use the parse_dates parameter to convert the Date column to the DatetimeIndex format.
df = pd.read_csv("/content/stock_data.csv", 
                 parse_dates=True, 
                 index_col="Date")
df.head()

# Step 3: Cleaning of Data
# We will drop columns from the dataset that are not important for our visualization.
df.drop(columns='Unnamed: 0', inplace =True)
df.head()

# Step 4: Plotting High Stock Prices
# Since the volume column is of continuous data type we will use line graph to visualize it.

# sns.lineplot(data=df, x=df.index, y='High', label='High Price', color='blue'): Plots High prices over time using the datetime index on x-axis.
sns.set(style="whitegrid") 

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Date', y='High', label='High Price', color='blue')
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Share Highest Price Over Time')
plt.show()

# Step 5: Resampling Data
# To better understand the trend of the data we will use the resampling method which provide a clearer view of trends and patterns when we are dealing with daily data.

# df_resampled = df.resample('M').mean(numeric_only=True): Resamples data to monthly frequency and calculates the mean of all numeric columns for each month.

df_resampled = df.resample('ME').mean(numeric_only=True) 

sns.set(style="whitegrid") 

plt.figure(figsize=(12, 6))  
sns.lineplot(data=df_resampled, x=df_resampled.index, y='High', label='Month Wise Average High Price', color='blue')
plt.xlabel('Date (Monthly)')
plt.ylabel('High')
plt.title('Monthly Resampling Highest Price Over Time')
plt.show()

# Step 6: Detecting Seasonality with Autocorrelation
# We will detect Seasonality using the autocorrelation function (ACF) plot. Peaks at regular intervals in the ACF plot suggest the presence of seasonality.
if 'Date' not in df.columns:
    print("'Date' is already the index or not present in the DataFrame.")
else:
    df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plot_acf(df['High'], lags=40)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function (ACF) Plot')
plt.show()

# Step 7: Testing Stationarity with ADF test
# We will perform the ADF test to formally test for stationarity.
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['High'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Step 8: Differencing to Achieve Stationarity
# Differencing involves subtracting the previous observation from the current observation to remove trends or seasonality.

df['high_diff'] = df['High'].diff()

plt.figure(figsize=(12, 6))
plt.plot(df['High'], label='Original High', color='blue')
plt.plot(df['high_diff'], label='Differenced High', linestyle='--', color='green')
plt.legend()
plt.title('Original vs Differenced High')
plt.show()

# Step 9: Smoothing Data with Moving Average
# df['High'].diff(): helps in calculating the difference between consecutive values in the High column. This differencing operation is used to transform a time series into a new series that represents the changes between consecutive observations.
window_size = 120
df['high_smoothed'] = df['High'].rolling(window=window_size).mean()

plt.figure(figsize=(12, 6))

plt.plot(df['High'], label='Original High', color='blue')
plt.plot(df['high_smoothed'], label=f'Moving Average (Window={window_size})', linestyle='--', color='orange')

plt.xlabel('Date')
plt.ylabel('High')
plt.title('Original vs Moving Average')
plt.legend()
plt.show()

# Step 10: Original Data Vs Differenced Data
# Printing the original and differenced data side by side we get:
df_combined = pd.concat([df['High'], df['high_diff']], axis=1)
print(df_combined.head())

# Hence the high_diff column represents the differences between consecutive high values. The first value of high_diff is NaN because there is no previous value to calculate the difference.

# As there is a NaN value we will drop that proceed with our test:
df.dropna(subset=['high_diff'], inplace=True)
df['high_diff'].head()

# After that if we conduct the ADF test:
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['high_diff'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])





