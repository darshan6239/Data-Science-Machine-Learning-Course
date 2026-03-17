""" 1: Importing Necessary Libraries
We will import Numpy, Pandas, Matplotlib, Seaborn, Statsmodels and sklearn library in python."""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


""" 2: Data Preparation
The dataset is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD). The dataset provides the patients information. It includes over 4,000 records and 15 attributes. """ 


"""2.1 Loading and Handling Missing Values from the Dataset
We will load the dataset and drop the irrelevant features from the the dataset like "education" and rename columns also.

disease_df.pd.read_csv(): This is used to read the contents of CSV file.
disease_df, dropna(axis=0, inplace=True): This removes any rows with missing values (NaN) from the DataFrame.
disease_df.TenYearCHD.value_counts(): This prints the count of unique values in the TenYearCHD column which likely indicates whether a patient has heart disease. """
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(columns=['education'], inplace = True, axis = 1)
disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)
disease_df.dropna(axis = 0, inplace = True)
disease_df

#    Handling Missing Values
print(disease_df.TenYearCHD.value_counts())

""" 3: Splitting the Dataset into Test and Train Sets
We will split the dataset into training and testing portions. But before that we will transform our data by scaling all the features using StandardScaler.

  X=preprocessing.StandardScaler().fit(X).transform(X): This scales the features in X to have a mean of 0 and standard deviation of 1 using StandardScaler. Scaling is important for many machine learning models, especially when the features have different units or magnitudes.
  Training set (70% of data, X_train and y_train)
    Test set (30% of data, X_test and y_test)
      random_state=4 ensures the split is reproducible."""

X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

""" 4: Exploratory Data Analysis of Heart Disease Dataset
In Exploratory Data Analysis (EDA) we perform EDA on the heart disease dataset to understand and gain insights into the dataset before building a predictive model for heart disease.

4.1: Ten Year's CHD Record of all the patients available in the dataset:
  sns.countplot(x='TenYearCHD', data=disease_df, palettte="BuGn_r"): creates a count plot using Seaborn which visualizes the distribution of the values in the TenYearCHD column showing how many individuals have heart disease (1) vs. how many don’t (0)."""
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df,
             palette="BuGn_r")
plt.show()

""" 4.2: Counting number of patients affected by CHD where (0= Not Affected; 1= Affected)"""
laste = disease_df['TenYearCHD'].plot()
plt.show(laste)


""" 5: Fitting Logistic Regression Model for Heart Disease Prediction
We will create a simple logistic regression model for prediction.

  logreg=LogisticRegression(): This creates an instance of the LogisticRegression model.
  logreg.fit(X_train, y_train): This trains the logistic regression model using the training data (X_train for features and y_train for the target).
  y_pred=logreg.predict(X_test): This uses the trained logistic regression model to make predictions on the test set (X_test). The predicted values are stored in y_pred."""
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

""" 6: Evaluating Logistic Regression Model """
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', 
      accuracy_score(y_test, y_pred))

""" Plotting Confusion Matrix
Confusion Matrix is a performance evaluation tool used to assess the accuracy of a classification model. It is used to evaluate the performance of our logistic regression model in predicting heart disease helping us understand how well the model distinguishes between positive and negative cases.

  cm=confusion_matrix(y_test, y_pred): Compute confusion matrix by comparing the actual values (y_test) with the predicted values(y_pred). It returns a 2x2 matrix showing true positives, true negatives, false positives and false negatives.""" 
from sklearn.metrics import confusion_matrix, classification_report

print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])

plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")

plt.show()


