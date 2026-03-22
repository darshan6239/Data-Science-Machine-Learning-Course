"""    Implementing Bernoulli Naive Bayes
For performing classification using Bernoulli Naive Bayes we have considered an email dataset.

The email dataset comprises of four columns named Unnamed: 0, label, label_num and text. The category of label is either ham or spam. For ham the number assigned is 0 and for spam 1 is assigned. Text comprises the body of the mail. The length of the dataset is 5171.  """

#    1. Importing Libraries
#    In the code we have imported necessary libraries like pandas, numpy and sklearn. Bernoulli Naive Bayes is a part of sklearn package.
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

#    2. Data Analysis
#    In this code we have performed a quick data analysis that includes reading the data, dropping unnecessary columns, printing shape of data, information about dataset etc.
df=pd.read_csv("spam_ham_dataset.csv")
print(df.shape)
print(df.columns)
df= df.drop(['Unnamed: 0'], axis=1)

#    3. Count Vectorizer
#    In the code since text data is used to train our classifier we convert the text into a matrix comprising numbers using Count Vectorizer so that the model can perform well.
x = df["text"].values
y = df["label_num"].values

cv = CountVectorizer()

x = cv.fit_transform(x)

#    4. Data Splitting, Model Training and Prediction
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)\

bnb = BernoulliNB(binarize=0.0)
model = bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))







