"""    Python Implementation of Gaussian Naive Bayes
Here we will be applying Gaussian Naive Bayes to the Iris Dataset, this dataset consists of four features namely Sepal Length in cm, Sepal Width in cm, Petal Length in cm, Petal Width in cm and from these features we have to identify which feature set belongs to which specie class. The iris flower dataset is available in Sklearn library of python.

Now we will be using Gaussian Naive Bayes in predicting the correct specie of Iris flower.

1. Importing Libraries
First we will be importing the required libraries:

pandas: for data manipulation
load_iris: to load dataset
train_test_split: to split the data into training and testing sets
GaussianNB: for the Gaussian Naive Bayes classifier
accuracy_score: to evaluate the model
LabelEncoder: to encode the categorical target variable.      """
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


#  2. Loading the Dataset and Preparing Features and Target Variable
#      After that we will load the Iris dataset from a CSV file named "Iris.csv" into a pandas DataFrame. Then we will separate the features (X) and the target variable (y) from the dataset. Features are obtained by dropping the "Species" column and the target variable is set to the "Species" column which we will be predicting.
iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['Species'] = iris.target

X = data.drop("Species", axis=1)
y = data['Species']


#  3. Encoding and Splitting the Dataset
#      Since the target variable "Species" is categorical we will be using Label Encoder to convert it into numerical form. This is necessary for the Gaussian Naive Bayes classifier as it requires numerical inputs.
#      We will be splitting the dataset into training and testing sets using the train_test_split function. 70% of the data is used for training and 30% is used for testing. The random_state parameter ensures reproducibility of the same data.
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#    4. Creating and Training the Model
#        We will be creating a Gaussian Naive Bayes Classifier (gnb) and then training it on the training data using the fit method.
gnb = GaussianNB()

gnb.fit(X_train, y_train)

#    5. Plotting 1D Gaussian Distributions for All Features
#        We visualize the Gaussian distributions for each feature in the Iris dataset across all classes. The distributions are modeled by the Gaussian Naive Bayes classifier where each class is represented by a normal (Gaussian) distribution with a mean and variance specific to each feature. Separate plots are created for each feature in the dataset showing how each class's feature values are distributed.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

feature_names = iris.feature_names
num_features = len(feature_names)
num_classes = len(np.unique(y))

X_np = X.to_numpy() 

for feature_index in range(num_features):
    feature_name = feature_names[feature_index]
    x_vals = np.linspace(X_np[:, feature_index].min(), X_np[:, feature_index].max(), 200)

    plt.figure(figsize=(8, 4))

    for cls in range(num_classes):
        mean = gnb.theta_[cls, feature_index]
        std = np.sqrt(gnb.var_[cls, feature_index])
        y_vals = norm.pdf(x_vals, mean, std)
        plt.plot(x_vals, y_vals, label=f"Class {cls} ({iris.target_names[cls]})")

    plt.title(f"Gaussian Distribution - {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)

    plt.show()


#    6. Making Predictions
#       At last we will be using the trained model to make predictions on the testing data.
y_pred = gnb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"The Accuracy of Prediction on Iris Flower is: {accuracy}")










