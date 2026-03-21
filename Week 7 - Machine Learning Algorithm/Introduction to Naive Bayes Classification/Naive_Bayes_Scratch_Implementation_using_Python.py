#  1. Importing Libraries
"""  Importing necessary libraries:

    math: for mathematical operations
    random: for random number generation
    pandas: for data manipulation
    numpy: for scientific computing   """
import math
import random
import pandas as pd
import numpy as np

#  2. Encoding Class
#  The encode_class function converts class labels in the dataset into numeric values. It assigns a unique numeric identifier to each class.
def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata

#  3. Splitting the Data
#  The splitting function is used to split the dataset into training and testing sets based on the given ratio.
def splitting(mydata, ratio):
    train_num = int(len(mydata) * ratio)
    train = []
    test = list(mydata)
    
    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test

#  4. Grouping Data by Class
#  The groupUnderClass function takes the data and returns a dictionary where each key is a class label and the value is a list of data points belonging to that class.
def groupUnderClass(mydata):
    data_dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
        data_dict[mydata[i][-1]].append(mydata[i])
    return data_dict


#    5. Calculating Mean and Standard Deviation for Class
#    The MeanAndStdDev function takes a list of numbers and calculates the mean and standard deviation.
#    The MeanAndStdDevForClass function takes the data and returns a dictionary where each key is a class label and the value is a list of lists, where each inner list contains the mean and standard deviation for each attribute of the class.
def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info


#  6. Calculating Gaussian and Class Probabilities
"""  The calculateGaussianProbability function takes a value, mean and standard deviation and calculates the probability of the value occurring under a Gaussian distribution with that mean and standard deviation.
     The calculateClassProbabilities function takes the information dictionary and a test data point as arguments. It iterates through each class and calculates the probability of the test data point belonging to that class based on the mean and standard deviation of each attribute for that class.  """
def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities


#    7. Predicting for Test Set
"""  The predict function takes the information dictionary and a test data point as arguments. It calculates the class probabilities and returns the class with the highest probability.
The getPredictions function takes the information dictionary and the test set as arguments. It iterates through each test data point and predicts its class using the predict function.  """
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel

def getPredictions(info, test):
    predictions = [predict(info, instance) for instance in test]
    return predictions

#    8. Calculating Accuracy
"""  The accuracy_rate function takes the test set and the predictions as arguments. It compares the predicted classes with the actual classes and calculates the percentage of correctly predicted data points.  """
filename = '/content/diabetes_data.csv' 
df = pd.read_csv(filename, header=None, comment='#')
mydata = df.values.tolist()

mydata = encode_class(mydata)
for i in range(len(mydata)):
    for j in range(len(mydata[i]) - 1):
        mydata[i][j] = float(mydata[i][j])

#    10. Splitting Data into Training and Testing Sets
"""  The code splits the data into training and testing sets using a specified ratio. It then trains the model by calculating the mean and standard deviation for each attribute in each class.  """

















