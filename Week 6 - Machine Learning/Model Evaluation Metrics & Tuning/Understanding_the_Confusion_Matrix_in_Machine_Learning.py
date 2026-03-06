""" Confusion matrix is a simple table used to measure how well a classification model is performing. It compares the predictions made by the model with the actual results and shows where the model was right or wrong. This helps you understand where the model is making mistakes so you can improve it. It breaks down the predictions into four categories:

True Positive (TP): The model correctly predicted a positive outcome i.e the actual outcome was positive.
True Negative (TN): The model correctly predicted a negative outcome i.e the actual outcome was negative.
False Positive (FP): The model incorrectly predicted a positive outcome i.e the actual outcome was negative. It is also known as a Type I error.
False Negative (FN): The model incorrectly predicted a negative outcome i.e the actual outcome was positive. It is also known as a Type II error. """


#    Implementation of Confusion Matrix for Binary classification using Python
#    Step 1: Import the necessary libraries
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#    Step 2: Create the NumPy array for actual and predicted labels
#    1) actual: represents the true labels or the actual classification of the items. In this case its a list of 10 items where each entry is either 'Dog' or 'Not Dog'.
#    2) predicted: represents the predicted labels or the classification made by the model.
actual    = np.array(
  ['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted = np.array(
  ['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])

#    Step 3: Compute the confusion matrix
#    confusion_matrix: This function from sklearn.metrics computes the confusion matrix which is a table used to evaluate the performance of a classification algorithm. It compares actual and predicted to generate a matrix.
cm = confusion_matrix(actual,predicted)

#    Step 4: Plot the confusion matrix with the help of the seaborn heatmap
#    1) sns.heatmap: This function from Seaborn is used to create a heatmap of the confusion matrix.
#    2) annot=True: Display the numerical values in each cell of the heatmap.
sns.heatmap(cm, 
            annot=True,
            fmt='g', 
            xticklabels=['Dog','Not Dog'],
            yticklabels=['Dog','Not Dog'])
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top') 
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()


