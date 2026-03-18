""" Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog."""


#    Types of SVM are -: 
#      1) LINEAR SVM
#      2) NON-LINEAR SVM


""" Implementing SVM Algorithm Using Scikit-Learn
We will predict whether cancer is Benign or Malignant using historical data about patients diagnosed with cancer. This data includes independent attributes such as tumor size, texture, and others. To perform this classification, we will use an SVM (Support Vector Machine) classifier to differentiate between benign and malignant cases effectively.

    load_breast_cancer(): Loads the breast cancer dataset (features and target labels).
    SVC(kernel="linear", C=1): Creates a Support Vector Classifier with a linear kernel and regularization parameter C=1.
    svm.fit(X, y): Trains the SVM model on the feature matrix X and target labels y.
    DecisionBoundaryDisplay.from_estimator(): Visualizes the decision boundary of the trained model with a specified color map.
    plt.scatter(): Creates a scatter plot of the data points, colored by their labels.
    plt.show(): Displays the plot to the screen."""
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

cancer = load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target
svm = SVC(kernel="linear", C=1)
svm.fit(X, y)
DecisionBoundaryDisplay.from_estimator(
        svm,
        X,
        response_method="predict",
        alpha=0.8,
        cmap="Pastel1",
        xlabel=cancer.feature_names[0],
        ylabel=cancer.feature_names[1],
    )
plt.scatter(X[:, 0], X[:, 1], 
            c=y, 
            s=20, edgecolors="k")
plt.show()

