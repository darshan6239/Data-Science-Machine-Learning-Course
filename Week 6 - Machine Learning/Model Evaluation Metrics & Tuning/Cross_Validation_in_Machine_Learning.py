""" Cross-validation is a technique used to check how well a machine learning model performs on unseen data while preventing overfitting. It works by:

Splitting the dataset into several parts.
Training the model on some parts and testing it on the remaining part.
Repeating this resampling process multiple times by choosing different parts of the dataset.
Averaging the results from each validation step to get the final performance. """ 

#    Python implementation for k fold cross-validation
#    Step 1: Importing necessary libraries
#    We will import essential modules from scikit-learn.

""" 1) cross_val_score helps evaluate model performance using cross-validation.
    2) KFold splits the data into defined folds.
    3) SVC is used for Support Vector Classification.
    4) load_iris loads the sample dataset. """ 

from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC
from sklearn.datasets import load_iris

#    Step 2: Loading the dataset
#    We will use the Iris dataset a built-in, multi-class dataset with 150 samples and 3 flower species (Setosa, Versicolor and Virginica).
iris = load_iris()
X, y = iris.data, iris.target

#    Step 3: Creating SVM classifier
#    SVC() from scikit-learn is used to build the Support Vector Machine model. Here, we are using a linear kernel, suitable for linearly separable data.
svm_classifier = SVC(kernel='linear')

#    Step 4: Defining the number of folds for cross-validation
#    We define 5 folds, meaning the dataset will be split into 5 parts. The model will train on 4 parts and test on 1, repeating this process 5 times for balanced evaluation.
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

#    Step 5: Performing k-fold cross-validation
#    We use cross_val_score() to automatically split data, train and evaluate the model across all folds. It returns the accuracy for each fold
cross_val_results = cross_val_score(svm_classifier, X, y, cv=kf)


#    Step 6: Evaluation metrics
#    We print individual fold accuracies and the mean accuracy across all folds to understand the model’s stability and generalization.
print("Cross-Validation Results (Accuracy):")
for i, result in enumerate(cross_val_results, 1):
    print(f"  Fold {i}: {result * 100:.2f}%")
    
print(f'Mean Accuracy: {cross_val_results.mean()* 100:.2f}%')

