""" F1 Score is a performance metric used in machine learning to evaluate how well a classification model performs on a dataset especially when the classes are imbalanced meaning one class appears much more frequently than another. It is the harmonic mean of precision and recall which combine both metrics into a single value that balances their importance. """


""" Implementing F1 Score in Python
We can easily calculate the F1 score in Python using the f1_score function from the sklearn.metrics module. This function supports both binary and multi-class classification.

Here's an explanation of the function and its parameters:
    1) f1_score function takes two required parameters: y_true and y_pred along with an optional parameter average.
    2) y_true: This parameter represents the true labels for the instances, providing the actual outcomes that the model is trying to predict.
    3) y_pred: This parameter contains the predicted labels from the model indicating the model's output based on the input data.
    4) average: This parameter defines the type of averaging performed on the data. It is a optional parameter.""" 

from sklearn.metrics import f1_score

y_true = [0, 1, 2, 2, 2, 2, 1, 0, 2, 1, 0]
y_pred = [0, 0, 2, 2, 1, 2, 1, 0, 1, 2, 1]

f1_per_class = f1_score(y_true, y_pred, average=None)
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("F1 score per class:", f1_per_class)
print("Micro-average F1 score:", f1_micro)
print("Macro-average F1 score:", f1_macro)
print("Weighted-average F1 score:", f1_weighted)
