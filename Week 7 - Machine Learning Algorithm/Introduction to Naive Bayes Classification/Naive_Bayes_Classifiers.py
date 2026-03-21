""" Naive Bayes is a machine learning classification algorithm that predicts the category of a data point using probability. It assumes that all features are independent of each other. Naive Bayes performs well in many real-world applications such as spam filtering, document categorisation and sentiment analysis.  



Assumption of Naive Bayes
The fundamental Naive Bayes assumption is that each feature makes an:

Feature independence: This means that when we are trying to classify something, we assume that each feature (or piece of information) in the data does not affect any other feature.
Continuous features are normally distributed: If a feature is continuous, then it is assumed to be normally distributed within each class.
Discrete features have multinomial distributions: If a feature is discrete, then it is assumed to have a multinomial distribution within each class.
Features are equally important: All features are assumed to contribute equally to the prediction of the class label.
No missing data: The data should not contain any missing values.


Advantages
    Easy to implement and computationally efficient.  
    Effective in cases with a large number of features.
    Performs well even with limited training data.
    It performs well in the presence of categorical features.
    For numerical features data is assumed to come from normal distributions
Disadvantages
    Assumes that features are independent, which may not always hold in real-world data.
    Can be influenced by irrelevant attributes.
    May assign zero probability to unseen events, leading to poor generalization.
Applications
    Spam Email Filtering: Classifies emails as spam or non-spam based on features.
    Text Classification: Used in sentiment analysis, document categorization and topic classification.  
    Medical Diagnosis: Helps in predicting the likelihood of a disease based on symptoms.
    Credit Scoring: Evaluates creditworthiness of individuals for loan approval.
    Weather Prediction: Classifies weather conditions based on various factors. """
