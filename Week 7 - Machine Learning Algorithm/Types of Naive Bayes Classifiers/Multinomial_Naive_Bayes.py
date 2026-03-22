"""    Python Implementation of Multinomial Naive Bayes
Let's understand it with a example of spam email detection. We'll classify emails into two categories: spam and not spam.

  1. Importing Libraries:
      We will import pandas and scikit learn where:

    pandas: Used for handling data in DataFrame format.
    CountVectorizer: Converts a collection of text documents into a matrix of token counts.
    train_test_split: Splits the data into training and test sets for model evaluation.
    MultinomialNB: A Naive Bayes classifier suited for classification tasks with discrete features                     (such as word counts).
    accuracy_score: Computes the accuracy of the model's predictions.    """

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#    2. Creating the Dataset
#        A simple dataset is created with text messages labeled as either spam or not spam. This data is then converted into a DataFrame for easy handling.
data = {
    'text': [
        'Free money now',
        'Call now to claim your prize',
        'Meet me at the park',
        'Let’s catch up later',
        'Win a new car today!',
        'Lunch plans?',
        'Congratulations! You won a lottery',
        'Can you send me the report?',
        'Exclusive offer for you',
        'Are you coming to the meeting?'
    ],
    'label': ['spam', 'spam', 'not spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}

df = pd.DataFrame(data)

#    3. Mapping Labels to Numerical Values
#        The labels (spam and not spam) are mapped to numerical values where spam becomes 1 and not spam becomes 0. This is necessary for the classifier, as it works with numerical data.
df['label'] = df['label'].map({'spam': 1, 'not spam': 0})

#    4. Splitting the Data
#        X contains the text messages (features), and y contains the labels (target).
#        The dataset is split into training (70%) and testing (30%) sets using train_test_split.
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#      5. Vectorizing the Text Data
#          CountVectorizer is used to convert text data into numerical vectors. It counts the               occurrences of each word in the corpus.
#          fit_transform() is applied to the training data to learn the vocabulary and transform             it into a feature matrix.
#          transform() is applied to the test data to convert it into the same feature space.
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

#    6. Training the Naive Bayes Model
#        A Multinomial Naive Bayes classifier is created and trained using the vectorized training data (X_train_vectors) and corresponding labels (y_train).
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

#    7. Making Predictions and Evaluating Accuracy
#          We are using model.predict(X_test_vectors) to generate predictions from the trained model on test data.
#          accuracy_score(y_test, y_pred) compares predicted labels y_pred with true labels y_test to calculate accuracy.
y_pred = model.predict(X_test_vectors)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

#    8. Predicting for a Custom Message
#        We create a custom message and transform it into a vector using vectorizer.transform().
#        The vectorized message is passed to model.predict() to get the prediction.
#        We print the result, interpreting 1 as “Spam” and 0 as “Not Spam”.
custom_message = ["Congratulations, you've won a free vacation"]
print(custom_message)
custom_vector = vectorizer.transform(custom_message)
prediction = model.predict(custom_vector)
print("Prediction for custom message:", "Spam" if prediction[0] == 1 else "Not Spam")






