""" A Word Cloud is a picture made up of words where the size of each word shows how frequently it appears in the dataset i.e the bigger the word appears in the cloud the more often that word is used. They help us to identify the most common and important words in a text. In this article, we will understand about word cloud and how to generate it using Python.

For Example: If we analyze customer reviews of a movie like "good", "bad" or "average" might be bigger if they are mentioned many times. These are useful because they:

  Quickly show the most common words in text data
  Help to understand what people are talking about in large text files
  Make text data look visually appealing
  Allow easy identification of important words
Implementing Word Cloud in Python
We will be using IMDB dataset and this dataset contains 50,000 movie reviews in CSV format. We will focus on the review column which contains the text data of the movie reviews. Below is the step by step implementation: """ 

#    Step 1: Loading the Dataset
#    Let's load the dataset using pandas. 

#    USE THE DATA SET NAME IMDB-Dataset.csv
import pandas as pd
df = pd.read_csv('/content/IMDB-Dataset.csv')
print(df.head())

#    Step 2: Understanding the Dataset
#    Before cleaning the text let's understand the dataset. The dataset contains two columns:

"""  review: Contains the movie review text
    sentiment: It shows whether the review is positive or negative
    We are only interested in the review column. Let's check the column names and some sample text. """
print(df.columns)
print(df['review'][0])

The review column contains detailed text reviews of movies. Our goal is to extract the most frequent words from these reviews.

#    Step 3: Cleaning the Text Data
#    Before generating the word cloud, we need to clean the text data which involves:

"""  1. Removing punctuation
     2. Converting text to lowercase
     3. Removing stopwords i.e common words like "the", "is", "and"
     
     re.sub(): This removes punctuation and numbers
     STOPWORDS: These are list of common stopwords """
import re
from wordcloud import STOPWORDS
text = ' '.join(df['review'].astype(str).tolist())
text = re.sub(r'[^A-Za-z\s]', '', text)
text = text.lower()
stopwords = set(STOPWORDS)
text = ' '.join(word for word in text.split() if word not in stopwords)


#    Step 4: Generating the Word Cloud
#    Now our text is clean, let's generate the word cloud.

#    WordCloud(): Generates the word cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.title("IMDB Movie Reviews Word Cloud")
plt.show()

#    Step 5: Customizing the Word Cloud
"""  We can customize the word cloud with different options like:
  1. Maximum number of words
  2. Color scheme
  3. Shape of the cloud

max_words: Limits the number of words
colormap: Changes the color of the word cloud """
wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='coolwarm').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Customized IMDB Movie Reviews Word Cloud")
plt.show()
