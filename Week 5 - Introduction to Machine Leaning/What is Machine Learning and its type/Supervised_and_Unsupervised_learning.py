Supervised and unsupervised learning are two main types of machine learning. In supervised learning, the model is trained with labeled data where each input has a corresponding output. On the other hand, unsupervised learning involves training the model with unlabeled data which helps to uncover patterns, structures or relationships within the data without predefined outputs. In this article we will see Supervised and unsupervised learning in more details.

_Learning-Without-Labels.webp_Learning-Without-Labels.webp
What is Supervised learning?
Supervised learning as the name suggests, works like a teacher or supervisor guiding the machine. In this approach we teach or train the machine using the labelled data(correct answers or classifications) which means each input has the correct output in the form of answer or category attached to it. After that machine is provided with a new set of examples (data) so that it can analyses the training data and produces a correct outcome from labeled data.

For example, a labeled dataset of images of Elephant, Camel and Cow would have each image tagged with either "Elephant", "Camel" or "Cow."

Supervised-learning

Example to Understand
Imagine we have a basket full of different fruits that we want the machine to identify. The machine first looks at the image of a fruit and extracts features like its shape, color and texture. Then it compares these features to the fruits it has already learned during training. If the new fruit’s features closely match those of an apple, the machine will predict that the fruit is an apple.

For example, suppose we train the machine by showing it fruits one by one:

If the fruit is round, has a small depression at the top and is red, it is labeled as an Apple.
If the fruit is long, curved and greenish-yellow, it is labeled as a Banana.
Now after this training, if we give the machine a new fruit (say a banana) from the basket and ask it to identify it, the machine will use what it has learned during training. It will analyze the shape and color of the new fruit and classify it as a Banana placing it in the correct category. In this way, the machine learns from the training data (the basket with labeled fruits) and applies that knowledge to recognize new, unseen fruits.

Types of Supervised Learning
Supervised learning is classified into two types of algorithms: 

1. Regression
A regression is used to predict continuous values such as house prices, stock prices or temperature. Regression algorithms learn how to connect input data to a specific number or value.

Some common regression algorithms include:

Linear Regression
Polynomial Regression
Lasso Regression
Ridge Regression
2. Classification
A classification is used to predict categorical values such as whether a customer will buy or not, whether an email is spam or not or whether a medical image shows a tumor or not. Classification algorithms learn how to connect input data to the probability of belonging to different groups or categories.

Some of the most common classification algorithms include:

Logistic Regression
Support Vector Machines
Decision Trees
Random Forests
Naive Baye
Applications of Supervised learning
It can be used to solve variety of problems which includes:

Image classification: It can automatically classify images into different categories such as animals, objects or scenes helps in the tasks like image search, content moderation and image-based product recommendations.
Medical diagnosis: It can assist in medical diagnosis by analyzing patient data such as medical images, test results and patient history to identify patterns that suggest specific diseases or conditions.
Fraud detection: They can analyze financial transactions and identify patterns that shows fraudulent activity which helps financial institutions prevent fraud and protect their customers.
Natural language processing (NLP): It plays a important role in NLP tasks including sentiment analysis, machine translation and text summarization which enables machines to understand and process human language effectively.
  
Advantages of Supervised learning
It learns from labeled examples to make accurate predictions on new, unseen data.
With more data and training, these models increases their accuracy which leads to better performance and more reliable predictions.
It works well for many tasks from detecting spam emails to predicting house prices as it has the ability to handle various computational challenges.
It can handle both classification (sorting data into categories) and regression (predicting numbers) which makes it flexible for different problems.

Disadvantages of Supervised learning
It requires a well-labeled dataset where each input has a corresponding output. Creating such datasets takes a lot of time, money and effort and can sometimes have mistakes, this makes supervised learning hard to use.
It works well on many tasks but can struggle with very complex or unstructured problems like understanding patterns or abstract ideas that doesn't relate to what it was trained on.
These models can sometimes overfit the training data which means they perform well on training data but poor on new, unseen data.
These models often need constant updating with new labeled data to stay accurate as real-world data changes over time.
What is Unsupervised learning?
Unsupervised learning is a part of machine learning which works differently from supervised because there is no teacher(supervisor) involved to guide the machine. In this approach the machine is given with data that has no labels or categories. It analyzes the data on its own to find patterns, groups or relationships without any prior knowledge. The machine learns by discovering hidden structures within the data without being told what the correct output should be.

For example, unsupervised learning can analyze animal data and group the animals by their traits and behavior. These groups might represent different species which allows the machine to organize animals without any prior labels or categories.

Unsupervised-learning

Example to understand
Imagine we have a machine learning model trained on many unlabeled images of dogs and cats. The model has never seen any labeled example that says “dog” or “cat” before so it doesn’t know how these animals look like.

Now, if we give the model a new image that contains both dogs and cats it won’t be able to directly label them as “dog” or “cat.” It will group parts of the image based on similarities and differences in features like shape or texture. It might separate the image into two groups one with dog-like features and other with cat-like features.

This happens because unsupervised learning doesn’t rely on prior knowledge or training with labeled data. It finds patterns and organizes data on its own helps in discovering information that wasn’t given before.

Types of Unsupervised Learning
Unsupervised learning is divided into two categories of algorithms: 

1. Clustering
A clustering is used to group similar data points together. Clustering algorithms work by repeatedly moving data points closer to to the center of their group (cluster) and farther from points in other groups. This helps the algorithm to create clear and meaningful clusters. Some popular clustering algorithms include:

K-means clustering
Hierarchical clustering
Principal Component Analysis (PCA)
Singular Value Decomposition (SVD)
Independent Component Analysis
Gaussian Mixture Models (GMMs)
Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

2. Association rule learning
An association rule learning used to find patterns and relationships between different items in a dataset. It looks for rules like “people who buy X often also buy Y”.

Some common Association rule learning algorithms include:

Apriori Algorithm
Eclat Algorithm
FP-Growth Algorithm
Application of Unsupervised learning
Unsupervised learning can be used to solve a variety of problems which includes:

Anomaly detection: It can identify unusual patterns or behaviors in data helps in the detection of fraud, security breaches or system problems.
Scientific discovery: It can show hidden relationships and patterns in scientific data which leads to new insights and ideas.
Recommendation systems: It finds similarities in user behavior and preferences to recommend products, movies or music that align with their interests.
Customer segmentation: It can identify groups of customers with similar characteristics which allows businesses to target marketing campaigns and improve customer service more effectively.
Advantages of Unsupervised learning
It doesn’t need labeled data so we can start working with large datasets more easily and quickly.
This handles large amounts of data and reduce it into simpler forms without losing important patterns which makes it manageable and efficient.
It discovers patterns and relationships in the data that were previously unknown which offers valuable insights.
By analyzing unlabeled data, it shows meaningful trends and groups that help us to understand our data deeply.
Disadvantages of Unsupervised learning
Without labeled answers, it’s difficult to tell how accurate or effective the model is.
Lack of clear guidance can lead to less precise results for complex problems.
After grouping the data, we may needs to check and label these groupings which can be time-consuming.
Missing data, outliers or noise in the data can easily affect the quality of the results.


                                  Supervised           vs     Unsupervised Machine Learning
Parameters	             Supervised machine learning	         Unsupervised machine learning
Input Data  	        They are trained on labeled data.	    They are trained on unlabeled data.
Computational Complexity 	    Simpler method	                Computationally complex
Output 	                 Desired output is given.	           Desired output is not given.
Training data  	     Use training data to infer model.	       No training data is used.
Complex model 	    It is not possible to learn larger     It is possible to learn larger and
                        and more complex models                more complex models with
                       with supervised learning.	             unsupervised learning.
Model 	                 We can test our model.             	We can not test our model.
Supervision          Supervised learning needs            Unsupervised learning does not need
                   supervision to train the model.        any supervision to train the model.
Classification        Divided into two types:                 Divided into two types:
                           i) Regression                          i)Clustering
                          ii) Classification                     ii)Association
Feedback            It has feedback mechanism.           It has no feedback mechanism.
Time Consumption    It's more time consuming.              It's less time consuming.
 Example 	          Optical character recognition.	       Find a face in an image.
 
With data growing every day, supervised and unsupervised learning will keep evolving which helps us to find new patterns and make better decisions in ways we can’t imagine yet.

