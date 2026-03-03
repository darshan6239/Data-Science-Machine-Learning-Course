""" Categorical data refers to features that contain a fixed set of possible values or categories that data points can belong to. Handling categorical data correctly is important because improper handling can lead to inaccurate analysis and poor model performance. In this article, we will see how to handle categorical data and its related concepts.

Why Do We Need to Handle Categorical Data?
Handling categorical data is important because:

Algorithms Require Numerical Inputs: Most machine learning algorithms cannot directly process categorical data and need it to be converted into numerical formats.
Inconsistent Categories: Categorical data contains inconsistencies like typos, case sensitivity or alternate spellings. We must standardize these to avoid treating them as separate categories.
Remapping Categories: Some categories might need to be grouped for simplicity and relevance. For example, remapping rare categories into an "Other" group.
Improves Model Performance: Proper encoding techniques like one-hot encoding or label encoding help models to understand the relationships of categories leading to better predictions.
Handles Real-World Complexity: It is used in many domains such as E-commerce, Finance, Healthcare, etc making it robust to handle important features. """


""" Implementation for Handling Categorical Data
Here we will be using a Demographics dataset which has some incorrect, invalid or meaningless data (bogus values) due to human error while filling survey form or any other reason. You can download dataset from here.

Step 1: Importing necessary Libraries
We will be using Numpy, Pandas, Matplotlib, Seaborn and Scikit-learn libraries for its implementation. """ 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

""" Step 2: Loading the Dataset
We load the dataset into a Pandas DataFrame for manipulation. """
file_path = '/content/demographics.csv'
main_data = pd.read_csv(file_path)
print(main_data.head())

""" Step 3: Identifying and Removing Bogus Blood Types
First we create a DataFrame containing all valid blood types to check for bogus values in the dataset:""" 
valid_blood_type_list = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
blood_type_categories = pd.DataFrame({'blood_type': valid_blood_type_list})
print(blood_type_categories)

#  Lets find bogus blood types by comparing the dataset values to this valid list:
unique_blood_types_main = set(main_data['blood_type'])
valid_blood_types_set = set(blood_type_categories['blood_type'])  
bogus_blood_types = unique_blood_types_main.difference(valid_blood_types_set)
bogus_blood_types

#  Once the bogus values are found the corresponding rows can be dropped from the dataset.
bogus_records_index = main_data['blood_type'].isin(bogus_blood_types)

without_bogus_records = main_data[~bogus_records_index].copy()
without_bogus_records['blood_type'].unique()

#  Step 4: Handling Inconsistent Marriage Status Categories
#  Checking the unique values in the marriage_status column:
main_data['marriage_status'].unique()

#  Standardizing the categories by converting all text to lowercase.
inconsistent_data = main_data.copy()
inconsistent_data['marriage_status'] = inconsistent_data['marriage_status'].str.lower()
inconsistent_data['marriage_status'].unique()

#  Now we will standardize the categories by stripping extra spaces:
inconsistent_data['marriage_status'] = inconsistent_data['marriage_status'].str.strip()
inconsistent_data['marriage_status'].unique()

#  Step 5: Grouping Income into Meaningful Bins
#  Numerical data like age or income can be mapped to different groups. Let us check income range to define bin intervals:
print(f"Max income - {main_data['income'].max()}, Min income - {main_data['income'].min()}")

#  Now, let us create the range and labels for the income feature. Pandas cut method is used here.
income_bins = [40000, 75000, 100000, 125000, 150000, np.inf]
income_labels = ['40k-75k', '75k-100k', '100k-125k', '125k-150k', '150k+']

remapping_data = main_data.copy()
remapping_data['income_groups'] = pd.cut(
    remapping_data['income'],
    bins=income_bins,
    labels=income_labels
)
remapping_data.head()

#    Step 6: Visualizing Income Group Distribution
#    Now lets visualize the distribution of income groups:
remapping_data['income_groups'].value_counts().sort_index().plot.bar()
plt.title('Income Group Distribution')
plt.xlabel('Income Groups')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#    Step 7: Cleaning Phone Number Data
#    Simulating phone numbers with inconsistent formats and cleaning them:
import random
phone_numbers = []

for i in range(100):
    number = random.randint(100000000, 9999999999)  # length can be 9 or 10 digits
    if i % 2 == 0:
        phone_numbers.append('+91 ' + str(number))  # add +91 prefix for some
    else:
        phone_numbers.append(str(number))

phone_numbers_data = pd.DataFrame({
    'phone_numbers': phone_numbers
})
phone_numbers_data.head()

#    Based on the use case the country code before numbers could be dropped or added for missing ones. Similarly phone numbers with less than 10 numbers should be discarded.
phone_numbers_data['phone_numbers'] = phone_numbers_data['phone_numbers'].str.replace(r'\+91 ', '', regex=True)
num_digits = phone_numbers_data['phone_numbers'].str.len()
invalid_numbers_index = phone_numbers_data[num_digits < 10].index
phone_numbers_data.drop(invalid_numbers_index, inplace=True)
phone_numbers_data.dropna(inplace=True)
phone_numbers_data.reset_index(drop=True, inplace=True)
phone_numbers_data.head()

#    Finally we can verify whether the data is clean or not.
assert not phone_numbers_data['phone_numbers'].str.contains(r'\+91 ').any(), "Found phone numbers with '+91 ' prefix"
assert (phone_numbers_data['phone_numbers'].str.len() == 10).all(), "Some phone numbers do not have 10 digits"

#    Step 8: Visualizing Categorical Data
#    Various plots could be used to visualize categorical data to get more insights about the data. So let us visualize the number of people belonging to each blood type.
import seaborn as sns
sns.countplot(x='blood_type', data=without_bogus_records)
plt.title('Count of Blood Types')
plt.xlabel('Blood Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#    Now we can see the relationship between income and the marital status of a person using a boxplot. 
sns.boxplot(x='marriage_status', y='income', data=inconsistent_data)

plt.title('Income Distribution by Marriage Status')
plt.xlabel('Marriage Status')
plt.ylabel('Income') 
plt.tight_layout()
plt.show()

#    Step 9: Encoding Categorical Data
#    Certain learning algorithms like regression and neural networks require their input to be numbers. Hence categorical data must be converted to numbers to use these algorithms. Let us see some encoding methods.

#    1. Label Encoding

#    With label encoding we can number the categories from 0 to num_categories - 1. Let us apply label encoding on the blood type feature.
le = LabelEncoder()
without_bogus_records['blood_type_encoded'] = le.fit_transform(without_bogus_records['blood_type'])

without_bogus_records[['blood_type', 'blood_type_encoded']].drop_duplicates()

#    2. One-hot Encoding in Python

#    There are certain limitations of label encoding that are taken care of by one-hot encoding. Some of them are:

"""1) Creates a false order: It gives numbers like 0, 1, 2 to categories which may make models think one category is bigger or better than the other.
    2)Misleads models: Algorithms like linear regression or decision trees might assume there's a ranking which can reduce accuracy.
    3)Problem with distance-based models: In models like KNN or K-Means, the numeric labels can wrongly influence distance calculations.
    4) Bias in training: Some models may give more importance to higher label values, even if all categories are equal.
    5) Not suitable for nominal data: Label encoding is not a good choice when categories have no natural order, like colors or city names. """ 
inconsistent_data = pd.get_dummies(inconsistent_data, columns=['marriage_status'])
inconsistent_data.head()

#    3. Ordinal Encoding in Python

#    Categorical data can be ordinal where the order is of importance. For such features, we want to preserve the order after encoding as well. We will perform ordinal encoding on income groups. We want to preserve the order as 40K-75K < 75K-100K < 100K-125K < 125K-150K < 150K+
custom_map = {
    '40k-75k': 1,
    '75k-100k': 2,
    '100k-125k': 3,
    '125k-150k': 4,
    '150k+': 5
}
remapping_data['income_groups_encoded'] = remapping_data['income_groups'].map(custom_map)
remapping_data[['income', 'income_groups', 'income_groups_encoded']].head()


#    With these techniques we can prepare categorical data for meaningful analysis and effective machine learning models.
