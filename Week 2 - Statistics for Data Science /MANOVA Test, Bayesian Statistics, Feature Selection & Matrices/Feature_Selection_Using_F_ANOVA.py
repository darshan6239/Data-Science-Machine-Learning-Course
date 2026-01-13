#        Implementation of F-ANOVA for Feature Selection 
#        In this example we will use the f_classif function from sklearn.feature_selection to find the F-statistic for each feature and select the top features based on the           highest F-scores.

#        Step 1: Loading Iris Dataset from sklearn.datasets
#        We will be using Pandas and Scikit-learn libraries for its implementation.

#        train_test_split(X, y, test_size=0.3, random_state=42): Splits the dataset X (features) and y (target) into training and testing sets with 30% of data allocated to           testing and a random seed set for reproducibility.
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#        Step 2: F-ANOVA Feature Selection
#        We use SelectKBest from sklearn.feature_selection with the scoring function f_classif which finds the F-statistic. Here we select the top 2 features (k=2) but we             can modify k to select more or fewer features.
selector = SelectKBest(score_func=f_classif, k=2)  
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
f_scores = selector.scores_[selector.get_support()]
print(f"Selected Features: {selected_features}")
print(f"F-Scores: {f_scores}")

#        OUTPUT 
#        Selected Features: Index(['petal length (cm)', 'petal width (cm)'], dtype='object') F-Scores: [713.45534904 526.54162416]

#        Step 3: Model training and Evaluation
#        We train a Random Forest classifier using only the selected features.

model = RandomForestClassifier(random_state=42)
model.fit(X_train_selected, y_train)

y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model with selected features: {accuracy:.4f}")

#  OUTPUT
#  Accuracy of the model with selected features: 1.0000
