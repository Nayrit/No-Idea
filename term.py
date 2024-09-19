# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Loading the Thyroid Disease Dataset from UCI
url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data'
url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-test.data'

# Loading the dataset
column_names = [
    'T3-resin uptake test', 'Total serum thyroxin', 'Total serum triiodothyronine', 
    'Basal thyroid-stimulating hormone', 'Maximal absolute difference of TSH', 'Target'
]

train_data = pd.read_csv(url_train, delim_whitespace=True, header=None, names=column_names)
test_data = pd.read_csv(url_test, delim_whitespace=True, header=None, names=column_names)

# Combine train and test datasets for more data
data = pd.concat([train_data, test_data], axis=0)

# Checking if we have more than 1000 records
print(f"Total records: {data.shape[0]}")

# Examining the dataset structure
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Convert the target into a binary classification (0 or 1)
# In this dataset, 'Target' is the last column and it has 3 classes: 1, 2, 3
# We will convert it to binary classification (normal vs not normal) 
# Assuming Class 3 is "normal" and others are "abnormal"
data['Target'] = data['Target'].apply(lambda x: 0 if x == 3 else 1)

# Handle missing values
data = data.dropna()

# Splitting the dataset into features and target variable
X = data.drop(['Target'], axis=1)
y = data['Target']


test_size = 0.02  

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Initializing the models
dt_model = DecisionTreeClassifier(random_state=42)
knn_model = KNeighborsClassifier()

# Training the models
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Evaluating the models
y_pred_dt = dt_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Calculating metrics for both models
metrics = ['accuracy', 'precision', 'recall', 'f1']
results = {}

for model_name, y_pred in zip(['Decision Tree', 'KNN'], [y_pred_dt, y_pred_knn]):
    results[model_name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

# Printing the results
print('Model Evaluation:')
for model_name, scores in results.items():
    print(f'\n{model_name}:')
    for metric in metrics:
        print(f'{metric.capitalize()}: {scores[metric]:.4f}')

# Determining which model performs better in terms of F1-score
better_model = max(results, key=lambda x: results[x]['f1'])
print(f'\nBetter model in terms of F1-score: {better_model}')
