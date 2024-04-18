'''
Hyper-parameter tuning script for Logistic Regression using sampled dataset
Result: Best parameters found: {'C': 1, 'penalty':'l2'', 'solver':'newton-cg'}
Best parameters used in Logistic Regression model in full dataset
'''

import pandas as pd
import os
import sys
import ast
from joblib import load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
df = pd.read_csv('sampled_twitter.csv')

my_stop_words = text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords'])
my_stop_words = list(my_stop_words)
max_df = 0.75
min_df = 0.001
vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=my_stop_words)
X_reduced = vectorizer.fit_transform(df['tokens'])

y = df['label']
logging.info("Features and labels prepared.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
logging.info("Data split into training and testing sets.")

# Define class weights based on category distribution
class_weights = {
    1: 1,
    0: 248092 / 180509,
    -1: 248092 / 71435
}

param_grid = {
    # 'newton-cg', 'saga'
    'solver': ['lbfgs','newton-cg', 'saga'],
    'penalty': ['l2'],
    'C': [0.01, 0.1, 1],
    'class_weight': [class_weights]

}

# Train a Logistic Regression model
model = GridSearchCV(LogisticRegression(max_iter=10000, multi_class='auto'), param_grid, cv=3, verbose=2, scoring='accuracy')
logging.info("Starting grid search for best model parameters.")
model.fit(X_train, y_train)
logging.info("Grid search completed.")

# Best model parameters
print("Best parameters found:", model.best_params_)

# Predict using the best model
best_model = model.best_estimator_
predictions = best_model.predict(X_test)
logging.info("Predictions completed.")

# Generate and print the classification report
report = classification_report(y_test, predictions)
print(report)
logging.info("Classification report generated and displayed.")
