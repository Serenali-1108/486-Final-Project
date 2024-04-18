import pandas as pd
import os
import sys
import ast
from joblib import load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

df = pd.read_csv('twitter.csv')

y = df['label']
X_reduced = load('X_reduced.joblib')
if not isinstance(X_reduced, pd.DataFrame):
    X_reduced = pd.DataFrame(X_reduced, index=df.index)
logging.info("Features and labels prepared.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
logging.info("Data split into training and testing sets.")

dual = X_train.shape[0] > X_train.shape[1]

class_weights = {
    1: 1,
    0: 248092 / 180509,
    -1: 248092 / 71435
}

# Testing a SVM model with selected parameters
svm_model = LinearSVC(C=1, dual=dual, class_weight=class_weights)
logging.info("SVM model training started.")
svm_model.fit(X_train, y_train)
logging.info("SVM model training completed.")

predictions = svm_model.predict(X_test)
logging.info("Predictions completed.")


output_df = pd.DataFrame({
    'Content': df.loc[X_test.index, 'content'],  
    'Polarity': df.loc[X_test.index, 'polarity'],
    'Actual Label': y_test,
    'Predicted Label': predictions
})

output_df.to_csv('linear_svm_output.csv', index=False)
print("Output saved to 'linear_svm_output.csv'.")

# Print the classification report
print(classification_report(y_test, predictions))




