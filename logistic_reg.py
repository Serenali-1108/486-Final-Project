import pandas as pd
import os
import sys
import ast
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset
df = pd.read_csv('twitter.csv')

# Assuming 'label' is the target and 'X_reduced.joblib' is the preprocessed feature set
y = df['label']
X_reduced = load('X_reduced.joblib')

if not isinstance(X_reduced, pd.DataFrame):
    X_reduced = pd.DataFrame(X_reduced, index=df.index)

logging.info("Features and labels prepared.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
logging.info("Data split into training and testing sets.")

# Define class weights based on category distribution
class_weights = {
    1: 1,
    0: 248092 / 180509,
    -1: 248092 / 71435
}

# Train a Logistic Regression model
logistic_model = LogisticRegression(class_weight=class_weights, max_iter=10000, random_state=42, penalty = 'l2', C = 1, solver='newton-cg', multi_class='auto')
logging.info("Logistic Regression model training started.")
logistic_model.fit(X_train, y_train)
logging.info("Logistic Regression model training completed.")

# Make predictions on the test set
predictions = logistic_model.predict(X_test)
logging.info("Predictions completed.")


output_df = pd.DataFrame({
    'Content': df.loc[X_test.index, 'content'],  
    'Polarity': df.loc[X_test.index, 'polarity'],
    'Actual Label': y_test,
    'Predicted Label': predictions
})

output_df.to_csv('log_reg_output.csv', index=False)
print("Output saved to 'log_reg_output.csv'.")


report = classification_report(y_test, predictions)
with open('log_reg_report.txt', 'w') as file:
    file.write(report)

logging.info("Classification report generated and displayed.")
