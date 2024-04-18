'''
Hyper-parameter tuning script for SVM using sampled dataset
Result: Best parameters found: {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
Best parameters used in Linear SVM model using full dataset
'''
import pandas as pd
from joblib import load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import TfidfVectorizer
import tqdm

df = pd.read_csv('0.01_twitter.csv')

my_stop_words = text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords'])
my_stop_words = list(my_stop_words)
max_df = 0.75
min_df = 0.001
vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=my_stop_words)
X_reduced = vectorizer.fit_transform(df['tokens'])

y = df['label']
#X_reduced = load('X_reduced.joblib')

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

param_grid = {
    'C': [0.1, 1], 
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'sigmoid']  # Regularization parameter
}

# Create a GridSearchCV object
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2)

grid_search.fit(X_train, y_train)
best_svm = grid_search.best_estimator_
print("Best parameters found:", grid_search.best_params_)

#Predict testing set using best model
predictions = best_svm.predict(X_test)

# Print the classification report
print(classification_report(y_test, predictions))




