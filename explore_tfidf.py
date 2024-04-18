'''

Step 1.
Use small dataset to choose min_df and max_df parameters for tf-idf vectorizer
1. Remove stopwords
2. Use grid search to explore values of min_df, max_df
3. Train a preliminary SVM model using each min_df and max_df, cross-validating
with 5-folds
4. Choose max_df and min_df with highest cross-validated accuracy


RESULT:
Best parameters: {'tfidfvectorizer__max_df': 0.75, 'tfidfvectorizer__min_df': 0.001}
Best cross-validation score: 0.76

Step 2.
Use small dataset to choose num_components for SVD
1. Use grid search to explore num_components
2. Train a simple logistic regression model using each num_component,
 cross validating with 5-folds
3. Plot cross-validated accuracy and select optimal num_components

'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt



############### Exploratory analysis to find best min_df, max_df ###############
def find_min_max_df(df, my_stop_words):
    param_grid = {
        'tfidfvectorizer__min_df': [0.001, 0.01],
        'tfidfvectorizer__max_df': [0.5, 0.75, 0.9],
    }

    pipeline = Pipeline([
        ('tfidfvectorizer', TfidfVectorizer(stop_words=my_stop_words)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Instantiate the grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(df['tokens'], df['label'])  

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    return grid_search.best_params_

def find_num_components(X_tfidf, y):

    scores = []
    n_components_options = [10, 100, 500, 1000, 1500, 2000]

    for n in n_components_options:
        svd = TruncatedSVD(n_components=n)
        X_reduced = svd.fit_transform(X_tfidf)
        model = LogisticRegression(max_iter=1000)
        score = np.mean(cross_val_score(model, X_reduced, y, cv=5))
        print(f"n: {n}, score: {score}")
        scores.append(score)

    plt.figure(figsize=(10, 6))
    plt.plot(n_components_options, scores, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Model Performance vs. Number of Components')
    plt.grid(True)
    plt.show()



# small_twitter contains 10k posts, about 1/50 of original size
df = pd.read_csv('small_twitter.csv')


my_stop_words = text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords'])
my_stop_words = list(my_stop_words)

#best_param = find_min_max_df(df, my_stop_words)
#max_df = best_param['tfidfvectorizer__max_df']
#min_df = best_param['tfidfvectorizer__min_df']
max_df = 0.75
min_df = 0.001

vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=my_stop_words)

# Extract feature matrix
X_tfidf = vectorizer.fit_transform(df['tokens'])
# Extract target variable
y = df['label']

find_num_components(X_tfidf, y)

