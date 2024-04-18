import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('twitter.csv')

# Assign labels to human-readable classes
df['label_name'] = df['label'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})

# Simulate train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df, df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

df['split'] = 'Training'  # Default to training
df.loc[X_test.index, 'split'] = 'Testing'  # Mark the test set

# Create summary table
summary_table = pd.crosstab(df['label_name'], df['split'], margins=True, margins_name='Total')
summary_table.columns = ['Training', 'Testing', 'Total']
summary_table.index = ['Negative', 'Neutral', 'Positive', 'Total']

print(summary_table)