import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('twitter.csv')

# Applying stratified sampling, retain 1% of data  (5k)
df_sample, _ = train_test_split(df, test_size=0.99, stratify=df['label'], random_state=42)

df_sample.to_csv('0.01_twitter.csv', index=False)
