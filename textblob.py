'''
Use TextBlob to label each Twitter post
Use TextBlob to tokenize each post
Save resulting dataframe as twitter.csv
'''
# Data preprocessing
import pandas as pd
import sys
import os
from textblob import TextBlob

df = pd.read_csv('twitter.csv')

tokens_list = []
polarity_list = []
labels_list = []

for i,text in enumerate(df['content']):
    print(i)
    text = str(text)
    blob = TextBlob(text)
    
    tokens_list.append(list(blob.words))
    
    polarity_list.append(blob.sentiment.polarity)
   
    # Assigning labels based on polarity
    if blob.sentiment.polarity > 0:
        labels_list.append(1)
    elif blob.sentiment.polarity < 0:
        labels_list.append(-1)
    else:
        labels_list.append(0)

print(f"tokens_list: {tokens_list[:10]}")
print()
print(f"polarity_list: {polarity_list[:10]}")
# Adding the lists as new columns to the DataFrame
df['tokens'] = tokens_list
df['polarity'] = polarity_list
df['label'] = labels_list

print(df.head())

df.to_csv('twitter.csv', index = False)