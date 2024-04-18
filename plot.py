import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

file_name = 'log_reg_output.csv'
df = pd.read_csv(file_name)  # Load the dataset

# Calculate confusion matrix (actual label vs. predicted label)
conf_matrix = confusion_matrix(df['Actual Label'], df['Predicted Label'], normalize='true')

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
plt.title(f'Confusion Matrix of Actual vs Predicted Labels ({file_name})')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.savefig(f'matrix_{file_name}.png')

# Box plot (Actual polarity vs. predicted label)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Predicted Label', y='Polarity', whis=[0, 100], width=0.5)
sns.stripplot(data=df, x='Predicted Label', y='Polarity', color='black', size=1, jitter=True, alpha=0.5)
plt.title(f'Box and Strip Plot of Polarity by Predicted Label({file_name})')
plt.xlabel('Predicted Label')
plt.ylabel('Polarity')
plt.savefig(f'boxplot_{file_name}.png')

