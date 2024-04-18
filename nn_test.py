
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# class SimpleNN(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(512, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(256, 256)
#         self.bn4 = nn.BatchNorm1d(256)
#         self.fc5 = nn.Linear(256, 128)
#         self.bn5 = nn.BatchNorm1d(128)
#         self.fc6 = nn.Linear(128, 128)
#         self.bn6 = nn.BatchNorm1d(128)
#         self.fc7 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = F.leaky_relu(self.bn1(self.fc1(x)))
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn2(self.fc2(x)))
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn3(self.fc3(x)))
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn4(self.fc4(x)))
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn5(self.fc5(x)))
#         x = self.dropout(x)
#         x = F.leaky_relu(self.bn6(self.fc6(x)))
#         x = self.fc7(x)
#         return x
df = pd.read_csv('twitter.csv')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
indices_test = np.load('indices_test.npy')  # Load saved test indices

# Generate label_encoder again
label_encoder = LabelEncoder()
original_labels = [-1, 0, 1]
label_encoder.fit(original_labels)

X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).long()

num_classes = 3
input_dim = X_test_tensor.shape[1]
model = SimpleNN(input_dim, num_classes)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    test_loss = criterion(outputs, y_test_tensor).item()

# Inverse transform the labels from {0, 1, 2} back to {-1, 0, 1}
predicted_np = label_encoder.inverse_transform(predicted.cpu().numpy())
y_test_np = label_encoder.inverse_transform(y_test_tensor.cpu().numpy())

output_df = pd.DataFrame({
    'Content': df.iloc[indices_test]['content'],
    'Polarity': df.iloc[indices_test]['polarity'],
    'Actual Label': y_test_np,
    'Predicted Label': predicted_np
})

output_df.to_csv('nn_output.csv', index=False)
print("Output saved to 'nn_output.csv'.")

# Print the classification report
report = classification_report(y_test_np, predicted_np)
print(report)
