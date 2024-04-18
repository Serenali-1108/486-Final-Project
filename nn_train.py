#!pip install torch torchvision torchaudio --upgrade
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
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt



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

# More layers, batch normalization
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
# my_stop_words = text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords'])
# my_stop_words = list(my_stop_words)
# max_df = 0.75
# min_df = 0.001
# vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, stop_words=my_stop_words)
# X_reduced = vectorizer.fit_transform(df['tokens'])

y = df['label']

X_reduced = load('X_reduced.joblib')
if not isinstance(X_reduced, pd.DataFrame):
    X_reduced = pd.DataFrame(X_reduced, index=df.index)

if hasattr(X_reduced, "toarray"):
    # Convert sparse matrix to dense
    X_reduced = X_reduced.toarray()
elif isinstance(X_reduced, pd.DataFrame):
    # Use values from DataFrame
    X_reduced = X_reduced.values

logging.info("Features and labels prepared.")

#  transform -1, 0, 1 to 0, 1, 2 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 0.6 train, 0.2 val, 0.2 test
# Split data into training and testing sets
X_train_val, X_test, y_train_val, y_test, indices_train_val, indices_test = train_test_split(
    X_reduced, y_encoded, range(len(y_encoded)), test_size=0.2, random_state=42, stratify=y_encoded
)

# Further split and keep the indices for validation
X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
    X_train_val, y_train_val, indices_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

class_weights = {
    1: 1,
    0: 248092 / 180509,
    -1: 248092 / 71435
}

logging.info("Data split into training and testing sets.")

# Convert the data into tensors
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()
X_val_tensor = torch.tensor(X_val).float()
y_val_tensor = torch.tensor(y_val).long()

np.save('indices_test.npy', indices_test) # Save indices for testing output
np.save('X_test.npy', X_test)  # Save X_test as .npy file
np.save('y_test.npy', y_test)      # Save Y_test as .npy file

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# we have 3 classes (-1, 0, and 1)
num_classes = 3
input_dim = X_train_tensor.shape[1]  # Number of features
model = SimpleNN(input_dim, num_classes)

weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight = weights_tensor)

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 15
best_val_loss = float('inf')
best_model_params = None

logging.info("Training started")
# Lists to store loss per epoch
train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_train_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    average_train_loss = total_train_loss / len(train_loader)
    train_losses.append(average_train_loss)

    scheduler.step()  # update learning rate

    # Evaluate on validation set
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    average_val_loss = total_val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    # Track the best model
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        best_model_params = model.state_dict()  # Save the best model parameters

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {average_train_loss:.4f} - Validation Loss: {average_val_loss:.4f}")

if best_model_params:
    torch.save(best_model_params, 'best_model.pth')
    print(f"Best model saved with validation loss: {best_val_loss:.4f}")
else:
    print("No improvement observed, model not saved.")

# Plotting the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('train_val_loss_plot.png')
