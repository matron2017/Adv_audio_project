import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define your neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 512)  # Increase neurons to 512
        self.relu1 = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)  # Increase neurons to 256
        self.relu2 = nn.ReLU()
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Custom dataset class
class MelSpectrogramDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data, self.labels = self.load_data()

    def load_data(self):
        data = []
        labels = []

        for filename in os.listdir(self.data_path):
            if filename.endswith(".pkl"):
                filepath = os.path.join(self.data_path, filename)
                with open(filepath, 'rb') as f:
                    data_dict = pickle.load(f)
                    mel_spectrogram = data_dict['mel_spectrogram']
                    label = data_dict['label']
                    data.append(mel_spectrogram)
                    labels.append(label)
                    #print(f"Label: {label}")

        return np.array(data), np.array(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]), torch.tensor(self.labels[index])

# Function to train the model
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return loss.item()

# Function to validate the model
def validate_model(model, val_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Set paths and parameters
output_directory = "output_mel_spectrograms/"
metadata_file = "Data/meta.csv"
num_classes = 10
input_shape = (128, 216)  # Adjust according to your mel-spectrogram shape



# Load mel-spectrogram data and labels
dataset = MelSpectrogramDataset(output_directory)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(dataset.labels)
print(f"Unique labels: {label_encoder.classes_}")
#print(f"Encoded labels: {encoded_labels}")

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(dataset.data, encoded_labels, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#Print number of total samples
print(f"Number of total samples: {len(dataset.data)}")

#Print number of train, test and validation samples
print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
print(f"Number of testing samples: {len(X_test)}")

# Create DataLoader instances
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=128, shuffle=True)
val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = SimpleNN(np.prod(input_shape), num_classes)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Adjust parameters as needed

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    # Train the model
    train_loss = train_model(model, train_loader, criterion, optimizer)
    scheduler.step()  # Update learning rate schedule
    
    # Validate the model
    val_accuracy = validate_model(model, val_loader, criterion)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Test the model
test_accuracy = test_model(model, test_loader, criterion)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model and label encoder
torch.save(model.state_dict(), 'pytorch_model.pth')
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

# Visualize confusion matrix and save the figure
def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    else:
        plt.show()

# Plot confusion matrix
y_true = y_test
model.eval()
with torch.no_grad():
    y_pred = []
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())

# Specify the directory to save the figures
save_directory = os.path.dirname(os.path.realpath(__file__))

plot_confusion_matrix(y_true, y_pred, classes=label_encoder.classes_, save_path=save_directory)

# Plot training and validation loss/accuracy over epochs and save the figure
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title('Training and Validation Metrics')
plt.legend()
plt.grid(True)
if save_directory:
    plt.savefig(os.path.join(save_directory, 'training_validation_metrics.png'))
else:
    plt.show()

