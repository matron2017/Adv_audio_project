import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pickle

# Define your neural network model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, num_classes)  # Adjust input size based on your mel-spectrogram dimensions

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Custom dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, data_folder, file_list):
        self.data_folder = data_folder
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Perform any additional preprocessing if necessary
        mel_spectrogram = data['mel_spectrogram']
        label = data['label']

        return mel_spectrogram, label

# Set your data folder
data_folder = "output_mel_spectrograms/"

# Split the dataset into training and validation sets
file_list = os.listdir(data_folder)
train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

# Create datasets and data loaders
train_dataset = SpectrogramDataset(data_folder, train_files)
val_dataset = SpectrogramDataset(data_folder, val_files)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
num_classes = 9  # Number of classes in your dataset
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    for mel_spectrogram, label in train_loader:
        optimizer.zero_grad()
        mel_spectrogram = mel_spectrogram.unsqueeze(1)  # Add channel dimension for 1-channel input
        output = model(mel_spectrogram)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for mel_spectrogram, label in val_loader:
            mel_spectrogram = mel_spectrogram.unsqueeze(1)
            output = model(mel_spectrogram)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)

    accuracy = total_correct / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')
