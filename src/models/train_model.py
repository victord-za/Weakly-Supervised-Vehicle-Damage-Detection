import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv(find_dotenv())
DATA_FOLDER = os.getenv('DATA_FOLDER')
HOME_FOLDER = os.getenv('HOME_FOLDER')

# Define the transforms for data augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # values for pretrained torchvision models
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class VehicleDamageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_name = self.annotations.loc[index, "global_key"]
        img_path = os.path.join(self.root_dir, f"{img_name}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.annotations.iloc[index, 1:].values.astype(np.float32))
        return image, label

# Define the datasets
# Define the training set
train_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'processed' / 'train.csv', root_dir=Path(DATA_FOLDER) / 'processed' / 'train', transform=transform)
# Define the validation set
val_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'processed' / 'val.csv', root_dir=Path(DATA_FOLDER) / 'processed' / 'val', transform=val_transform)
# Define the test set
test_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'processed' / 'test.csv', root_dir=Path(DATA_FOLDER) / 'processed' / 'test', transform=val_transform)


# Define the model
#model = models.resnet50(pretrained=True)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = len(train_dataset.annotations.columns) - 1
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join(HOME_FOLDER, 'device_info.txt'), 'w') as f:
    f.write(f"Using device: {device}")

model = model.to(device)
min_val_loss = np.inf

num_epochs = 50
batch_size = 16


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Check for existing checkpoint
checkpoint_path = os.path.join(HOME_FOLDER, 'models', 'checkpoint.pt')

if os.path.exists(checkpoint_path):
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accuracies = checkpoint['train_accuracies']
    val_accuracies = checkpoint['val_accuracies']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    # No checkpoint was found
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')

# Define patience and initial best loss for early stopping
patience = 10
early_stopping_counter = 0

for epoch in range(start_epoch, num_epochs):
    # Training loop
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted_labels = torch.sigmoid(outputs) > 0.5
        correct_predictions += (predicted_labels == labels).float().sum().item()
        total_predictions += labels.numel()

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_predictions / total_predictions
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f"Train - Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted_labels = torch.sigmoid(outputs) > 0.5
            correct_predictions += (predicted_labels == labels).float().sum().item()
            total_predictions += labels.numel()

    # Calculate validation loss and accuracy
    val_loss = running_loss / len(val_dataloader)
    val_accuracy = correct_predictions / total_predictions
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Val - Epoch {epoch+1}/{num_epochs} - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f}")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stopping_counter = 0

        # Save a checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_val_loss': best_val_loss
        }, checkpoint_path)

    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load the best model state
#model.load_state_dict(best_model_state)

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(HOME_FOLDER, 'reports', 'figures', 'results', 'loss_plot.png'))

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(HOME_FOLDER, 'reports', 'figures', 'results', 'accuracy_plot.png'))



# Test set evaluation
model.eval()
test_running_loss = 0.0
test_correct_predictions = 0
test_total_predictions = 0
test_batch_size = 2
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

#with torch.no_grad():
#    for images, labels in test_dataloader:
#        images = images.to(device)
#        labels = labels.to(device)
#        outputs = model(images)
#        loss = criterion(outputs, labels)
#        test_running_loss += loss.item()
#        predicted = torch.sigmoid(outputs) > 0.5
#        test_correct_predictions += (predicted == labels).float().sum().item()
#        test_total_predictions += labels.numel()

# Calculate test set loss and accuracy
#test_epoch_loss = test_running_loss / len(test_dataloader)
#test_epoch_accuracy = test_correct_predictions / test_total_predictions
#print(f"Test Evaluation - Loss: {test_epoch_loss:.4f} - Accuracy: {test_epoch_accuracy:.4f}")

# Saliency map generation
model.eval()
class_names = train_dataset.annotations.columns[1:]  # Assuming the class names are the column names, excluding 'global_key'

for i, (images, labels) in enumerate(test_dataloader):
    images = images.to(device)
    images.requires_grad = True

    outputs = model(images)
    outputs = torch.sigmoid(outputs)

    for j in range(len(class_names)):
        model.zero_grad()
        outputs[:, j].backward(torch.ones_like(outputs[:, j]), retain_graph=True)

        saliency, _ = torch.max(images.grad.data.abs(), dim=1)
        saliency = saliency.cpu().numpy()
        # Save saliency maps
        for k in range(images.size(0)):
            # Get the original image
            original_image = images[k].permute(1, 2, 0).detach().cpu().numpy()
            # Reverse normalization
            original_image = original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            # Clip to be in range [0,1] (optional)
            original_image = np.clip(original_image, 0, 1)
            original_image_name = test_dataset.annotations.loc[i * batch_size + k, "global_key"]
            os.makedirs(f"{HOME_FOLDER}/reports/figures/saliency/test/{original_image_name}", exist_ok=True)

            # Get the saliency map for the current class and multiply by the class prediction
            class_saliency = saliency[k] * outputs[k, j].item()
            # Create a new figure, plot the original image, and add the saliency map overlay
            plt.figure()
            plt.imshow(original_image)
            plt.imshow(class_saliency, cmap='bwr', alpha=0.5)  # overlay saliency map
            plt.axis('off')

            plt.savefig(f"{HOME_FOLDER}/reports/figures/saliency/test/{original_image_name}/{class_names[j]}.jpg")
            plt.close()  # Closes the figure, so it doesn't get displayed in your Python environment
