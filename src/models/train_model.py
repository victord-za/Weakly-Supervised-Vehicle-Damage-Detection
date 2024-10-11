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
import wandb
import sys
import time
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import multilabel_confusion_matrix, classification_report


start_time = time.time()
timeout_flag = False
TIMEOUT_THRESHOLD = 9.25 * 60 * 60  # 9.25 hours in seconds
PREDICTION_THRESHOLD = 0.5

# Load environment variables from .env file
load_dotenv(find_dotenv())
DATA_FOLDER = os.getenv('DATA_FOLDER')
HOME_FOLDER = os.getenv('HOME_FOLDER')
sys.stdout = open(os.path.join(os.getenv('HOME_FOLDER'), 'reports', 'training_log.txt'), 'w')

# Set up weights and biases for logging
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')
os.environ['WANDB_ENTITY'] = os.getenv('WANDB_ENTITY')
os.environ['WANDB_PROJECT'] = os.getenv('WANDB_PROJECT')
wandb.init(project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_ENTITY'], resume="allow")

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        original_length = len(self.annotations)
        valid_indices = self.annotations['global_key'].apply(lambda x: os.path.isfile(os.path.join(self.root_dir, f"{x}.jpg")))
        self.annotations = self.annotations[valid_indices].reset_index(drop=True)
        new_length = len(self.annotations)
        files_dropped = original_length - new_length
        drop_percentage = (files_dropped / original_length) * 100
        print(f"Number of files dropped: {files_dropped} ({drop_percentage:.2f}%)")

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
train_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'train.csv', root_dir=Path(DATA_FOLDER) / 'final' / 'train', transform=transform)
val_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'val.csv', root_dir=Path(DATA_FOLDER) / 'final' / 'val', transform=val_transform)
test_dataset = VehicleDamageDataset(csv_file=Path(DATA_FOLDER) / 'test.csv', root_dir=Path(DATA_FOLDER) / 'final' / 'test', transform=val_transform)

# Define the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_classes = len(train_dataset.annotations.columns) - 1
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

min_val_loss = np.inf
num_epochs = 20
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Check for existing checkpoint
best_checkpoint_path = os.path.join(HOME_FOLDER, 'models', 'best_checkpoint.pt')
latest_checkpoint_path = os.path.join(HOME_FOLDER, 'models', 'latest_checkpoint.pt')

if os.path.exists(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path)
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
    if time.time() - start_time > TIMEOUT_THRESHOLD:
        print(f"Timeout threshold of {TIMEOUT_THRESHOLD} seconds reached. Stopping training.")
        wandb.finish()
        sys.stdout.close()
        timeout_flag = True
        break

    # Training loop
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predicted_labels = torch.sigmoid(outputs) > PREDICTION_THRESHOLD
        correct_predictions += (predicted_labels == labels).float().sum().item()
        total_predictions += labels.numel()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted_labels.cpu().numpy())

    # Calculate training loss and accuracy
    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_predictions / total_predictions
    train_f1 = f1_score(all_labels, all_predictions, average='macro')
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Train - Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - F1: {train_f1:.4f}")
    wandb.log({
        "Train Loss": train_loss,
        "Train Accuracy": train_accuracy,
        "Train F1 Score": train_f1,
        "Epoch": epoch
    })

    # Validation loop
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predicted_labels = torch.sigmoid(outputs) > PREDICTION_THRESHOLD
            correct_predictions += (predicted_labels == labels).float().sum().item()
            total_predictions += labels.numel()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    # Calculate validation loss and accuracy
    val_loss = running_loss / len(val_dataloader)
    val_accuracy = correct_predictions / total_predictions
    val_f1 = f1_score(all_labels, all_predictions, average='macro')
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Val - Epoch {epoch+1}/{num_epochs} - Loss: {val_loss:.4f} - Accuracy: {val_accuracy:.4f} - F1: {val_f1:.4f}")
    wandb.log({
        "Validation Loss": val_loss,
        "Validation Accuracy": val_accuracy,
        "Validation F1 Score": val_f1,
        "Epoch": epoch
    })

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        early_stopping_counter = 0
        # Save the best model
        torch.save(model.state_dict(), os.path.join(HOME_FOLDER, 'models', 'best_model.pt'))
        print(f"Saved best model at epoch {epoch+1}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

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
    }, latest_checkpoint_path)

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
if not(timeout_flag):
    model.eval()
    test_running_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0
    test_batch_size = 8
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > PREDICTION_THRESHOLD
            test_correct_predictions += (predicted == labels).float().sum().item()
            test_total_predictions += labels.numel()
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # Calculate F1 scores
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_micro = f1_score(all_labels, all_predictions, average='micro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')

    # Calculate test set loss and accuracy
    test_epoch_loss = test_running_loss / len(test_dataloader)
    test_epoch_accuracy = test_correct_predictions / test_total_predictions

    print(f"Test Evaluation - Loss: {test_epoch_loss:.4f} - Accuracy: {test_epoch_accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}, Micro F1: {f1_micro:.4f}, Weighted F1: {f1_weighted:.4f}")

    # Log to wandb
    wandb.log({
        "Test Loss": test_epoch_loss,
        "Test Accuracy": test_epoch_accuracy,
        "Test Macro F1": f1_macro,
        "Test Micro F1": f1_micro,
        "Test Weighted F1": f1_weighted
    })

    # Calculate confusion matrix for each class
    conf_matrices = multilabel_confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix for each class
    class_names = train_dataset.annotations.columns[1:]  # Assuming the class names are the column names, excluding 'global_key'
    for i, conf_matrix in enumerate(conf_matrices):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix: {class_names[i]}')
        plt.savefig(os.path.join(HOME_FOLDER, 'reports', 'figures', 'results', f'confusion_matrix_{class_names[i]}.png'))
        plt.close()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    wandb.finish()
    sys.stdout.close()
    
# GradCAM
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=torch.cuda.is_available())
    class_names = train_dataset.annotations.columns[1:]  # Assuming the class names are the column names, excluding 'global_key'

    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        for j in range(len(class_names)):
            grayscale_cam = cam(input_tensor=images, target_category=j)
            
            for k in range(images.size(0)):
                original_image = images[k].permute(1, 2, 0).detach().cpu().numpy()
                original_image = original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                original_image = np.clip(original_image, 0, 1)

                visualization = show_cam_on_image(original_image, grayscale_cam[k, :], use_rgb=True)

                original_image_name = test_dataset.annotations.loc[i * test_batch_size + k, "global_key"]
                os.makedirs(f"{HOME_FOLDER}/reports/figures/saliency/test/{original_image_name}", exist_ok=True)

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(original_image)
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(visualization)
                plt.title(f'GradCAM: {class_names[j]}')
                plt.axis('off')

                plt.savefig(f"{HOME_FOLDER}/reports/figures/saliency/test/{original_image_name}/{class_names[j]}_gradcam.jpg")
                plt.close()

    print("GradCAM saliency maps generated.")

print("Training, evaluation, and saliency map generation completed.")