import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import time
import copy
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Verify dataset structure exists
data_dir = "/home/liunazhou/ie4428/Faces"

# Check if dataset exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

# Find all subdirectories (these are the class labels)
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
if not subdirs:
    raise FileNotFoundError(f"No subdirectories (classes) found in {data_dir}")

print(f"Found {len(subdirs)} classes: {subdirs}")

# Check for JPG files in each subdirectory
all_jpg_files = []
for subdir in subdirs:
    subdir_path = os.path.join(data_dir, subdir)
    jpg_files = glob.glob(os.path.join(subdir_path, '*.jpg'))
    all_jpg_files.extend(jpg_files)
    print(f"Found {len(jpg_files)} JPG files in class '{subdir}'")

if len(all_jpg_files) == 0:
    print(f"Warning: No JPG files found in any subdirectory. Make sure your images are in .jpg format.")
else:
    print(f"Found a total of {len(all_jpg_files)} JPG files across {len(subdirs)} classes")

# Data transforms for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),            # Resize images to 224x224 (ResNet input size)
        transforms.RandomHorizontalFlip(),        # Data augmentation
        transforms.RandomRotation(10),            # Additional augmentation
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Additional augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Standard normalization for pre-trained models
                             std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# Load the full dataset (where each subdirectory is a class)
try:
    # Load the full dataset using ImageFolder
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=None  # We'll apply transforms later
    )
    
    # Print class mapping
    class_to_idx = full_dataset.class_to_idx
    print("Class to index mapping:")
    for class_name, idx in class_to_idx.items():
        print(f"  {class_name}: {idx}")
    
    # Split dataset into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Apply the appropriate transforms to each subset
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4,
                           pin_memory=True if torch.cuda.is_available() else False),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4,
                         pin_memory=True if torch.cuda.is_available() else False)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = full_dataset.classes
    
    print(f"Dataset loaded successfully:")
    print(f"Training images: {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['val']}")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
except Exception as e:
    raise Exception(f"Error loading dataset: {str(e)}")

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Modify the final fully connected layer to match the number of classes in your dataset
num_features = model.fc.in_features
num_classes = len(class_names)
model.fc = nn.Linear(num_features, num_classes)

def set_finetune_layers(model, n):
    """
    Unfreeze the last n top-level modules of the model.
    For ResNet50, model.children() returns:
      [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
    If n=1, only fc is trainable; if n=2, avgpool and fc are trainable; etc.
    """
    children = list(model.children())
    total_children = len(children)
    # Ensure n does not exceed the number of modules
    n = min(n, total_children)

    # Freeze all modules except the last n modules
    for idx, child in enumerate(children):
        if idx < total_children - n:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
    print(f"Unfroze the last {n} out of {total_children} modules.")

n = 1  # Adjusted to unfreeze the last 2 layers for better fine-tuning
set_finetune_layers(model, n)

# Print trainable parameters
trainable_params = 0
total_params = 0
print("Trainable parameters:")
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print(name)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# Move the model to the appropriate device
model = model.to(device)

# Define loss function and optimizer; only parameters with requires_grad=True will be updated
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Add learning rate scheduler
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

num_epochs = 25  # adjust the number of epochs as needed
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
checkpoints_dir = "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

since = time.time()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 30)
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluation mode

        running_loss = 0.0
        running_corrects = 0
        
        # For per-class accuracy tracking
        all_preds = []
        all_labels = []
        
        # Iterate over data
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass (with gradient tracking during training)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels for per-class metrics
            if phase == 'val':
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
        
        # Print per-class metrics for validation
        if phase == 'val':
            # Calculate per-class accuracy
            per_class_accuracies = {}
            confusion = confusion_matrix(all_labels, all_preds)
            # Per-class accuracy is the diagonal divided by the sum across rows
            per_class_acc = confusion.diagonal() / confusion.sum(axis=1)
            
            print("\nPer-class validation accuracy:")
            for i, (class_name, acc) in enumerate(zip(class_names, per_class_acc)):
                print(f"  {class_name}: {acc:.4f}")
                per_class_accuracies[class_name] = acc
            
            # Print classification report for more detailed metrics
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=class_names))
            
            # Update scheduler on validation loss
            scheduler.step(epoch_loss)
            
            # Save checkpoint if best model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'acc': epoch_acc,
                    'class_to_idx': class_to_idx,
                    'per_class_accuracies': per_class_accuracies
                }, os.path.join(checkpoints_dir, f"best_model_epoch_{epoch+1}.pth"))
                print(f"Saved new best model with accuracy: {best_acc:.4f}")
        
        # Save checkpoint every 5 epochs to avoid losing progress
        if epoch % 5 == 0 and phase == 'val':
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'class_to_idx': class_to_idx,  # Save class mapping for inference
                'per_class_accuracies': per_class_accuracies if phase == 'val' else None
            }, os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch+1}.pth"))
            
    print()

time_elapsed = time.time() - since
print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
print(f'Best validation accuracy: {best_acc:.4f}')

# Load best model weights
model.load_state_dict(best_model_wts)

# Save the final model with class mapping
torch.save({
    'model_state_dict': model.state_dict(),
    'class_to_idx': class_to_idx,  # Important to save the mapping between classes and indices
    'classes': class_names
}, "resnet50_face_recognition_finetuned.pth")
print("Training complete and model saved!")