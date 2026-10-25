import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

from model import AcneClassifier, get_train_transforms, get_inference_transforms, CLASSES

def calculate_class_weights(dataset):
    """Calculate inverse frequency class weights to balance training"""
    targets = [label for _, label in dataset.samples]
    class_counts = np.bincount(targets, minlength=len(CLASSES))
    total_samples = len(targets)
    weights = total_samples / (len(CLASSES) * (class_counts + 1e-5))
    return torch.FloatTensor(weights)

def train_model(train_dir='train', valid_dir='valid', epochs=5, batch_size=32, lr=0.001, output_path='acne_classifier.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Acne Classifier Training on device: {device}")
    
    # Datasets and Loaders
    train_transform = get_train_transforms()
    valid_transform = get_inference_transforms()
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset summary: Train={len(train_dataset)} images, Valid={len(valid_dataset)} images")
    print(f"Classes: {train_dataset.classes}")
    
    # Class weights for loss
    class_weights = calculate_class_weights(train_dataset).to(device)
    print(f"Calculated Class Weights: {class_weights.cpu().numpy().round(3)}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize Model
    model = AcneClassifier(num_classes=len(CLASSES), pretrained=True)
    model.to(device)
    
    # Phase 1: Freeze backbone features, train classifier head
    print("\n--- Phase 1: Training Classification Head ---")
    for param in model.backbone.features.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.Adam(model.backbone.classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze top layers after epoch 2 for fine-tuning
        if epoch == 3:
            print("\n--- Phase 2: Fine-Tuning High-Level Convolutional Layers ---")
            for param in model.backbone.features[-4:].parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {'params': model.backbone.features[-4:].parameters(), 'lr': lr * 0.1},
                {'params': model.backbone.classifier.parameters(), 'lr': lr * 0.5}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 2)

        # Train loop
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)
            
            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (correct_train / total_train) * 100
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f} Acc: {batch_acc:.2f}%")
                
        scheduler.step()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)
                
        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(round(epoch_train_loss, 4))
        history['train_acc'].append(round(epoch_train_acc, 2))
        history['val_loss'].append(round(epoch_val_loss, 4))
        history['val_acc'].append(round(epoch_val_acc, 2))
        
        print(f"[Epoch {epoch}/{epochs}] ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        # Save best checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': epoch_val_acc,
                'val_loss': epoch_val_loss,
                'classes': CLASSES
            }, output_path)
            print(f"   --> Best model checkpoint saved to {output_path} (Val Acc: {best_val_acc:.2f}%)")

    total_training_time = time.time() - start_time
    print(f"\n[OK] Training completed in {total_training_time/60:.2f} minutes. Peak Val Acc: {best_val_acc:.2f}%")
    
    # Save training history
    with open('model_metrics.json', 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'epochs': epochs,
            'history': history,
            'classes': CLASSES,
            'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
        
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Acne Classification Model')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs (default: 4)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output', type=str, default='acne_classifier.pth', help='Output model path')
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_path=args.output)
