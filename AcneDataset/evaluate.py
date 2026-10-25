import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from model import AcneClassifier, get_inference_transforms, CLASSES

def evaluate_test_set(model_path='acne_classifier.pth', test_dir='test', output_dir='.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running Evaluation on Test Set using: {model_path} ({device})")
    
    transform = get_inference_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = AcneClassifier(num_classes=len(CLASSES), pretrained=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded trained weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with default initialized weights.")
        
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate Overall Accuracy
    accuracy = np.mean(all_preds == all_labels) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Per-class metrics
    class_p, class_r, class_f1, class_supp = precision_recall_fscore_support(all_labels, all_preds, average=None)
    
    per_class_metrics = {}
    for idx, cls_name in enumerate(CLASSES):
        per_class_metrics[cls_name] = {
            'precision': round(float(class_p[idx]) * 100, 2),
            'recall': round(float(class_r[idx]) * 100, 2),
            'f1_score': round(float(class_f1[idx]) * 100, 2),
            'support': int(class_supp[idx])
        }
        
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Acne Classification Confusion Matrix\nTest Accuracy: {accuracy:.2f}%', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Ground Truth Label', fontsize=11)
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()
    
    results = {
        'test_accuracy': round(accuracy, 2),
        'weighted_f1': round(f1 * 100, 2),
        'weighted_precision': round(precision * 100, 2),
        'weighted_recall': round(recall * 100, 2),
        'total_test_samples': len(all_labels),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_plot': cm_path
    }
    
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n================ TEST EVALUATION RESULTS ================")
    print(f"Overall Test Accuracy : {accuracy:.2f}%")
    print(f"Weighted F1-Score     : {f1*100:.2f}%")
    print(f"Weighted Precision    : {precision*100:.2f}%")
    print(f"Weighted Recall       : {recall*100:.2f}%")
    print(f"=========================================================\n")
    print("Per-Class Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    
    return results

if __name__ == '__main__':
    evaluate_test_set()
