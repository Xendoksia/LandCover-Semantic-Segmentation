import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from config import Config
import os
import numpy as np


def get_colored_mask(mask):
    color_map = {
        0: (0, 0, 0),           # No data - Black
        1: (169, 169, 169),     # Background - Gray
        2: (255, 0, 0),         # Building - Red
        3: (255, 255, 0),       # Road - Yellow
        4: (0, 0, 255),         # Water - Blue
        5: (139, 69, 19),       # Barren - Brown
        6: (34, 139, 34),       # Forest - Green
        7: (0, 255, 255),       # Agriculture - Cyan
    }

    height, width = mask.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)

    for cls_id, color in color_map.items():
        colored[mask == cls_id] = color

    return colored

def get_class_names():
    return [
        "No data",
        "Background",
        "Building",
        "Road",
        "Water",
        "Barren",
        "Forest",
        "Agriculture"
    ]
# Loss Functions
class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=Config.NUM_CLASSES).permute(0, 3, 1, 2).float()
        
        # Flatten
        predictions = predictions.contiguous().view(-1)
        targets_one_hot = targets_one_hot.contiguous().view(-1)
        
        intersection = (predictions * targets_one_hot).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets_one_hot.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined Dice + CrossEntropy Loss"""
    def __init__(self, class_weights=None, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        ce = self.ce_loss(predictions, targets)
        return self.dice_weight * dice + self.ce_weight * ce

def get_loss_function(class_weights=None):
    """Get loss function based on config"""
    if Config.LOSS_TYPE == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif Config.LOSS_TYPE == "dice":
        return DiceLoss()
    elif Config.LOSS_TYPE == "focal":
        return FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, weight=class_weights)
    elif Config.LOSS_TYPE == "combined":
        return CombinedLoss(class_weights, Config.DICE_WEIGHT, Config.CE_WEIGHT)
    else:
        raise ValueError(f"Unknown loss type: {Config.LOSS_TYPE}")

# Metrics
class MetricsCalculator:
    """Calculate segmentation metrics"""
    def __init__(self, num_classes=Config.NUM_CLASSES):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions, targets):
        """Update confusion matrix"""
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        mask = (targets >= 0) & (targets < self.num_classes)
        self.confusion_matrix += confusion_matrix(
            targets[mask], predictions[mask], labels=range(self.num_classes)
        )
    
    def get_metrics(self):
        """Calculate metrics from confusion matrix"""
        eps = 1e-6
        
        # Per-class metrics
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        # Overall metrics
        accuracy = tp.sum() / (self.confusion_matrix.sum() + eps)
        mean_iou = np.mean(iou)
        mean_f1 = np.mean(f1)
        
        return {
            'accuracy': accuracy,
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'per_class_iou': iou,
            'per_class_f1': f1,
            'per_class_precision': precision,
            'per_class_recall': recall
        }

# Visualization Functions
class_colors = {
    0: [0, 0, 0],           # Background - Black
    1: [169, 169, 169],     # Class 1 - Gray
    2: [255, 0, 0],         # Class 2 - Red
    3: [255, 255, 0],       # Class 3 - Yellow
    4: [0, 0, 255],         # Class 4 - Blue
    5: [139, 69, 19],       # Class 5 - Brown
    6: [34, 139, 34],       # Class 6 - Green
    7: [0, 255, 255],       # Class 7 - Cyan
}

def mask_to_rgb(mask):
    """Convert segmentation mask to RGB image"""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        rgb_mask[mask == class_id] = color
    
    return rgb_mask

def visualize_predictions(images, true_masks, pred_masks, num_samples=4, save_path=None):
    """Visualize predictions"""
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        true_mask = true_masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()
        
        # Original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # True mask
        axes[i, 1].imshow(mask_to_rgb(true_mask))
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Predicted mask
        axes[i, 2].imshow(mask_to_rgb(pred_mask))
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        overlay = img.copy()
        pred_rgb = mask_to_rgb(pred_mask) / 255.0
        overlay = 0.6 * overlay + 0.4 * pred_rgb
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_confusion_matrix(confusion_matrix, class_names=None, save_path=None):
    """Plot confusion matrix"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    plt.figure(figsize=(10, 8))
    
    # Check if confusion matrix contains integers or floats
    if confusion_matrix.dtype.kind in 'iu':  # integer types
        fmt = 'd'
    else:  # float types
        fmt = '.2f'
    
    sns.heatmap(confusion_matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, save_path=None):
    """Plot training history"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU
    ax2.plot(epochs, [m['mean_iou'] for m in train_metrics], 'b-', label='Training IoU')
    ax2.plot(epochs, [m['mean_iou'] for m in val_metrics], 'r-', label='Validation IoU')
    ax2.set_title('Mean IoU')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU')
    ax2.legend()
    ax2.grid(True)
    
    # Accuracy
    ax3.plot(epochs, [m['accuracy'] for m in train_metrics], 'b-', label='Training Accuracy')
    ax3.plot(epochs, [m['accuracy'] for m in val_metrics], 'r-', label='Validation Accuracy')
    ax3.set_title('Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # F1 Score
    ax4.plot(epochs, [m['mean_f1'] for m in val_metrics], 'r-', label='Validation F1')
    ax4.set_title('F1 Score')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('F1 Score')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()