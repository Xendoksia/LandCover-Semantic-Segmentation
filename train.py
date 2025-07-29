import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter  # Optional dependency
import numpy as np
import os
import shutil
from tqdm import tqdm
import time
from datetime import datetime

# Import custom modules
from config import Config
from model import create_model
from dataset import create_dataloaders
from utils import (
    get_loss_function, MetricsCalculator, visualize_predictions,
    plot_training_history, plot_confusion_matrix, get_class_names,
    get_colored_mask
)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, accuracy, iou, model):
        # Combined score: equal weight to accuracy and IoU
        current_score = (accuracy + iou) / 2
        
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, criterion, optimizer, device, metrics_calculator):
    model.train()
    running_loss = 0.0
    metrics_calculator.reset()
    
    # No gradient accumulation needed for batch size 8
    accumulation_steps = 1  # Direct batch size = 8
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (images, masks) in enumerate(pbar):
        try:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping to prevent explosion
                if hasattr(Config, 'GRADIENT_CLIPPING') and Config.GRADIENT_CLIPPING > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIPPING)
                
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps  # Unscale for logging
            
            # Calculate metrics (with memory cleanup)
            with torch.no_grad():
                predictions = torch.argmax(outputs.detach(), dim=1)
                metrics_calculator.update(predictions, masks)
                del predictions  # Explicit cleanup
            
            # Less frequent cache clearing for better performance
            if batch_idx % 10 == 0:  # More frequent cache clearing
                torch.cuda.empty_cache()
            
            # Cleanup intermediate tensors
            del outputs
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item() * accumulation_steps:.4f}'})
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\nOOM at batch {batch_idx}, clearing cache and skipping...")
                torch.cuda.empty_cache()
                optimizer.zero_grad()  # Clear any accumulated gradients
                continue
            else:
                raise e
    
    # Final gradient step if needed
    if len(train_loader) % accumulation_steps != 0:
        if hasattr(Config, 'GRADIENT_CLIPPING') and Config.GRADIENT_CLIPPING > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIPPING)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    metrics = metrics_calculator.get_metrics()
    
    return epoch_loss, metrics

def validate_epoch(model, val_loader, criterion, device, metrics_calculator):
    model.eval()
    running_loss = 0.0
    metrics_calculator.reset()
    
    # Clear cache before validation
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(pbar):
            try:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                running_loss += loss.item()
                
                # Calculate metrics with memory cleanup
                predictions = torch.argmax(outputs.detach(), dim=1)
                metrics_calculator.update(predictions, masks)
                
                # Cleanup
                del outputs, predictions
                
                # Less frequent cache clearing for validation too
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM in validation at batch {batch_idx}, clearing cache and skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    epoch_loss = running_loss / len(val_loader)
    metrics = metrics_calculator.get_metrics()
    
    return epoch_loss, metrics

def cleanup_old_milestones(checkpoints_dir, keep_last=2):
    """Keep only the last N milestone files to save space"""
    try:
        import glob
        milestone_files = glob.glob(os.path.join(checkpoints_dir, '*_milestone_*.pth'))
        
        if len(milestone_files) > keep_last:
            # Sort by creation time and keep only the newest ones
            milestone_files.sort(key=os.path.getctime)
            files_to_delete = milestone_files[:-keep_last]  # Delete all but the last N
            
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"🗑️  Removed old milestone: {os.path.basename(file_path)}")
                except:
                    pass
    except:
        pass  # Silent fail if cleanup doesn't work

def save_checkpoint(model, optimizer, epoch, loss, metrics, filepath):
    """Save model checkpoint"""
    import time
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': {
            'num_classes': Config.NUM_CLASSES,
            'image_size': Config.IMAGE_HEIGHT,
            'model_name': getattr(Config, 'MODEL_NAME', 'unet'),
            'encoder': getattr(Config, 'ENCODER', 'resnet34'),
            'batch_size': Config.BATCH_SIZE,
            'learning_rate': Config.LEARNING_RATE,
            'loss_type': getattr(Config, 'LOSS_TYPE', 'combined')
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'best_accuracy': metrics.get('accuracy', 0),
        'best_iou': metrics.get('mean_iou', 0),
        'best_f1': metrics.get('mean_f1', 0)
    }
    
    torch.save(checkpoint, filepath)
    
    if 'best_model.pth' in filepath:
        print(f"✅ Best model updated")
    elif 'milestone' in filepath:
        print(f"💾 Milestone saved: {os.path.basename(filepath)}")
    
    print(f"   📊 Acc: {metrics.get('accuracy', 0):.4f}, IoU: {metrics.get('mean_iou', 0):.4f}, F1: {metrics.get('mean_f1', 0):.4f}")

def train_model():
    # Setup
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory before start: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB total")
    
    # Create directories
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Get checkpoints directory for cleanup
    checkpoints_dir = os.path.dirname(Config.MODEL_SAVE_PATH)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders()
    
    # Create model
    print("Creating model...")
    print(f"Model: {getattr(Config, 'MODEL_NAME', 'unet')}")
    print(f"Encoder: {getattr(Config, 'ENCODER', 'resnet34')}")
    model = create_model()
    model = model.to(device)
    
    # Loss function with class weights (simplified)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    criterion = get_loss_function(class_weights_tensor)
    
    # Advanced optimizer with weight decay
    if hasattr(Config, 'OPTIMIZER') and Config.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=getattr(Config, 'WEIGHT_DECAY', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=1e-4
        )
    
    # Advanced learning rate scheduler
    if hasattr(Config, 'SCHEDULER') and Config.SCHEDULER == 'cosine_with_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=getattr(Config, 'MIN_LR', 1e-6)
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE) if Config.EARLY_STOPPING else None
    
    # Metrics calculator
    train_metrics_calc = MetricsCalculator()
    val_metrics_calc = MetricsCalculator()
    
    # Training timestamp for logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Training started at: {timestamp}")
    
    # Training history
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_val_accuracy = 0.0
    
    print(f"\nStarting training for {Config.EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 30)
        
        # Training
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics_calc
        )
        
        # Validation
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, val_metrics_calc
        )
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train IoU: {train_metrics['mean_iou']:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Smart model saving: only save when there are real improvements
        current_accuracy = val_metrics['accuracy']
        current_iou = val_metrics['mean_iou']
        current_f1 = val_metrics['mean_f1']
        
        # Track improvements
        accuracy_improved = current_accuracy > best_val_accuracy
        iou_improved = current_iou > best_val_iou
        
        # Save only if significant improvement in key metrics
        if accuracy_improved or iou_improved:
            # Update best values
            if accuracy_improved:
                best_val_accuracy = current_accuracy
            if iou_improved:
                best_val_iou = current_iou
            best_val_loss = val_loss
            
            # Save the improved model
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_metrics,
                Config.MODEL_SAVE_PATH
            )
            
            improvements = []
            if accuracy_improved:
                improvements.append(f"Accuracy: {current_accuracy:.4f}")
            if iou_improved:
                improvements.append(f"IoU: {current_iou:.4f}")
            
            print(f"🎯 NEW BEST MODEL SAVED! {', '.join(improvements)}")
        else:
            print(f"⏸️  No key improvements - Current: Acc={current_accuracy:.4f}, IoU={current_iou:.4f}")
        
        # Periodic milestone saves (every 15 epochs) for backup
        if (epoch + 1) % 15 == 0:
            milestone_path = Config.MODEL_SAVE_PATH.replace('.pth', f'_milestone_e{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics, milestone_path)
            print(f"🏁 Milestone backup saved: epoch {epoch+1}")
            
            # Clean old milestones (keep only last 2)
            cleanup_old_milestones(checkpoints_dir)
        
        # Early stopping - use combined score instead of loss
        if early_stopping is not None:
            current_accuracy = val_metrics['accuracy']
            current_iou = val_metrics['mean_iou']
            if early_stopping(current_accuracy, current_iou, model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"No improvement in combined score (accuracy + IoU) for {early_stopping.patience} epochs")
                break
        
        print("=" * 60)
    
    # Training completed - clear cache
    torch.cuda.empty_cache()
    
    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics_calc = MetricsCalculator()
    test_loss, test_metrics = validate_epoch(
        model, test_loader, criterion, device, test_metrics_calc
    )
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Per-class results
    class_names = get_class_names()
    print(f"\nPer-class IoU:")
    for i, (name, iou) in enumerate(zip(class_names, test_metrics['per_class_iou'])):
        print(f"  {name}: {iou:.4f}")
    
    # Plot training history
    print("\nGenerating plots...")
    plot_training_history(
        train_losses, val_losses, train_metrics_history, val_metrics_history,
        save_path=os.path.join(Config.RESULTS_DIR, 'training_history.png')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics_calc.confusion_matrix,
        class_names=class_names,
        save_path=os.path.join(Config.RESULTS_DIR, 'confusion_matrix.png')
    )
    
    # Visualize some predictions
    print("Visualizing predictions...")
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= 5:  # Only first 5 batches
                break
            
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            visualize_predictions(
                images, masks, predictions, num_samples=min(4, images.size(0)),
                save_path=os.path.join(Config.RESULTS_DIR, f'predictions_batch_{i}.png')
            )
    
    print(f"\nTraining completed!")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Results saved in: {Config.RESULTS_DIR}")
    print(f"Model saved as: {Config.MODEL_SAVE_PATH}")
    
    # Final memory cleanup
    torch.cuda.empty_cache()

if __name__ == "__main__":
    train_model()