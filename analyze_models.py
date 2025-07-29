import torch
import os
import glob
from config import Config

def analyze_all_models():
    """Analyze all available model checkpoints"""
    print("=== MODEL CHECKPOINT ANALYSIS ===\n")
    
    checkpoints_dir = os.path.dirname(Config.MODEL_SAVE_PATH)
    
    if not os.path.exists(checkpoints_dir):
        print("No checkpoints directory found!")
        return
    
    # Find all checkpoint files
    patterns = ['*.pth', '*.pt']
    checkpoint_files = []
    
    for pattern in patterns:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoints_dir, pattern)))
    
    if not checkpoint_files:
        print("No checkpoint files found!")
        return
    
    models_info = []
    
    for file_path in sorted(checkpoint_files):
        try:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            filename = os.path.basename(file_path)
            epoch = checkpoint.get('epoch', 'N/A')
            accuracy = checkpoint.get('metrics', {}).get('accuracy', 0)
            iou = checkpoint.get('metrics', {}).get('mean_iou', 0)
            f1 = checkpoint.get('metrics', {}).get('mean_f1', 0)
            loss = checkpoint.get('loss', 'N/A')
            timestamp = checkpoint.get('timestamp', 'Unknown')
            
            # Combined score
            score = (accuracy + iou) / 2 if isinstance(accuracy, (int, float)) and isinstance(iou, (int, float)) else 0
            
            models_info.append({
                'filename': filename,
                'epoch': epoch,
                'accuracy': accuracy,
                'iou': iou,
                'f1': f1,
                'loss': loss,
                'score': score,
                'timestamp': timestamp,
                'path': file_path
            })
            
        except Exception as e:
            print(f"Error loading {os.path.basename(file_path)}: {e}")
            continue
    
    # Sort by combined score (descending)
    models_info.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Found {len(models_info)} valid model checkpoints:\n")
    print(f"{'Rank':<4} {'Filename':<25} {'Epoch':<6} {'Accuracy':<9} {'IoU':<9} {'F1':<9} {'Score':<9} {'Timestamp'}")
    print("=" * 100)
    
    for i, info in enumerate(models_info, 1):
        print(f"{i:<4} {info['filename']:<25} {info['epoch']:<6} "
              f"{info['accuracy']:<9.4f} {info['iou']:<9.4f} {info['f1']:<9.4f} "
              f"{info['score']:<9.4f} {info['timestamp']}")
    
    if models_info:
        best_model = models_info[0]
        print(f"\n🏆 BEST MODEL (by combined score):")
        print(f"   File: {best_model['filename']}")
        print(f"   Epoch: {best_model['epoch']}")
        print(f"   Accuracy: {best_model['accuracy']:.4f}")
        print(f"   IoU: {best_model['iou']:.4f}")
        print(f"   F1: {best_model['f1']:.4f}")
        print(f"   Combined Score: {best_model['score']:.4f}")
        print(f"   Timestamp: {best_model['timestamp']}")
        
        # Compare with current best_model.pth
        current_best_path = Config.MODEL_SAVE_PATH
        if os.path.exists(current_best_path):
            current_is_best = best_model['path'] == current_best_path
            print(f"\n📋 Current best_model.pth is {'✅ optimal' if current_is_best else '❌ NOT optimal'}")
            
            if not current_is_best:
                print(f"   Recommendation: Replace best_model.pth with {best_model['filename']}")
                print(f"   Command: copy \"{best_model['path']}\" \"{current_best_path}\"")

if __name__ == "__main__":
    analyze_all_models()
