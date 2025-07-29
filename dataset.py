import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

class LandCoverDataset(Dataset):
    def __init__(self, split='train', regions=['Urban', 'Rural']):
        """
        Land Cover Segmentation Dataset
        
        Args:
            split: 'train', 'val', 'test'  
            regions: List of regions to include
            
        Note: Augmentation already done in preprocessing, so we only need normalization
        """
        self.split = split
        self.regions = regions
        
        # Simple normalization only - dataset already augmented during preprocessing
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Build file paths
        self.image_paths = []
        self.mask_paths = []
        
        split_path = os.path.join(Config.DATASET_PATH, split.capitalize())
        
        for region in regions:
            region_path = os.path.join(split_path, region)
            img_dir = os.path.join(region_path, 'images_png')
            mask_dir = os.path.join(region_path, 'masks_png')
            
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
                
                for img_file in img_files:
                    mask_file = img_file  # Same filename for mask
                    
                    img_path = os.path.join(img_dir, img_file)
                    mask_path = os.path.join(mask_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)
        
        print(f"{split.upper()} dataset: {len(self.image_paths)} samples")
        if split == 'train':
            # Count original vs augmented
            orig_count = len([p for p in self.image_paths if '_aug' not in os.path.basename(p)])
            aug_count = len(self.image_paths) - orig_count
            print(f"  Original images: {orig_count}")
            print(f"  Augmented images: {aug_count}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No data found for {split} split in {Config.DATASET_PATH}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Ensure proper size (should already be 512x512 from preprocessing)
        if image.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
            image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
            mask = cv2.resize(mask, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Apply transforms (normalization + tensor conversion)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropy
        mask = mask.long()
        
        return image, mask
    
    def get_class_distribution(self):
        """Calculate class distribution for weighting"""
        class_counts = np.zeros(Config.NUM_CLASSES)
        
        print("Calculating class distribution...")
        sample_size = min(100, len(self.image_paths))  # Sample for speed
        indices = np.random.choice(len(self.image_paths), sample_size, replace=False)
        
        for idx in indices:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < Config.NUM_CLASSES:
                    class_counts[class_id] += count
        
        # Calculate weights (inverse frequency)
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (Config.NUM_CLASSES * class_counts + 1e-6)
        
        print("Class distribution:")
        for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
            print(f"  Class {i}: {count:.0f} pixels, weight: {weight:.3f}")
        
        return class_weights

def create_dataloaders():
    """Create train, validation and test dataloaders"""
    
    # Create datasets
    train_dataset = LandCoverDataset('train')
    val_dataset = LandCoverDataset('val') 
    
    # Try to create test dataset, but handle case where no test data exists
    try:
        test_dataset = LandCoverDataset('test')
        has_test_data = True
    except ValueError as e:
        print(f"Warning: {e}")
        print("Using validation set as test set for evaluation.")
        test_dataset = val_dataset  # Use validation set as test set
        has_test_data = False
    
    # Calculate class weights from training data
    class_weights = train_dataset.get_class_distribution()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # For easier visualization
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"\nDataLoader Summary:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    if not has_test_data:
        print("Note: Test dataset is using validation data (no separate test masks found)")
    
    return train_loader, val_loader, test_loader, class_weights

if __name__ == "__main__":
    # Test the dataset
    train_loader, val_loader, test_loader, weights = create_dataloaders()
    
    # Test a batch
    for images, masks in train_loader:
        print(f"Batch shape - Images: {images.shape}, Masks: {masks.shape}")
        print(f"Image range: {images.min():.3f} to {images.max():.3f}")
        print(f"Mask classes: {masks.unique()}")
        break