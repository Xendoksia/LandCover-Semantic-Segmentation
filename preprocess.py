import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Configuration - Updated for multiple dataset splits
BASE_DIR = r"-----------------------------------------"  # Parent directory containing Train, Val, Test
OUTPUT_DIR = r"-------------------------------------------"  # Will create same structure but processed

# Dataset splits to process
DATASET_SPLITS = ["Train", "Val", "Test"]  # Add or remove splits as needed
REGIONS = ["Urban", "Rural"]

# Preprocessing settings
TARGET_SIZE = 512  # Recommended: 512 for better detail, 256 for faster training
ENABLE_AUGMENTATION = True
AUGMENTATION_FACTOR = 2  # How many augmented versions per original image

# Augmentation settings - Different for each split
AUG_SETTINGS = {
    'Train': {  # More aggressive augmentation for training
        'horizontal_flip': True,
        'vertical_flip': True,
        'rotation_90': True,
        'small_rotation': False,
        'brightness_adjust': False,
    },
    'Val': {  # Light augmentation for validation
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_90': False,
        'small_rotation': False,
        'brightness_adjust': False,
    },
    'Test': {  # No augmentation for test set
        'horizontal_flip': False,
        'vertical_flip': False,
        'rotation_90': False,
        'small_rotation': False,
        'brightness_adjust': False,
    }
}

CLASS_COLORS = {
    0: [0, 0, 0], 1: [169, 169, 169], 2: [255, 0, 0], 3: [255, 255, 0],
    4: [0, 0, 255], 5: [139, 69, 19], 6: [34, 139, 34], 7: [0, 255, 255],
}

def analyze_image_sizes(base_dir):
    """Analyze current image sizes to understand resizing impact"""
    all_sizes = []
    
    for split in DATASET_SPLITS:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            continue
            
        for region in REGIONS:
            image_dir = os.path.join(split_path, region, "images_png")
            
            if not os.path.exists(image_dir):
                continue
            
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')][:10]  # Sample first 10 per region
            
            for filename in image_files:
                img_path = os.path.join(image_dir, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    h, w = image.shape[:2]
                    all_sizes.append((w, h))
    
    return all_sizes

def recommend_target_size(all_sizes):
    """Recommend target size based on current image sizes"""
    if not all_sizes:
        return TARGET_SIZE
    
    widths = [s[0] for s in all_sizes]
    heights = [s[1] for s in all_sizes]
    
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)
    avg_size = (avg_width + avg_height) / 2
    
    print(f"\nImage Size Analysis:")
    print(f"  Average size: {avg_width:.0f} x {avg_height:.0f}")
    print(f"  Size range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
    
    # Recommend based on average size
    if avg_size > 800:
        recommended = 512
        print(f"  📊 Recommendation: Use {recommended}x{recommended} (moderate downscaling)")
        print(f"      ⚠️ Warning: Some detail loss expected from larger images")
    elif avg_size > 400:
        recommended = 512
        print(f"  📊 Recommendation: Use {recommended}x{recommended} (minimal scaling)")
    else:
        recommended = 256
        print(f"  📊 Recommendation: Use {recommended}x{recommended} (avoid upscaling)")
        print(f"      ⚠️ Warning: Small images detected, consider keeping original size")
    
    return recommended

def resize_image_and_mask(image, mask, target_size=512, method='pad'):
    """
    Resize image and mask with different strategies
    
    Args:
        image: input image
        mask: input mask
        target_size: target size (will be target_size x target_size)
        method: 'stretch', 'pad', or 'crop'
    """
    h, w = image.shape[:2]
    
    if method == 'stretch':
        # Simple resize (may distort aspect ratio)
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
    elif method == 'pad':
        # Maintain aspect ratio with padding
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize maintaining aspect ratio
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create padded images
        image_resized = np.zeros((target_size, target_size, 3), dtype=image.dtype)
        mask_resized = np.zeros((target_size, target_size), dtype=mask.dtype)
        
        # Center the scaled image
        y_offset = (target_size - new_h) // 2
        x_offset = (target_size - new_w) // 2
        
        image_resized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_scaled
        mask_resized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_scaled
        
    elif method == 'crop':
        # Center crop to target size
        if h < target_size or w < target_size:
            # If smaller, pad first
            return resize_image_and_mask(image, mask, target_size, 'pad')
        
        y_start = (h - target_size) // 2
        x_start = (w - target_size) // 2
        
        image_resized = image[y_start:y_start+target_size, x_start:x_start+target_size]
        mask_resized = mask[y_start:y_start+target_size, x_start:x_start+target_size]
    
    return image_resized, mask_resized

def apply_augmentation(image, mask, aug_type):
    """Apply specific augmentation to image and mask"""
    if aug_type == 'horizontal_flip':
        image_aug = cv2.flip(image, 1)
        mask_aug = cv2.flip(mask, 1)
        
    elif aug_type == 'vertical_flip':
        image_aug = cv2.flip(image, 0)
        mask_aug = cv2.flip(mask, 0)
        
    elif aug_type == 'rotation_90':
        k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
        image_aug = np.rot90(image, k)
        mask_aug = np.rot90(mask, k)
        
    elif aug_type == 'small_rotation':
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_aug = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
        mask_aug = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
    elif aug_type == 'brightness_adjust':
        factor = random.uniform(0.8, 1.2)
        image_aug = np.clip(image * factor, 0, 255).astype(image.dtype)
        mask_aug = mask.copy()  # Mask unchanged
        
    else:
        return image, mask
    
    return image_aug, mask_aug

def create_augmented_versions(image, mask, filename_base, split_name):
    """Create augmented versions of image and mask based on split type"""
    augmented_data = []
    
    # Original - keep original filename structure
    augmented_data.append((image, mask, f"{filename_base}.png"))
    
    # Skip augmentation if disabled globally or no augmentations for this split
    split_aug_settings = AUG_SETTINGS.get(split_name, {})
    if not ENABLE_AUGMENTATION or not any(split_aug_settings.values()):
        return augmented_data
    
    # Apply enabled augmentations for this split
    aug_count = 1
    
    for aug_type, enabled in split_aug_settings.items():
        if enabled and aug_count < AUGMENTATION_FACTOR + 1:
            try:
                img_aug, mask_aug = apply_augmentation(image, mask, aug_type)
                # Better naming: originalname_aug001_fliptype.png
                aug_filename = f"{filename_base}_aug{aug_count:03d}_{aug_type}.png"
                augmented_data.append((img_aug, mask_aug, aug_filename))
                aug_count += 1
            except Exception as e:
                print(f"Warning: Failed to apply {aug_type} to {filename_base}: {e}")
    
    return augmented_data

def visualize_preprocessing_example(original_img, original_mask, processed_img, processed_mask, title):
    """Show before/after preprocessing"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Original
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"Original Image\n{original_img.shape[:2]}")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title(f"Original Mask")
    axes[0, 1].axis('off')
    
    # Original colored mask
    color_mask_orig = np.zeros((*original_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask_orig[original_mask == class_id] = color
    axes[0, 2].imshow(color_mask_orig)
    axes[0, 2].set_title("Original Colored")
    axes[0, 2].axis('off')
    
    # Processed
    axes[1, 0].imshow(processed_img)
    axes[1, 0].set_title(f"Processed Image\n{processed_img.shape[:2]}")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(processed_mask, cmap='gray')
    axes[1, 1].set_title("Processed Mask")
    axes[1, 1].axis('off')
    
    # Processed colored mask
    color_mask_proc = np.zeros((*processed_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask_proc[processed_mask == class_id] = color
    axes[1, 2].imshow(color_mask_proc)
    axes[1, 2].set_title("Processed Colored")
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def process_region(region_name, split_name, target_size, resize_method='pad'):
    """Process a single region within a dataset split"""
    print(f"\n  📁 Processing {region_name} region...")
    
    # Input paths: old-dataset/Train/Rural/images_png
    region_path = os.path.join(BASE_DIR, split_name, region_name)
    image_dir = os.path.join(region_path, "images_png")
    mask_dir = os.path.join(region_path, "masks_png")
    
    # Output paths: processed-dataset/Train/Rural/images_png
    output_region_path = os.path.join(OUTPUT_DIR, split_name, region_name)
    output_image_dir = os.path.join(output_region_path, "images_png")
    output_mask_dir = os.path.join(output_region_path, "masks_png")
    
    # Create output directories
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"    ⚠️ Warning: Input directories not found for {region_name} in {split_name}")
        return 0
    
    # Get file lists
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    common_files = list(set(image_files) & set(mask_files))
    
    if not common_files:
        print(f"    ⚠️ No matching image-mask pairs found in {region_name}")
        return 0
    
    print(f"    📊 Found {len(common_files)} image-mask pairs")
    
    # Get augmentation settings for this split
    split_aug_settings = AUG_SETTINGS.get(split_name, {})
    enabled_augs = [k for k, v in split_aug_settings.items() if v]
    
    if enabled_augs:
        print(f"    🔄 Augmentations: {', '.join(enabled_augs)}")
    else:
        print(f"    🔄 Augmentations: None")
    
    # Process files
    total_processed = 0
    show_example = True
    
    for i, filename in enumerate(tqdm(common_files, desc=f"    {region_name}", leave=False)):
        try:
            # Load original files
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            original_image = cv2.imread(img_path)
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if original_image is None or original_mask is None:
                print(f"    ⚠️ Warning: Could not load {filename}")
                continue
            
            # Resize
            processed_image, processed_mask = resize_image_and_mask(
                original_image, original_mask, target_size, resize_method
            )
            
            # Convert image to RGB for consistency
            processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Show example for first image of first region in first split
            if show_example and i == 0 and split_name == DATASET_SPLITS[0] and region_name == REGIONS[0]:
                visualize_preprocessing_example(
                    original_image, original_mask, 
                    processed_image_rgb, processed_mask,
                    f"Preprocessing Example - {split_name}/{region_name}"
                )
                show_example = False
            
            # Create augmented versions
            filename_base = filename.replace('.png', '')
            augmented_data = create_augmented_versions(
                processed_image_rgb, processed_mask, filename_base, split_name
            )
            
            # Save all versions
            for img_data, mask_data, save_filename in augmented_data:
                # Save image (convert back to BGR for saving)
                img_save_path = os.path.join(output_image_dir, save_filename)
                img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_save_path, img_bgr)
                
                # Save mask
                mask_save_path = os.path.join(output_mask_dir, save_filename)
                cv2.imwrite(mask_save_path, mask_data)
                
                total_processed += 1
                
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {e}")
            continue
    
    print(f"    ✅ {region_name}: {len(common_files)} original → {total_processed} total files")
    return total_processed

def process_split(split_name, target_size, resize_method='pad'):
    """Process a single dataset split (Train/Val/Test)"""
    print(f"\n{'='*60}")
    print(f"PROCESSING SPLIT: {split_name}")
    print(f"{'='*60}")
    
    split_path = os.path.join(BASE_DIR, split_name)
    
    if not os.path.exists(split_path):
        print(f"⚠️ Warning: {split_name} directory not found at {split_path}")
        return 0
    
    # Process each region in this split
    total_files = 0
    for region in REGIONS:
        region_count = process_region(region, split_name, target_size, resize_method)
        total_files += region_count
    
    print(f"\n✅ {split_name} completed: {total_files} total files processed")
    return total_files

def main():
    print("🚀 MULTI-SPLIT DATASET PREPROCESSING STARTED")
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Splits to process: {', '.join(DATASET_SPLITS)}")
    print(f"Regions per split: {', '.join(REGIONS)}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Global augmentation: {'Enabled' if ENABLE_AUGMENTATION else 'Disabled'}")
    
    # Analyze image sizes across all splits
    print(f"\n🔍 ANALYZING IMAGE SIZES...")
    all_sizes = analyze_image_sizes(BASE_DIR)
    
    if all_sizes:
        print(f"  Total images sampled: {len(all_sizes)}")
        recommended_size = recommend_target_size(all_sizes)
        if TARGET_SIZE != recommended_size:
            print(f"\n⚠️  Current target size: {TARGET_SIZE}")
            print(f"⚠️  Recommended size: {recommended_size}")
            print("Consider updating TARGET_SIZE in the code if needed.")
    else:
        print("  No images found for analysis")
    
    # Show augmentation settings per split
    print(f"\n🔄 AUGMENTATION SETTINGS:")
    for split in DATASET_SPLITS:
        split_settings = AUG_SETTINGS.get(split, {})
        enabled_augs = [k for k, v in split_settings.items() if v]
        if enabled_augs:
            print(f"  {split}: {', '.join(enabled_augs)}")
        else:
            print(f"  {split}: No augmentation")
    
    print(f"\n📋 PREPROCESSING SETTINGS:")
    print(f"   Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"   Resize method: pad (maintains aspect ratio)")
    print(f"   Augmentation factor: {AUGMENTATION_FACTOR}x per image (where enabled)")
    
    input("\nPress Enter to continue with preprocessing...")
    
    # Process each split
    total_files_all = 0
    split_results = {}
    
    for split in DATASET_SPLITS:
        split_count = process_split(split, TARGET_SIZE, 'pad')
        split_results[split] = split_count
        total_files_all += split_count
    
    # Final summary
    print(f"\n{'='*60}")
    print(" ALL PREPROCESSING COMPLETED!")
    print(f"{'='*60}")
    
    for split, count in split_results.items():
        print(f"  {split}: {count} files")
    
    print(f"\nTotal processed files: {total_files_all}")
    print(f"Output structure:")
    for split in DATASET_SPLITS:
        print(f"   {OUTPUT_DIR}/Processed_{split}/")
        for region in REGIONS:
            print(f"     {region}/")
            print(f"       images_png/")
            print(f"       masks_png/")

if __name__ == "__main__":
    main()
