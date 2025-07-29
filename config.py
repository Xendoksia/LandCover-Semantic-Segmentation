# config.py

import os

class Config:
    # Dataset paths
    DATASET_PATH = r"---------------------------------------------"  # Base dataset folder
    BASE_DIR = r"---------------------------------------------"# Training data folder
    URBAN_DIR = os.path.join(BASE_DIR, "Urban")
    RURAL_DIR = os.path.join(BASE_DIR, "Rural")
    
    IMAGE_FOLDER = "images_png"       # Image subfolder name
    MASK_FOLDER = "masks_png"         # Mask subfolder name

    # Training parameters - SPEED OPTIMIZED
    BATCH_SIZE = 8  # Increase batch size for speed
    NUM_WORKERS = 6  # More workers for faster data loading
    IMAGE_HEIGHT = 384  # Smaller images for speed
    IMAGE_WIDTH = 384
    NUM_CLASSES = 8
    EPOCHS = 15  # Fewer epochs for faster testing
    LEARNING_RATE = 8e-5  # Slightly higher LR to compensate
    DEVICE = "cuda"
    
        # Augmentation settings
    AUGMENTATION = False
    MODEL_NAME = "unet"  # Faster than DeepLabV3Plus
    ENCODER = "efficientnet-b2"  # Faster than B3 but still good
    IMG_SIZE = 384  # Match the new resolution
    
        # Loss function settings - ADVANCED COMBINATION
    LOSS_TYPE = "combined"  # Use combined loss for better performance
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    DICE_WEIGHT = 0.6  # Higher weight for IoU optimization
    CE_WEIGHT = 0.4
    
    # Model ve kayıt yolları
    MODEL_SAVE_PATH = "./checkpoints/best_model.pth"
    RESULTS_DIR = "./results"
    LOG_DIR = "./logs"
    
    # Visualize.py veya utils.py için kullanılabilecek renkler (RGB)
    CLASS_LABELS = {
        0: ("No-data", (0, 0, 0)),
        1: ("Background", (255, 255, 255)),
        2: ("Building", (255, 0, 0)),
        3: ("Road", (255, 255, 0)),
        4: ("Water", (0, 0, 255)),
        5: ("Barren", (150, 75, 0)),
        6: ("Forest", (0, 255, 0)),
        7: ("Agriculture", (0, 255, 255)),
    }

    # Early stopping ve optimizer settings - FAST TRAINING
    EARLY_STOPPING = True
    PATIENCE = 5  # Earlier stopping for speed
    
    # Advanced training settings - CONSERVATIVE FOR STABILITY
    OPTIMIZER = "adamw"  # Keep AdamW
    WEIGHT_DECAY = 1e-4  # Conservative weight decay
    GRADIENT_CLIPPING = 1.0  # Conservative gradient clipping
    SCHEDULER = "reduce_on_plateau"  # Simple scheduler
    MIN_LR = 1e-6  # Conservative minimum learning rate

    # Model checkpoint ayarı
    SAVE_BEST_ONLY = True
    


config = Config()
