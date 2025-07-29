# 🌍 Land Cover Semantic Segmentation

In this project Loveda dataset is used: https://github.com/Junjue-Wang/LoveDA/tree/master/Semantic_Segmentation
A deep learning project for semantic segmentation of land cover imagery using PyTorch and state-of-the-art architectures.

## 📋 Overview

This project implements semantic segmentation for land cover classification with 8 different classes:

- 0-**No-data** (Black)
- 1-**Background** (Gray)
- 2-**Building** (Red)
- 3-**Road** (Yellow)
- 4-**Water** (Blue)
- 5-**Barren** (Brown)
- 6-**Forest** (Green)
- 7-**Agriculture** (Cyan)

## 🚀 Features

- **Multiple Model Architectures**: UNet, DeepLabV3+, FPN support
- **Multiple Encoders**: EfficientNet-B2/B3, ResNet34/50, MobileNetV3
- **Advanced Loss Functions**: Combined Dice + Cross-Entropy + Focal Loss
- **Data Processing**: Automated preprocessing with augmentation
- **Interactive GUI**: Real-time testing interface
- **Comprehensive Analysis**: Model comparison and performance tracking

## 🏗️ Project Structure

```
seg/
├── config.py              # Configuration settings
├── model.py               # Model architectures (UNet, DeepLabV3+, FPN)
├── dataset.py             # Data loading and preprocessing
├── train.py               # Main training script
├── utils.py               # Loss functions, metrics, visualization
├── preprocess.py          # Data preprocessing and augmentation
├── segmentation_gui.py    # Interactive testing GUI
├── dataset_analysis.py    # Dataset statistics and visualization
├── analyze_models.py      # Model checkpoint analysis
├── checkpoints/           # Saved models
├── results/               # Training outputs and visualizations
└── processed-dataset/     # Processed training data
    ├── Train/
    ├── Val/
    └── Test/
```

## 🔧 Configuration

Edit `config.py` to customize training parameters:

```python
class Config:
    # Model settings
    MODEL_NAME = "unet"              # unet, deeplabv3plus, fpn
    ENCODER = "efficientnet-b2"      # efficientnet-b2/b3, resnet34/50

    # Training parameters
    BATCH_SIZE = 8
    EPOCHS = 15
    LEARNING_RATE = 8e-5
    IMAGE_HEIGHT = 384
    IMAGE_WIDTH = 384

    # Loss function
    LOSS_TYPE = "combined"           # ce, dice, focal, combined
    DICE_WEIGHT = 0.6
    CE_WEIGHT = 0.4
```

## 🚀 Quick Start

### 1. Data Preprocessing

```bash
python preprocess.py
```

- Resizes images to target resolution
- Applies augmentation for training data
- Creates train/val/test splits

### 2. Training

```bash
python train.py
```

- Trains the model with specified configuration
- Saves best model checkpoints
- Generates training visualizations

### 3. Interactive Testing

```bash
python segmentation_gui.py
```

- Load trained models
- Test on custom images
- Real-time segmentation visualization

### 4. Model Analysis

```bash
python analyze_models.py
```

- Compare different model checkpoints
- Performance metrics analysis

## 📊 Model Performance

Current best configuration:

- **Architecture**: UNet + EfficientNet-B2
- **Parameters**: ~10M (optimized for speed)
- **Input Size**: 384x384
- **Training Speed**: ~4 it/s on RTX GPU

### Performance Optimizations

- **Speed Optimized**: 384px resolution, UNet architecture
- **Memory Efficient**: Gradient accumulation, cache clearing
- **Stable Training**: Conservative learning rates, early stopping

## 🎨 Visualization Features

- **Training History**: Loss and metrics plots
- **Confusion Matrix**: Per-class performance analysis
- **Prediction Visualization**: Side-by-side comparisons
- **Class Distribution**: Dataset statistics

## 📈 Training Features

- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rates
- **Mixed Precision**: Faster training on modern GPUs
- **Gradient Clipping**: Stable training
- **Class Weighting**: Handles imbalanced datasets

## 🔍 Dataset

Expected directory structure:

```
processed-dataset/
├── Train/
│   ├── Urban/
│   │   ├── images_png/
│   │   └── masks_png/
│   └── Rural/
│       ├── images_png/
│       └── masks_png/
├── Val/
└── Test/
```

### Loss Functions

- **Cross-Entropy**: Standard classification loss
- **Dice Loss**: IoU optimization for segmentation
- **Focal Loss**: Handles class imbalance
- **Combined Loss**: Weighted combination for best results

## 📊 Model Comparison

| Model      | Encoder         | Params | Speed  | IoU  | Accuracy |
| ---------- | --------------- | ------ | ------ | ---- | -------- |
| UNet       | EfficientNet-B2 | 10M    | 4 it/s | 0.60 | 0.79      |
| UNet       | ResNet34        | 24M    | 3 it/s | 0.60 | 0.84     |
| DeepLabV3+ | EfficientNet-B3 | 32M    | 2 it/s | TBD  | TBD      |

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions and support, please open an issue in the repository.

---

**Note**: This project is optimized for land cover segmentation but can be adapted for other semantic segmentation tasks by modifying the class labels and configuration.
