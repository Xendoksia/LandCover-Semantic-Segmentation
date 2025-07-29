import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from config import Config

class UNet(nn.Module):
    """
    U-Net model using segmentation_models_pytorch
    Optimized for land cover segmentation
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER):
        super(UNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",       # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,              # model output channels (number of classes in your dataset)
            activation=None,                  # could be None for logits or 'softmax2d' for multiclass segmentation
        )
    
    def forward(self, x):
        return self.model(x)

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ model - good for detailed segmentation
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER):
        super(DeepLabV3Plus, self).__init__()
        
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)

class FPN(nn.Module):
    """
    Feature Pyramid Network - good balance of speed and accuracy
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER):
        super(FPN, self).__init__()
        
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights="imagenet", 
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)

class LinkNet(nn.Module):
    """
    LinkNet - lightweight and fast
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, encoder_name="resnet34"):
        super(LinkNet, self).__init__()
        
        self.model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    
    def forward(self, x):
        return self.model(x)

# Custom U-Net implementation (if you want more control)
class CustomUNet(nn.Module):
    """
    Custom U-Net implementation with residual connections
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, in_channels=3, features=64):
        super(CustomUNet, self).__init__()
        
        # Encoder (downsampling)
        self.encoder1 = self._make_encoder(in_channels, features)
        self.encoder2 = self._make_encoder(features, features*2)
        self.encoder3 = self._make_encoder(features*2, features*4)
        self.encoder4 = self._make_encoder(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = self._make_encoder(features*8, features*16)
        
        # Decoder (upsampling)
        self.decoder4 = self._make_decoder(features*16, features*8)
        self.decoder3 = self._make_decoder(features*8, features*4)
        self.decoder2 = self._make_decoder(features*4, features*2)
        self.decoder1 = self._make_decoder(features*2, features)
        
        # Final classifier
        self.final_conv = nn.Conv2d(features, num_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)
        
    def _make_encoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        bottleneck = self.dropout(bottleneck)
        
        # Decoder path with skip connections
        dec4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        output = self.final_conv(dec1)
        
        return output

def create_model(model_name=Config.MODEL_NAME):
    """
    Create model based on configuration
    """
    models = {
        'unet': UNet,
        'deeplabv3plus': DeepLabV3Plus,
        'fpn': FPN,
        'linknet': LinkNet,
        'custom_unet': CustomUNet
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    
    model = models[model_name]()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

def test_model():
    """Test model with dummy input"""
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [2, {Config.NUM_CLASSES}, {Config.IMG_SIZE}, {Config.IMG_SIZE}]")
    
    return model

if __name__ == "__main__":
    test_model()