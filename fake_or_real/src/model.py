"""
Model architectures for AI-Generated Image Detection
"""
import torch
import torch.nn as nn
from torchvision import models
import config

class AIDetectorCNN(nn.Module):
    """Simple CNN for binary classification"""
    
    def __init__(self, num_classes=2):
        super(AIDetectorCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AIDetectorResNet(nn.Module):
    """ResNet-based classifier for AI detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(AIDetectorResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Get number of features in the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

class AIDetectorEfficientNet(nn.Module):
    """EfficientNet-based classifier for AI detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(AIDetectorEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
        # Get number of features in the last layer
        num_features = self.efficientnet.classifier[1].in_features
        
        # Replace the classifier
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)

class AIDetectorMobileNet(nn.Module):
    """MobileNetV3-based classifier for AI detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(AIDetectorMobileNet, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        self.mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Get number of features in the last layer
        num_features = self.mobilenet.classifier[3].in_features
        
        # Replace the classifier
        self.mobilenet.classifier[3] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.mobilenet(x)

def get_model(model_name=None, num_classes=None, pretrained=None):
    """
    Factory function to create model based on configuration
    
    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Model instance
    """
    model_name = model_name or config.MODEL_NAME
    num_classes = num_classes or config.NUM_CLASSES
    pretrained = pretrained if pretrained is not None else config.PRETRAINED
    
    print(f"\n{'='*60}")
    print(f"Building Model: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Pretrained: {pretrained}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {config.DEVICE}")
    
    if model_name == 'resnet18':
        model = AIDetectorResNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet_b0':
        model = AIDetectorEfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet_v3':
        model = AIDetectorMobileNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'simple_cnn':
        model = AIDetectorCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Move model to device
    model = model.to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    return model

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test ResNet
    model = get_model('resnet18')
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 32, 32).to(config.DEVICE)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model creation successful!")
