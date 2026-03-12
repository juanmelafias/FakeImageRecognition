"""
Face recognition model with pre-trained backbone and embedding layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config


class FaceEmbeddingModel(nn.Module):
    """Face recognition model that outputs embeddings"""
    
    def __init__(self, embedding_dim=512, backbone='resnet50', pretrained=True):
        """
        Args:
            embedding_dim (int): Dimension of face embeddings
            backbone (str): Backbone architecture
            pretrained (bool): Use ImageNet pre-trained weights
        """
        super(FaceEmbeddingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_out_features = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            backbone_out_features = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            backbone_out_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(backbone_out_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # L2 normalization for embeddings
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, H, W]
        
        Returns:
            torch.Tensor: L2-normalized embeddings [batch_size, embedding_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get embeddings
        embeddings = self.embedding(features)
        
        # L2 normalize
        embeddings = self.l2_norm(embeddings)
        
        return embeddings
    
    def freeze_backbone_layers(self, num_layers):
        """
        Freeze first N layers of backbone for fine-tuning
        
        Args:
            num_layers (int): Number of layers to freeze
        """
        if num_layers == 0:
            return
        
        layers = list(self.backbone.children())
        for i, layer in enumerate(layers[:num_layers]):
            for param in layer.parameters():
                param.requires_grad = False
        
        print(f"Froze first {num_layers} layers of {self.backbone_name}")
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
        print("Unfroze all layers")


class TripletLoss(nn.Module):
    """Triplet loss for face recognition"""
    
    def __init__(self, margin=0.2):
        """
        Args:
            margin (float): Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss
        
        Args:
            anchor (torch.Tensor): Anchor embeddings [batch_size, embedding_dim]
            positive (torch.Tensor): Positive embeddings [batch_size, embedding_dim]
            negative (torch.Tensor): Negative embeddings [batch_size, embedding_dim]
        
        Returns:
            torch.Tensor: Triplet loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        return losses.mean()


def get_model(embedding_dim=None, backbone=None, pretrained=None):
    """
    Create face recognition model
    
    Args:
        embedding_dim (int, optional): Embedding dimension
        backbone (str, optional): Backbone architecture
        pretrained (bool, optional): Use pre-trained weights
    
    Returns:
        FaceEmbeddingModel: Face recognition model
    """
    embedding_dim = embedding_dim or config.EMBEDDING_DIM
    backbone = backbone or config.MODEL_BACKBONE
    pretrained = pretrained if pretrained is not None else config.PRETRAINED
    
    print("\n" + "=" * 70)
    print(f"Creating Face Recognition Model")
    print("=" * 70)
    print(f"  Backbone: {backbone}")
    print(f"  Pre-trained: {pretrained}")
    print(f"  Embedding dimension: {embedding_dim}")
    
    model = FaceEmbeddingModel(
        embedding_dim=embedding_dim,
        backbone=backbone,
        pretrained=pretrained
    )
    
    # Freeze layers if specified
    if config.FREEZE_BACKBONE_LAYERS > 0:
        model.freeze_backbone_layers(config.FREEZE_BACKBONE_LAYERS)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 70 + "\n")
    
    return model


def compute_distance(embedding1, embedding2):
    """
    Compute Euclidean distance between embeddings
    
    Args:
        embedding1 (torch.Tensor): First embedding
        embedding2 (torch.Tensor): Second embedding
    
    Returns:
        torch.Tensor: Distance
    """
    return F.pairwise_distance(embedding1, embedding2, p=2)


def verify_faces(model, img1, img2, threshold=None):
    """
    Verify if two face images are from the same person
    
    Args:
        model (FaceEmbeddingModel): Face recognition model
        img1 (torch.Tensor): First image tensor
        img2 (torch.Tensor): Second image tensor
        threshold (float, optional): Distance threshold
    
    Returns:
        dict: Verification results
    """
    threshold = threshold or config.VERIFICATION_THRESHOLD
    
    model.eval()
    with torch.no_grad():
        # Get embeddings
        emb1 = model(img1)
        emb2 = model(img2)
        
        # Compute distance
        distance = compute_distance(emb1, emb2).item()
        
        # Verify
        is_same_person = distance < threshold
        confidence = 1 - (distance / threshold) if distance < threshold else 0
    
    return {
        'distance': distance,
        'threshold': threshold,
        'is_same_person': is_same_person,
        'confidence': confidence
    }


if __name__ == '__main__':
    # Test model creation
    model = get_model()
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output embedding shape: {output.shape}")
    print(f"Embedding L2 norms: {torch.norm(output, p=2, dim=1)}")
