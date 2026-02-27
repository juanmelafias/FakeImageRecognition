"""
Dataset loading and preprocessing for CIFAKE
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import config

class CIFAKEDataset(Dataset):
    """Custom Dataset for CIFAKE (Real and AI-Generated Images)"""
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Path to dataset root (e.g., data/raw)
            transform (callable, optional): Transformations to apply
            split (str): 'train' or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []
        
        # Load image paths and labels
        self._load_dataset()
    
    def _load_dataset(self):
        """Load image paths and corresponding labels"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Dataset split not found: {split_dir}")
        
        # Load REAL images (label: 0)
        real_dir = os.path.join(split_dir, 'REAL')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)  # REAL = 0
        
        # Load FAKE images (label: 1)
        fake_dir = os.path.join(split_dir, 'FAKE')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)  # FAKE = 1
        
        print(f"\n{self.split.upper()} set loaded:")
        print(f"  REAL images: {self.labels.count(0)}")
        print(f"  FAKE images: {self.labels.count(1)}")
        print(f"  Total: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and label at index"""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(augment=True):
    """
    Get image transformations for training and validation
    
    Args:
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # Normalization values for pretrained models
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if augment:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=config.RANDOM_FLIP_PROB),
            transforms.RandomRotation(config.RANDOM_ROTATION),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ) if config.COLOR_JITTER else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            normalize
        ])
    else:
        # Training without augmentation
        train_transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize
        ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, val_transform

def get_data_loaders(batch_size=None, num_workers=None):
    """
    Create train, validation, and test data loaders
    
    Args:
        batch_size (int, optional): Batch size for loaders
        num_workers (int, optional): Number of worker processes
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=config.USE_AUGMENTATION)
    
    # Load datasets
    print("\n" + "=" * 60)
    print("Loading CIFAKE Dataset")
    print("=" * 60)
    
    # Training set
    train_dataset_full = CIFAKEDataset(
        root_dir=config.RAW_DATA_DIR,
        transform=train_transform,
        split='train'
    )
    
    # Split training set into train and validation
    train_size = int(config.TRAIN_SPLIT * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Apply validation transform to validation split
    val_dataset.dataset.transform = val_transform
    
    # Test set
    test_dataset = CIFAKEDataset(
        root_dir=config.RAW_DATA_DIR,
        transform=val_transform,
        split='test'
    )
    
    print(f"\nDataset splits:")
    print(f"  Training: {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")
    print(f"  Test: {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset loading
    print("Testing dataset loading...")
    try:
        train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
        
        # Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"\nSample batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: REAL={sum(labels==0).item()}, FAKE={sum(labels==1).item()}")
        print("\n✓ Dataset loading successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to run download_data.py first!")
