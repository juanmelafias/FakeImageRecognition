"""
Dataset loading and preprocessing for LFW face recognition
"""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import config

class LFWDataset(Dataset):
    """LFW Dataset for face recognition training with triplet loss"""
    
    def __init__(self, root_dir, transform=None, min_images_per_person=10):
        """
        Args:
            root_dir (str): Path to LFW dataset root (e.g., data/raw/lfw)
            transform (callable, optional): Transformations to apply
            min_images_per_person (int): Minimum images per person to include
        """
        self.root_dir = root_dir
        self.transform = transform
        self.min_images = min_images_per_person
        
        # Load dataset structure
        self.person_to_images = {}
        self.all_people = []
        self._load_dataset()
        
    def _load_dataset(self):
        """Load image paths organized by person"""
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Dataset not found: {self.root_dir}")
        
        # Scan through people directories
        for person_name in os.listdir(self.root_dir):
            person_dir = os.path.join(self.root_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            # Get all images for this person
            images = [os.path.join(person_dir, f) for f in os.listdir(person_dir) 
                     if f.endswith('.jpg')]
            
            # Only include people with enough images for training
            if len(images) >= self.min_images:
                self.person_to_images[person_name] = images
                self.all_people.append(person_name)
        
        print(f"\nLoaded LFW dataset:")
        print(f"  Total people with ≥{self.min_images} images: {len(self.all_people)}")
        total_images = sum(len(imgs) for imgs in self.person_to_images.values())
        print(f"  Total images for training: {total_images}")
        print(f"  Average images per person: {total_images / len(self.all_people):.1f}")
    
    def __len__(self):
        """Return total number of images"""
        return sum(len(imgs) for imgs in self.person_to_images.values())
    
    def __getitem__(self, idx):
        """
        Get a triplet: (anchor, positive, negative)
        - anchor: random image of person A
        - positive: different image of person A
        - negative: random image of person B (B ≠ A)
        """
        # Select random person for anchor/positive
        anchor_person = random.choice(self.all_people)
        anchor_images = self.person_to_images[anchor_person]
        
        # Need at least 2 images to form anchor-positive pair
        while len(anchor_images) < 2:
            anchor_person = random.choice(self.all_people)
            anchor_images = self.person_to_images[anchor_person]
        
        # Sample anchor and positive from same person
        anchor_path, positive_path = random.sample(anchor_images, 2)
        
        # Sample negative from different person
        negative_person = random.choice([p for p in self.all_people if p != anchor_person])
        negative_path = random.choice(self.person_to_images[negative_person])
        
        # Load images
        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative, anchor_person


class LFWPairsDataset(Dataset):
    """LFW Pairs dataset for face verification evaluation"""
    
    def __init__(self, root_dir, pairs_file, transform=None):
        """
        Args:
            root_dir (str): Path to LFW dataset root
            pairs_file (str): Path to pairs.txt file
            transform (callable, optional): Transformations to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = []
        self.labels = []
        
        self._load_pairs(pairs_file)
    
    def _load_pairs(self, pairs_file):
        """Load pairs from pairs.txt"""
        if not os.path.exists(pairs_file):
            print(f"Warning: pairs.txt not found at {pairs_file}")
            return
        
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            parts = line.strip().split('\t')
            
            if len(parts) == 3:
                # Same person: name, image1_id, image2_id
                person = parts[0]
                img1_id = int(parts[1])
                img2_id = int(parts[2])
                
                img1_path = os.path.join(self.root_dir, person, f"{person}_{img1_id:04d}.jpg")
                img2_path = os.path.join(self.root_dir, person, f"{person}_{img2_id:04d}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(1)  # Same person
                    
            elif len(parts) == 4:
                # Different people: name1, image1_id, name2, image2_id
                person1 = parts[0]
                img1_id = int(parts[1])
                person2 = parts[2]
                img2_id = int(parts[3])
                
                img1_path = os.path.join(self.root_dir, person1, f"{person1}_{img1_id:04d}.jpg")
                img2_path = os.path.join(self.root_dir, person2, f"{person2}_{img2_id:04d}.jpg")
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    self.pairs.append((img1_path, img2_path))
                    self.labels.append(0)  # Different people
        
        print(f"\nLoaded {len(self.pairs)} pairs for evaluation")
        print(f"  Same person pairs: {sum(self.labels)}")
        print(f"  Different person pairs: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """Get image pair and label"""
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label


def get_transforms(augment=True):
    """
    Get image transformations for training and validation
    
    Args:
        augment (bool): Whether to apply data augmentation
    
    Returns:
        tuple: (train_transform, val_transform)
    """
    
    # Normalization values for ImageNet pre-trained models
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
    Create train and validation data loaders
    
    Args:
        batch_size (int, optional): Batch size for loaders
        num_workers (int, optional): Number of worker processes
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=config.USE_AUGMENTATION)
    
    # Load dataset
    print("\n" + "=" * 70)
    print("Loading LFW Dataset for Training")
    print("=" * 70)
    
    # Check for both possible LFW directory names
    lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw')
    if not os.path.exists(lfw_dir):
        lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw_funneled')
    
    full_dataset = LFWDataset(
        root_dir=lfw_dir,
        transform=train_transform,
        min_images_per_person=config.MIN_IMAGES_PER_PERSON
    )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(config.TRAIN_VAL_SPLIT * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Change val dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
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
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print("=" * 70)
    
    return train_loader, val_loader


def get_evaluation_loader(batch_size=None, num_workers=None):
    """
    Create evaluation data loader for LFW pairs
    
    Args:
        batch_size (int, optional): Batch size
        num_workers (int, optional): Number of workers
    
    Returns:
        DataLoader: Evaluation data loader
    """
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers or config.NUM_WORKERS
    
    _, val_transform = get_transforms(augment=False)
    
    # Check for both possible LFW directory names
    lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw')
    if not os.path.exists(lfw_dir):
        lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw_funneled')
    
    pairs_file = os.path.join(config.RAW_DATA_DIR, 'pairs.txt')
    
    eval_dataset = LFWPairsDataset(
        root_dir=lfw_dir,
        pairs_file=pairs_file,
        transform=val_transform
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY
    )
    
    return eval_loader
