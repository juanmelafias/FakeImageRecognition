"""
Configuration file for Face Recognition project using LFW dataset
"""
import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset
DATASET_NAME = 'LFW'  # Labeled Faces in the Wild
# Alternative mirrors for LFW dataset
LFW_URLS = [
    'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
    'https://ndownloader.figshare.com/files/5976015',  # Alternative mirror
]
LFW_PAIRS_URL = 'http://vis-www.cs.umass.edu/lfw/pairs.txt'
LFW_PEOPLE_URL = 'http://vis-www.cs.umass.edu/lfw/lfw-names.txt'

# Image settings
IMAGE_SIZE = 160  # Standard for FaceNet
NUM_CHANNELS = 3  # RGB

# Model settings
EMBEDDING_DIM = 512  # Face embedding dimension
MODEL_BACKBONE = 'resnet50'  # Options: 'resnet18', 'resnet34', 'resnet50', 'inception_resnetv1'
PRETRAINED = True  # Use ImageNet pre-trained weights
FREEZE_BACKBONE_LAYERS = 30  # Number of layers to freeze initially (0 = train all)

# Training hyperparameters
BATCH_SIZE = 32  # Triplet batch size (adjust for M4 Pro memory)
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001  # Lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Triplet loss settings
MARGIN = 0.2  # Triplet loss margin
MINING_STRATEGY = 'semi-hard'  # 'hard', 'semi-hard', 'all'

# Data augmentation
USE_AUGMENTATION = True
RANDOM_FLIP_PROB = 0.5
RANDOM_ROTATION = 10
COLOR_JITTER = True

# Face verification thresholds
VERIFICATION_THRESHOLD = 0.6  # Euclidean distance threshold for same person
MIN_IMAGES_PER_PERSON = 10  # Minimum images to include person in training

# Training settings
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
NUM_WORKERS = 4  # DataLoader workers (adjust for M4 Pro)
PIN_MEMORY = True
SAVE_CHECKPOINT_EVERY = 5  # Save model every N epochs

# Device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Logging
LOG_INTERVAL = 20  # Log every N batches
USE_TENSORBOARD = True

# Random seed for reproducibility
SEED = 42

# Class names for verification
CLASS_NAMES = ['Different Person', 'Same Person']

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
