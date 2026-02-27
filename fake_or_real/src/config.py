"""
Configuration file for AI-Generated Image Detection project
"""
import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Dataset
DATASET_NAME = 'birdy654/cifake-real-and-ai-generated-synthetic-images'
IMAGE_SIZE = 32  # CIFAKE images are 32x32
NUM_CLASSES = 2  # Binary classification: Real (0) or Fake (1)
CLASS_NAMES = ['REAL', 'FAKE']

# Training hyperparameters
BATCH_SIZE = 128  # Adjust based on available memory
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Data split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model
MODEL_NAME = 'resnet18'  # Options: 'resnet18', 'efficientnet_b0', 'mobilenet_v3'
PRETRAINED = True
DROPOUT_RATE = 0.5

# Training settings
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 
                      'cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # For data loading
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 10
SAVE_BEST_ONLY = True

# Data augmentation
USE_AUGMENTATION = True
RANDOM_FLIP_PROB = 0.5
RANDOM_ROTATION = 15
COLOR_JITTER = True

# Logging
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, 'runs')

# Inference
CONFIDENCE_THRESHOLD = 0.5

# Random seed for reproducibility
RANDOM_SEED = 42
