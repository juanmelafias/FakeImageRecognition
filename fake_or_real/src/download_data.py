"""
Script to download the CIFAKE dataset from Kaggle
"""
import os
import zipfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()

def download_cifake():
    """Download and extract CIFAKE dataset using Kaggle API"""
    
    print("=" * 60)
    print("CIFAKE Dataset Downloader")
    print("=" * 60)
    
    # Set up Kaggle credentials from environment variables
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if not kaggle_username or not kaggle_key:
        print("\n⚠️  Kaggle API credentials not found in .env file!")
        print("\nPlease update the .env file with:")
        print("KAGGLE_USERNAME=your_kaggle_username")
        print("KAGGLE_KEY=your_kaggle_api_key")
        print("\nYou can find these at: https://www.kaggle.com/settings")
        return
    
    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    print(f"\n✓ Kaggle credentials loaded")
    print(f"  Username: {kaggle_username}")
    
    # Create data directories
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    dataset_path = os.path.join(config.RAW_DATA_DIR, 'cifake')
    
    # Check if dataset already exists
    if os.path.exists(dataset_path) and len(os.listdir(dataset_path)) > 0:
        print(f"\n✓ Dataset already exists at: {dataset_path}")
        print("\nIf you want to re-download, delete the folder and run again.")
        return
    
    print(f"\n📥 Downloading CIFAKE dataset...")
    print(f"Destination: {config.RAW_DATA_DIR}\n")
    
    try:
        # Download using Kaggle API
        import kaggle
        kaggle.api.dataset_download_files(
            config.DATASET_NAME,
            path=config.RAW_DATA_DIR,
            unzip=True
        )
        
        print("\n✓ Download complete!")
        
        # Check what was downloaded
        downloaded_files = os.listdir(config.RAW_DATA_DIR)
        print(f"\nDownloaded files: {downloaded_files}")
        
        # Organize the dataset
        organize_dataset()
        
        print("\n" + "=" * 60)
        print("✓ Dataset ready for training!")
        print("=" * 60)
        print(f"\nDataset location: {config.RAW_DATA_DIR}")
        print(f"Total size: {get_folder_size(config.RAW_DATA_DIR):.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Kaggle API installed: pip install kaggle")
        print("2. Verify your kaggle.json is in the correct location")
        print("3. Check your internet connection")
        print("4. Ensure you've accepted the dataset terms on Kaggle website")

def organize_dataset():
    """Organize downloaded dataset into proper structure"""
    print("\n📁 Organizing dataset structure...")
    
    # CIFAKE typically comes with train/ and test/ folders
    # Each containing REAL/ and FAKE/ subfolders
    raw_dir = config.RAW_DATA_DIR
    
    # Check structure
    expected_folders = ['train', 'test']
    for folder in expected_folders:
        folder_path = os.path.join(raw_dir, folder)
        if os.path.exists(folder_path):
            real_path = os.path.join(folder_path, 'REAL')
            fake_path = os.path.join(folder_path, 'FAKE')
            
            if os.path.exists(real_path):
                real_count = len([f for f in os.listdir(real_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {folder}/REAL: {real_count} images")
            
            if os.path.exists(fake_path):
                fake_count = len([f for f in os.listdir(fake_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {folder}/FAKE: {fake_count} images")

def get_folder_size(folder_path):
    """Calculate folder size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

if __name__ == "__main__":
    download_cifake()
