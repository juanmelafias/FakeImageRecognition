"""
Download and prepare the LFW (Labeled Faces in the Wild) dataset
"""
import os
import urllib.request
import tarfile
import shutil
from tqdm import tqdm
import config

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    print(f"\nDownloading from {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_tar(tar_path, extract_path):
    """Extract tar.gz file"""
    print(f"\nExtracting {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted to {extract_path}")

def download_lfw_dataset():
    """Download LFW dataset"""
    print("\n" + "=" * 70)
    print("DOWNLOADING LFW (Labeled Faces in the Wild) DATASET")
    print("=" * 70)
    
    # Check if already downloaded
    lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw')
    if os.path.exists(lfw_dir) and len(os.listdir(lfw_dir)) > 0:
        print(f"\n✓ LFW dataset already exists at {lfw_dir}")
        print(f"  Found {len(os.listdir(lfw_dir))} people directories")
        return
    
    # Download LFW images
    lfw_tar_path = os.path.join(config.RAW_DATA_DIR, 'lfw.tgz')
    
    if not os.path.exists(lfw_tar_path):
        print("\n📥 Downloading LFW images (~173 MB)...")
        
        # Try multiple mirrors
        success = False
        for idx, url in enumerate(config.LFW_URLS, 1):
            try:
                print(f"\nAttempt {idx}/{len(config.LFW_URLS)}: {url}")
                download_url(url, lfw_tar_path)
                success = True
                break
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                if idx < len(config.LFW_URLS):
                    print(f"  Trying alternative mirror...")
                continue
        
        if not success:
            print(f"\n❌ All download attempts failed.")
            print("\n📥 MANUAL DOWNLOAD INSTRUCTIONS:")
            print("=" * 70)
            print("Option 1 - Direct Download:")
            print(f"  1. Visit: http://vis-www.cs.umass.edu/lfw/")
            print(f"  2. Download 'lfw.tgz' (All images aligned with deep funneling)")
            print(f"  3. Place file in: {config.RAW_DATA_DIR}")
            print(f"  4. Run this script again to extract")
            print("\nOption 2 - Alternative Source:")
            print("  1. Visit: https://www.kaggle.com/datasets/atulanandjha/lfwpeople")
            print("  2. Download and extract to: {config.RAW_DATA_DIR}/lfw/")
            print("=" * 70)
            return
    else:
        print(f"\n✓ LFW archive already downloaded: {lfw_tar_path}")
    
    # Extract dataset
    extract_tar(lfw_tar_path, config.RAW_DATA_DIR)
    
    # Download pairs.txt for verification tasks
    pairs_path = os.path.join(config.RAW_DATA_DIR, 'pairs.txt')
    if not os.path.exists(pairs_path):
        print("\n📥 Downloading LFW pairs.txt...")
        try:
            download_url(config.LFW_PAIRS_URL, pairs_path)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not download pairs.txt: {e}")
    
    # Download people names list
    people_path = os.path.join(config.RAW_DATA_DIR, 'lfw-names.txt')
    if not os.path.exists(people_path):
        print("\n📥 Downloading LFW people names...")
        try:
            download_url(config.LFW_PEOPLE_URL, people_path)
        except Exception as e:
            print(f"\n⚠️  Warning: Could not download lfw-names.txt: {e}")
    
    # Clean up tar file to save space
    if os.path.exists(lfw_tar_path):
        print(f"\n🗑️  Removing tar file to save space...")
        os.remove(lfw_tar_path)
    
    print("\n" + "=" * 70)
    print("✅ LFW DATASET DOWNLOAD COMPLETE")
    print("=" * 70)

def analyze_dataset():
    """Analyze the downloaded LFW dataset"""
    print("\n" + "=" * 70)
    print("ANALYZING LFW DATASET")
    print("=" * 70)
    
    # Check for both possible LFW directory names
    lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw')
    if not os.path.exists(lfw_dir):
        lfw_dir = os.path.join(config.RAW_DATA_DIR, 'lfw_funneled')
    
    if not os.path.exists(lfw_dir):
        print("\n❌ LFW dataset not found. Please download first.")
        return
    
    # Count people and images
    people = [d for d in os.listdir(lfw_dir) if os.path.isdir(os.path.join(lfw_dir, d))]
    total_images = 0
    people_with_multiple_images = 0
    image_counts = []
    
    for person in people:
        person_dir = os.path.join(lfw_dir, person)
        images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
        num_images = len(images)
        image_counts.append(num_images)
        total_images += num_images
        if num_images >= config.MIN_IMAGES_PER_PERSON:
            people_with_multiple_images += 1
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total people: {len(people)}")
    print(f"  Total images: {total_images}")
    print(f"  Average images per person: {total_images / len(people):.1f}")
    print(f"  People with ≥{config.MIN_IMAGES_PER_PERSON} images: {people_with_multiple_images}")
    print(f"  Max images for one person: {max(image_counts)}")
    print(f"  Min images for one person: {min(image_counts)}")
    
    # Show top people by image count
    people_image_count = [(person, len([f for f in os.listdir(os.path.join(lfw_dir, person)) 
                                        if f.endswith('.jpg')])) 
                          for person in people]
    people_image_count.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🔝 Top 10 people by image count:")
    for i, (person, count) in enumerate(people_image_count[:10], 1):
        print(f"  {i:2d}. {person:30s} - {count:3d} images")
    
    print("\n" + "=" * 70)

def main():
    """Main function"""
    print("\n🚀 Starting LFW Dataset Download and Setup")
    
    # Download dataset
    download_lfw_dataset()
    
    # Analyze dataset
    analyze_dataset()
    
    print("\n✅ Setup complete! Ready to train face recognition model.")
    print(f"\n📁 Dataset location: {os.path.join(config.RAW_DATA_DIR, 'lfw')}")
    print("\n🎯 Next steps:")
    print("  1. python FaceRecognition/src/train.py  # Train the model")
    print("  2. python FaceRecognition/src/evaluate.py  # Evaluate on LFW pairs")

if __name__ == '__main__':
    main()
