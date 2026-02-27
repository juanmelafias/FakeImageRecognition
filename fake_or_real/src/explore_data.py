"""
Dataset Explorer - Interactive viewer for CIFAKE dataset
"""
import streamlit as st
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import config

st.set_page_config(
    page_title="CIFAKE Dataset Explorer",
    page_icon="🔍",
    layout="wide"
)

@st.cache_data
def load_dataset_info():
    """Load dataset file paths and labels"""
    dataset = {'train': {'REAL': [], 'FAKE': []}, 'test': {'REAL': [], 'FAKE': []}}
    
    for split in ['train', 'test']:
        split_dir = os.path.join(config.RAW_DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in ['REAL', 'FAKE']:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                dataset[split][class_name] = images
    
    return dataset

def display_image_grid(images, labels, cols=5):
    """Display images in a grid"""
    rows = (len(images) + cols - 1) // cols
    
    for row in range(rows):
        cols_obj = st.columns(cols)
        for col_idx in range(cols):
            idx = row * cols + col_idx
            if idx < len(images):
                with cols_obj[col_idx]:
                    img = Image.open(images[idx])
                    st.image(img, caption=labels[idx], use_container_width=True)

def main():
    st.title("🔍 CIFAKE Dataset Explorer")
    st.markdown("Explore the AI-Generated Image Detection dataset")
    
    # Check if dataset exists
    if not os.path.exists(config.RAW_DATA_DIR):
        st.error("⚠️ Dataset not found! Please run `python src/download_data.py` first.")
        return
    
    # Load dataset
    with st.spinner("Loading dataset..."):
        dataset = load_dataset_info()
    
    # Check if data loaded
    total_images = sum(len(imgs) for split in dataset.values() for imgs in split.values())
    if total_images == 0:
        st.error("⚠️ No images found in the dataset directory!")
        return
    
    # Sidebar - Dataset Statistics
    st.sidebar.header("📊 Dataset Statistics")
    
    train_real = len(dataset['train']['REAL'])
    train_fake = len(dataset['train']['FAKE'])
    test_real = len(dataset['test']['REAL'])
    test_fake = len(dataset['test']['FAKE'])
    
    st.sidebar.metric("Total Images", f"{total_images:,}")
    
    st.sidebar.subheader("Training Set")
    st.sidebar.write(f"🟢 REAL: {train_real:,}")
    st.sidebar.write(f"🔴 FAKE: {train_fake:,}")
    st.sidebar.write(f"**Total: {train_real + train_fake:,}**")
    
    st.sidebar.subheader("Test Set")
    st.sidebar.write(f"🟢 REAL: {test_real:,}")
    st.sidebar.write(f"🔴 FAKE: {test_fake:,}")
    st.sidebar.write(f"**Total: {test_real + test_fake:,}**")
    
    # Sidebar - Controls
    st.sidebar.header("🎛️ Controls")
    
    split = st.sidebar.selectbox("Dataset Split", ["train", "test"])
    class_filter = st.sidebar.selectbox("Class Filter", ["All", "REAL", "FAKE"])
    
    num_images = st.sidebar.slider("Number of images to display", 5, 50, 20)
    view_mode = st.sidebar.radio("View Mode", ["Random Sample", "Sequential"])
    
    if st.sidebar.button("🔄 Refresh / Load New Images"):
        st.rerun()
    
    # Main content
    st.header(f"📸 Viewing: {split.upper()} Set")
    
    # Collect images to display
    images_to_show = []
    labels_to_show = []
    
    if class_filter == "All":
        classes = ["REAL", "FAKE"]
    else:
        classes = [class_filter]
    
    for class_name in classes:
        available_images = dataset[split][class_name]
        
        if view_mode == "Random Sample":
            selected = random.sample(available_images, min(num_images // len(classes), len(available_images)))
        else:
            selected = available_images[:num_images // len(classes)]
        
        images_to_show.extend(selected)
        labels_to_show.extend([class_name] * len(selected))
    
    # Shuffle if random mode
    if view_mode == "Random Sample":
        combined = list(zip(images_to_show, labels_to_show))
        random.shuffle(combined)
        images_to_show, labels_to_show = zip(*combined)
    
    # Display images
    st.write(f"Displaying **{len(images_to_show)}** images")
    
    cols_per_row = st.sidebar.slider("Images per row", 3, 8, 5)
    display_image_grid(images_to_show, labels_to_show, cols=cols_per_row)
    
    # Additional analysis section
    st.header("📈 Quick Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if split == "train":
            counts = [train_real, train_fake]
        else:
            counts = [test_real, test_fake]
        
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(['REAL', 'FAKE'], counts, color=colors, alpha=0.7)
        ax.set_ylabel('Number of Images')
        ax.set_title(f'{split.upper()} Set Class Distribution')
        ax.grid(axis='y', alpha=0.3)
        
        for i, count in enumerate(counts):
            ax.text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Sample Comparison")
        st.write("**Left: REAL** | **Right: FAKE**")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            if dataset[split]['REAL']:
                real_img = Image.open(random.choice(dataset[split]['REAL']))
                st.image(real_img, caption="REAL Image", use_container_width=True)
        
        with col_b:
            if dataset[split]['FAKE']:
                fake_img = Image.open(random.choice(dataset[split]['FAKE']))
                st.image(fake_img, caption="FAKE (AI-Generated)", use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**CIFAKE Dataset**: Real images from CIFAR-10 vs AI-generated images from Stable Diffusion")

if __name__ == "__main__":
    main()
