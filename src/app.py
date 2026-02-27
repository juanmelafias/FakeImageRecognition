"""
AI Image Detection - Complete Application
Dataset Explorer + AI Detector in one app
"""
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import config
from model import get_model
from dataset import get_transforms

st.set_page_config(
    page_title="AI Image Detection Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SHARED FUNCTIONS
# ============================================================================

@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        return None, "Model not found! Please train the model first."
    
    try:
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model = get_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, f"Model loaded (Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['val_acc']:.2f}%)"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

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

# ============================================================================
# PAGE 1: DATASET EXPLORER
# ============================================================================

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

def dataset_explorer_page():
    """Dataset Explorer Page"""
    st.title("🔍 CIFAKE Dataset Explorer")
    st.markdown("Explore the AI-Generated Image Detection training dataset")
    
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
    
    # Dataset Statistics
    train_real = len(dataset['train']['REAL'])
    train_fake = len(dataset['train']['FAKE'])
    test_real = len(dataset['test']['REAL'])
    test_fake = len(dataset['test']['FAKE'])
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Images", f"{total_images:,}")
    col2.metric("Training Set", f"{train_real + train_fake:,}")
    col3.metric("Test Set", f"{test_real + test_fake:,}")
    col4.metric("Image Size", "32×32")
    
    st.markdown("---")
    
    # Controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        split = st.selectbox("Dataset Split", ["train", "test"])
    with col2:
        class_filter = st.selectbox("Class Filter", ["All", "REAL", "FAKE"])
    with col3:
        num_images = st.slider("Number of images", 5, 50, 20)
    with col4:
        view_mode = st.radio("View Mode", ["Random", "Sequential"], horizontal=True)
    
    cols_per_row = st.slider("Images per row", 3, 8, 5)
    
    if st.button("🔄 Refresh / Load New Images", type="primary"):
        st.rerun()
    
    st.markdown("---")
    
    # Collect images to display
    images_to_show = []
    labels_to_show = []
    
    if class_filter == "All":
        classes = ["REAL", "FAKE"]
    else:
        classes = [class_filter]
    
    for class_name in classes:
        available_images = dataset[split][class_name]
        
        if view_mode == "Random":
            selected = random.sample(available_images, min(num_images // len(classes), len(available_images)))
        else:
            selected = available_images[:num_images // len(classes)]
        
        images_to_show.extend(selected)
        labels_to_show.extend([class_name] * len(selected))
    
    # Shuffle if random mode
    if view_mode == "Random":
        combined = list(zip(images_to_show, labels_to_show))
        random.shuffle(combined)
        images_to_show, labels_to_show = zip(*combined)
    
    # Display images
    st.subheader(f"📸 {split.upper()} Set - Displaying {len(images_to_show)} images")
    display_image_grid(images_to_show, labels_to_show, cols=cols_per_row)
    
    # Analysis section
    st.markdown("---")
    st.subheader("📈 Quick Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Class Distribution**")
        fig, ax = plt.subplots(figsize=(8, 5))
        
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
        st.markdown("**Sample Comparison**")
        col_a, col_b = st.columns(2)
        
        with col_a:
            if dataset[split]['REAL']:
                real_img = Image.open(random.choice(dataset[split]['REAL']))
                st.image(real_img, caption="REAL Image", use_container_width=True)
        
        with col_b:
            if dataset[split]['FAKE']:
                fake_img = Image.open(random.choice(dataset[split]['FAKE']))
                st.image(fake_img, caption="FAKE (AI-Generated)", use_container_width=True)

# ============================================================================
# PAGE 2: AI DETECTOR
# ============================================================================

def preprocess_image(image):
    """Preprocess uploaded image for inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    _, val_transform = get_transforms(augment=False)
    image_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)
    
    return image_tensor

def predict_image(model, image):
    """Predict if image is real or AI-generated"""
    image_tensor = preprocess_image(image)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    real_prob = probabilities[0, 0].item()
    fake_prob = probabilities[0, 1].item()
    
    return {
        'prediction': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'real_probability': real_prob,
        'fake_probability': fake_prob,
        'is_ai_generated': predicted_class == 1
    }

def ai_detector_page():
    """AI Detector Page"""
    st.title("🤖 AI-Generated Image Detector")
    st.markdown("Upload an image to determine if it's real or AI-generated")
    
    # Load model
    with st.spinner("Loading model..."):
        model, status_msg = load_trained_model()
    
    if model is None:
        st.error(f"⚠️ {status_msg}")
        st.info("Please train the model first by running: `python src/train.py`")
        return
    
    st.success(f"✓ {status_msg}")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'],
        help="Upload any image format. It will be automatically resized for analysis."
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            st.info(f"📁 **{uploaded_file.name}** | Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")
            
            with st.spinner("Analyzing image..."):
                result = predict_image(model, image)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📸 Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"Original: {image.size[0]}×{image.size[1]} → Analyzed: {config.IMAGE_SIZE}×{config.IMAGE_SIZE}")
            
            with col2:
                st.subheader("🔍 Analysis Results")
                
                if result['is_ai_generated']:
                    st.error(f"### 🤖 AI-GENERATED")
                    st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
                else:
                    st.success(f"### ✅ REAL IMAGE")
                    st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
                
                st.markdown("---")
                st.markdown("**Probability Breakdown:**")
                
                st.metric("REAL", f"{result['real_probability']*100:.2f}%")
                st.progress(result['real_probability'])
                
                st.metric("AI-GENERATED", f"{result['fake_probability']*100:.2f}%")
                st.progress(result['fake_probability'])
                
                st.markdown("---")
                
                if result['confidence'] > 0.95:
                    st.info("💡 Very high confidence")
                elif result['confidence'] > 0.85:
                    st.info("💡 High confidence")
                elif result['confidence'] > 0.70:
                    st.warning("💡 Moderate confidence")
                else:
                    st.warning("💡 Low confidence - results may be unreliable")
            
            # Detailed metrics
            st.markdown("---")
            st.subheader("📊 Detailed Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", result['prediction'])
            with col2:
                st.metric("Confidence Level", f"{result['confidence']*100:.1f}%")
            with col3:
                certainty = "High" if result['confidence'] > 0.9 else "Medium" if result['confidence'] > 0.75 else "Low"
                st.metric("Certainty", certainty)
            
            # Export results
            st.markdown("---")
            results_text = f"""AI Image Detection Results
==========================
File: {uploaded_file.name}
Original Size: {image.size[0]}×{image.size[1]}

PREDICTION: {result['prediction']}
Confidence: {result['confidence']*100:.2f}%

Probability Breakdown:
- REAL: {result['real_probability']*100:.2f}%
- AI-GENERATED: {result['fake_probability']*100:.2f}%
"""
            st.download_button(
                label="📥 Download Results",
                data=results_text,
                file_name=f"detection_{uploaded_file.name}.txt",
                mime="text/plain"
            )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
    
    else:
        st.info("👆 Upload an image to get started!")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Works Best With:**
            - Photos and images
            - Any resolution (auto-resized)
            - Common formats (JPG, PNG, etc.)
            """)
        
        with col2:
            st.markdown("""
            **⚠️ Limitations:**
            - Trained on 32×32 images
            - Best for CIFAR-10 style content
            - May not detect all AI generators
            """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    
    page = st.sidebar.radio(
        "Select Page:",
        ["🤖 AI Detector", "🔍 Dataset Explorer"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Sidebar info
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    **AI Image Detection Suite**
    
    A complete tool for detecting AI-generated images using deep learning.
    
    **Features:**
    - ResNet18 model
    - 96.8% test accuracy
    - Trained on CIFAKE dataset
    - Real-time predictions
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Info:**")
    st.sidebar.markdown("- Architecture: ResNet18")
    st.sidebar.markdown("- Parameters: 11M")
    st.sidebar.markdown("- Training: 80K images")
    st.sidebar.markdown("- Input: 32×32 RGB")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset:**")
    st.sidebar.markdown("- CIFAKE Dataset")
    st.sidebar.markdown("- Real: CIFAR-10")
    st.sidebar.markdown("- Fake: Stable Diffusion")
    st.sidebar.markdown("- Total: 120K images")
    
    # Route to pages
    if page == "🤖 AI Detector":
        ai_detector_page()
    else:
        dataset_explorer_page()

if __name__ == "__main__":
    main()
