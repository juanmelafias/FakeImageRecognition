"""
Interactive AI Image Detector - Upload and classify images
"""
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import config
from model import get_model
from dataset import get_transforms

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    """Load the trained model"""
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        return None, "Model not found! Please train the model first."
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        
        # Create and load model
        model = get_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, f"Model loaded (Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['val_acc']:.2f}%)"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def preprocess_image(image):
    """
    Preprocess uploaded image for inference
    Handles any size/format and converts to model input
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get validation transform (no augmentation)
    _, val_transform = get_transforms(augment=False)
    
    # Apply transform
    image_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)
    
    return image_tensor

def predict_image(model, image):
    """
    Predict if image is real or AI-generated
    """
    # Preprocess
    image_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Get probabilities
    real_prob = probabilities[0, 0].item()
    fake_prob = probabilities[0, 1].item()
    
    return {
        'prediction': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'real_probability': real_prob,
        'fake_probability': fake_prob,
        'is_ai_generated': predicted_class == 1
    }

def display_result(image, result):
    """Display the image and prediction result"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(image, use_container_width=True)
        
        # Show image info
        st.caption(f"Original size: {image.size[0]}×{image.size[1]} pixels")
        st.caption(f"Resized to: {config.IMAGE_SIZE}×{config.IMAGE_SIZE} for analysis")
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Main prediction with color coding
        if result['is_ai_generated']:
            st.error(f"### 🤖 AI-GENERATED")
            st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
        else:
            st.success(f"### ✅ REAL IMAGE")
            st.markdown(f"**Confidence:** {result['confidence']*100:.2f}%")
        
        st.markdown("---")
        
        # Probability bars
        st.markdown("**Probability Breakdown:**")
        
        # Real probability
        real_pct = result['real_probability'] * 100
        st.metric("REAL", f"{real_pct:.2f}%")
        st.progress(result['real_probability'])
        
        # Fake probability
        fake_pct = result['fake_probability'] * 100
        st.metric("AI-GENERATED", f"{fake_pct:.2f}%")
        st.progress(result['fake_probability'])
        
        # Interpretation
        st.markdown("---")
        st.markdown("**💡 Interpretation:**")
        
        if result['confidence'] > 0.95:
            st.info("Very high confidence - the model is quite certain.")
        elif result['confidence'] > 0.85:
            st.info("High confidence - the model is fairly confident.")
        elif result['confidence'] > 0.70:
            st.warning("Moderate confidence - some uncertainty in classification.")
        else:
            st.warning("Low confidence - the model is unsure. Results may be unreliable.")

def main():
    # Header
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
    
    # Sidebar info
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This tool uses a **ResNet18** model trained on the **CIFAKE dataset** to detect AI-generated images.
    
    **How it works:**
    1. Upload an image (any format/size)
    2. Image is automatically resized to 32×32
    3. Model analyzes and predicts
    4. Get results with confidence scores
    
    **Model Performance:**
    - Test Accuracy: ~96.8%
    - Balanced detection of REAL vs FAKE
    
    **Note:** Model trained on 32×32 images from CIFAR-10 and Stable Diffusion. 
    Performance may vary on different image types.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Info:**")
    st.sidebar.markdown("- Training: 80,000 images")
    st.sidebar.markdown("- Validation: 20,000 images")
    st.sidebar.markdown("- Test: 20,000 images")
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'],
        help="Upload any image format. It will be automatically resized for analysis."
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Show original image info
            st.info(f"📁 File: {uploaded_file.name} | Size: {image.size[0]}×{image.size[1]} | Format: {image.format}")
            
            # Predict
            with st.spinner("Analyzing image..."):
                result = predict_image(model, image)
            
            # Display results
            display_result(image, result)
            
            # Additional analysis
            st.markdown("---")
            st.subheader("📊 Detailed Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Prediction",
                    result['prediction'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Confidence Level",
                    f"{result['confidence']*100:.1f}%",
                    delta=f"{abs(result['confidence']-0.5)*200:.1f}% from 50%"
                )
            
            with col3:
                certainty = "High" if result['confidence'] > 0.9 else "Medium" if result['confidence'] > 0.75 else "Low"
                st.metric(
                    "Certainty",
                    certainty
                )
            
            # Download results
            st.markdown("---")
            if st.button("📥 Export Results as Text"):
                results_text = f"""
AI Image Detection Results
==========================
File: {uploaded_file.name}
Original Size: {image.size[0]}×{image.size[1]}

PREDICTION: {result['prediction']}
Confidence: {result['confidence']*100:.2f}%

Probability Breakdown:
- REAL: {result['real_probability']*100:.2f}%
- AI-GENERATED: {result['fake_probability']*100:.2f}%

Status: {'AI-Generated' if result['is_ai_generated'] else 'Real Image'}
                """
                st.download_button(
                    label="Download Results",
                    data=results_text,
                    file_name=f"detection_results_{uploaded_file.name}.txt",
                    mime="text/plain"
                )
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image.")
    
    else:
        # Show example/placeholder
        st.info("👆 Upload an image to get started!")
        
        # Show some example results (if available)
        st.markdown("---")
        st.subheader("💡 Tips for Best Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **✅ Works Best With:**
            - Photos and images
            - Clear, well-lit images
            - Any resolution (auto-resized)
            - Common formats (JPG, PNG)
            """)
        
        with col2:
            st.markdown("""
            **⚠️ Limitations:**
            - Trained on 32×32 low-res images
            - Best for CIFAR-10 style images
            - May not generalize to all AI generators
            - Confidence varies by image complexity
            """)

if __name__ == "__main__":
    main()
