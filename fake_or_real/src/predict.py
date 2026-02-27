"""
Inference script for predicting if an image is AI-generated
"""
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import config
from model import get_model
from dataset import get_transforms

def load_model(model_path=None):
    """Load trained model from checkpoint"""
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    # Create model
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

def predict_image(image_path, model, show_result=True):
    """
    Predict if an image is AI-generated
    
    Args:
        image_path (str): Path to image file
        model: Trained PyTorch model
        show_result (bool): Whether to display the result
    
    Returns:
        dict: Prediction results
    """
    # Load and preprocess image
    _, val_transform = get_transforms(augment=False)
    
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    # Transform image
    input_tensor = val_transform(image).unsqueeze(0).to(config.DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # Get class probabilities
    real_prob = probabilities[0, 0].item()
    fake_prob = probabilities[0, 1].item()
    
    # Create result dictionary
    result = {
        'predicted_class': config.CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'real_probability': real_prob,
        'fake_probability': fake_prob,
        'is_ai_generated': predicted_class == 1
    }
    
    # Display result
    if show_result:
        visualize_prediction(image, result)
    
    return result

def visualize_prediction(image, result):
    """Visualize prediction result"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display prediction
    classes = config.CLASS_NAMES
    probs = [result['real_probability'], result['fake_probability']]
    colors = ['green' if result['predicted_class'] == 'REAL' else 'lightgray',
              'red' if result['predicted_class'] == 'FAKE' else 'lightgray']
    
    bars = ax2.barh(classes, probs, color=colors)
    ax2.set_xlim([0, 1])
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    # Add prediction text
    prediction_text = f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']*100:.1f}%"
    fig.text(0.5, 0.02, prediction_text, ha='center', fontsize=13, 
             fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()

def predict_batch(image_folder, model, output_csv=None):
    """
    Predict multiple images from a folder
    
    Args:
        image_folder (str): Path to folder containing images
        model: Trained PyTorch model
        output_csv (str, optional): Path to save results as CSV
    
    Returns:
        list: List of prediction results
    """
    import glob
    import pandas as pd
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
        image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return []
    
    print(f"\nProcessing {len(image_files)} images...")
    
    results = []
    for image_path in image_files:
        try:
            result = predict_image(image_path, model, show_result=False)
            result['image_path'] = image_path
            result['image_name'] = os.path.basename(image_path)
            results.append(result)
            
            # Print result
            icon = "🤖" if result['is_ai_generated'] else "✓"
            print(f"{icon} {result['image_name']}: {result['predicted_class']} ({result['confidence']*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    # Save to CSV if requested
    if output_csv and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to: {output_csv}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    total = len(results)
    fake_count = sum(1 for r in results if r['is_ai_generated'])
    real_count = total - fake_count
    print(f"Total images processed: {total}")
    print(f"Real images: {real_count} ({100*real_count/total:.1f}%)")
    print(f"AI-generated images: {fake_count} ({100*fake_count/total:.1f}%)")
    
    return results

def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict if an image is AI-generated')
    parser.add_argument('--image_path', type=str, help='Path to single image')
    parser.add_argument('--image_folder', type=str, help='Path to folder with multiple images')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model checkpoint (default: models/best_model.pth)')
    parser.add_argument('--output_csv', type=str, help='Save batch results to CSV')
    parser.add_argument('--no_display', action='store_true', help='Do not display visualization')
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_folder:
        parser.error("Please provide either --image_path or --image_folder")
    
    print("\n" + "="*60)
    print("AI-Generated Image Detection - Inference")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.model_path)
    print("✓ Model loaded successfully")
    
    # Single image prediction
    if args.image_path:
        print(f"\nAnalyzing: {args.image_path}")
        result = predict_image(args.image_path, model, show_result=not args.no_display)
        
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"{'='*60}")
        print(f"Prediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"\nProbabilities:")
        print(f"  REAL: {result['real_probability']*100:.2f}%")
        print(f"  FAKE: {result['fake_probability']*100:.2f}%")
    
    # Batch prediction
    if args.image_folder:
        results = predict_batch(args.image_folder, model, args.output_csv)

if __name__ == "__main__":
    main()
