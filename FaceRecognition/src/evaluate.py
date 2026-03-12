"""
Evaluation script for face recognition model on LFW pairs
"""
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config
from model import get_model, compute_distance
from dataset import get_evaluation_loader

def load_model(model_path):
    """Load trained model"""
    print(f"\nLoading model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    model = get_model(
        embedding_dim=checkpoint['config']['embedding_dim'],
        backbone=checkpoint['config']['backbone']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch'] + 1} epochs)")
    
    return model

def evaluate_model(model, eval_loader):
    """Evaluate model on LFW pairs"""
    print("\n" + "=" * 70)
    print("EVALUATING ON LFW PAIRS")
    print("=" * 70)
    
    all_distances = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (img1, img2, labels) in enumerate(eval_loader):
            # Move to device
            img1 = img1.to(config.DEVICE)
            img2 = img2.to(config.DEVICE)
            
            # Get embeddings
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Compute distances
            distances = compute_distance(emb1, emb2)
            
            # Store results
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(eval_loader)} batches")
    
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    return all_distances, all_labels

def find_best_threshold(distances, labels):
    """Find best threshold for face verification"""
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, -distances)  # Negative distances for "same person"
    
    # Find optimal threshold (maximize TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]
    best_threshold = -best_threshold  # Convert back to distance
    
    return best_threshold

def compute_metrics(distances, labels, threshold):
    """Compute evaluation metrics"""
    # Predictions: distance < threshold means same person (label 1)
    predictions = (distances < threshold).astype(int)
    
    # Accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Accept Rate (FAR) and False Reject Rate (FRR)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'far': far,
        'frr': frr,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

def plot_results(distances, labels, threshold, metrics, save_dir):
    """Plot evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Distance distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    same_person_distances = distances[labels == 1]
    diff_person_distances = distances[labels == 0]
    
    plt.hist(same_person_distances, bins=50, alpha=0.5, label='Same Person', color='green')
    plt.hist(diff_person_distances, bins=50, alpha=0.5, label='Different Person', color='red')
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    plt.subplot(1, 2, 2)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Different', 'Same'],
                yticklabels=['Different', 'Same'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    plt.tight_layout()
    dist_plot_path = os.path.join(save_dir, 'distance_distribution.png')
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {dist_plot_path}")
    plt.close()
    
    # 3. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = metrics['roc_auc']
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    roc_plot_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {roc_plot_path}")
    plt.close()

def print_results(metrics, threshold):
    """Print evaluation results"""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 Performance Metrics:")
    print(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1-Score:        {metrics['f1_score']:.4f}")
    print(f"  ROC AUC:         {metrics['roc_auc']:.4f}")
    
    print(f"\n🎯 Verification Metrics:")
    print(f"  Threshold:       {threshold:.4f}")
    print(f"  FAR (False Accept Rate):  {metrics['far']:.4f}")
    print(f"  FRR (False Reject Rate):  {metrics['frr']:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']} (Correctly identified same person)")
    print(f"  True Negatives:  {metrics['true_negatives']} (Correctly identified different person)")
    print(f"  False Positives: {metrics['false_positives']} (Incorrectly matched different people)")
    print(f"  False Negatives: {metrics['false_negatives']} (Incorrectly rejected same person)")
    
    print("\n" + "=" * 70)

def main():
    """Main evaluation function"""
    print("\n" + "=" * 70)
    print("FACE RECOGNITION EVALUATION - LFW PAIRS")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    model = load_model(model_path)
    
    # Load evaluation data
    eval_loader = get_evaluation_loader()
    
    if len(eval_loader) == 0:
        print("\n❌ No evaluation pairs found. Please ensure pairs.txt is downloaded.")
        return
    
    # Evaluate
    distances, labels = evaluate_model(model, eval_loader)
    
    # Find best threshold
    print("\n🔍 Finding optimal threshold...")
    best_threshold = find_best_threshold(distances, labels)
    print(f"✓ Optimal threshold: {best_threshold:.4f}")
    
    # Use config threshold or best threshold
    threshold = config.VERIFICATION_THRESHOLD
    print(f"Using threshold: {threshold:.4f}")
    
    # Compute metrics
    metrics = compute_metrics(distances, labels, threshold)
    
    # Print results
    print_results(metrics, threshold)
    
    # Plot results
    plot_results(distances, labels, threshold, metrics, config.RESULTS_DIR)
    
    print(f"\n📁 Results saved in: {config.RESULTS_DIR}")
    print("\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()
