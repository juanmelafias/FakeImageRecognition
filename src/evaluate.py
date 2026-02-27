"""
Evaluation script for AI-Generated Image Detection
"""
import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import config
from model import get_model
from dataset import get_data_loaders

def evaluate_model(model, test_loader, device):
    """
    Comprehensive model evaluation
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of FAKE class
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=config.CLASS_NAMES,
        output_dict=True
    )
    
    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    return results

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    plt.show()

def plot_roc_curve(labels, probs, roc_auc, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {save_path}")
    plt.show()

def print_evaluation_results(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.2f}%")
    print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
    
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    
    report = results['classification_report']
    for class_name in config.CLASS_NAMES:
        metrics = report[class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")
    
    print("\n" + "-"*60)
    print("Confusion Matrix:")
    print("-"*60)
    cm = results['confusion_matrix']
    print("\n             Predicted")
    print("           ", "  ".join(f"{name:>8}" for name in config.CLASS_NAMES))
    print("Actual")
    for i, name in enumerate(config.CLASS_NAMES):
        print(f"{name:>8}   ", "  ".join(f"{cm[i,j]:>8d}" for j in range(len(config.CLASS_NAMES))))
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print("\n" + "-"*60)
    print("Additional Metrics:")
    print("-"*60)
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print(f"\nSensitivity (True Positive Rate):  {sensitivity:.4f}")
    print(f"Specificity (True Negative Rate):  {specificity:.4f}")

def main(model_path=None):
    """Main evaluation function"""
    print("\n" + "="*60)
    print("AI-Generated Image Detection - Evaluation")
    print("="*60)
    
    # Default to best model
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\nLoading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model = get_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']+1} epochs)")
    print(f"  Validation accuracy at checkpoint: {checkpoint['val_acc']:.2f}%")
    
    # Get test data loader
    _, _, test_loader = get_data_loaders()
    
    # Evaluate
    results = evaluate_model(model, test_loader, config.DEVICE)
    
    # Print results
    print_evaluation_results(results)
    
    # Create results directory
    results_dir = os.path.join(config.PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], save_path=cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(results_dir, 'roc_curve.png')
    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        results['roc_auc'],
        save_path=roc_path
    )
    
    print(f"\n{'='*60}")
    print("Evaluation completed successfully!")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate AI-Generated Image Detection model')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to model checkpoint (default: models/best_model.pth)')
    
    args = parser.parse_args()
    
    try:
        results = main(model_path=args.model_path)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        raise
