"""
Training script for face recognition model on LFW dataset
"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import config
from model import get_model, TripletLoss
from dataset import get_data_loaders

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_pos_dist = 0.0
    running_neg_dist = 0.0
    
    start_time = time.time()
    
    for batch_idx, (anchor, positive, negative, _) in enumerate(train_loader):
        # Move to device
        anchor = anchor.to(config.DEVICE)
        positive = positive.to(config.DEVICE)
        negative = negative.to(config.DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        # Compute loss
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Compute distances for monitoring
        with torch.no_grad():
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb).mean()
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb).mean()
            running_pos_dist += pos_dist.item()
            running_neg_dist += neg_dist.item()
        
        # Log progress
        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_pos_dist = running_pos_dist / (batch_idx + 1)
            avg_neg_dist = running_neg_dist / (batch_idx + 1)
            
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Pos Dist: {avg_pos_dist:.3f} | "
                  f"Neg Dist: {avg_neg_dist:.3f}")
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_pos_dist = running_pos_dist / len(train_loader)
    epoch_neg_dist = running_neg_dist / len(train_loader)
    epoch_time = time.time() - start_time
    
    # Log to tensorboard
    if writer:
        global_step = epoch * len(train_loader)
        writer.add_scalar('Loss/train', epoch_loss, global_step)
        writer.add_scalar('Distance/positive', epoch_pos_dist, global_step)
        writer.add_scalar('Distance/negative', epoch_neg_dist, global_step)
    
    print(f"\n  Train Loss: {epoch_loss:.4f} | "
          f"Pos Dist: {epoch_pos_dist:.3f} | "
          f"Neg Dist: {epoch_neg_dist:.3f} | "
          f"Time: {epoch_time:.1f}s")
    
    return epoch_loss

def validate(model, val_loader, criterion, epoch, writer):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_pos_dist = 0.0
    running_neg_dist = 0.0
    
    with torch.no_grad():
        for anchor, positive, negative, _ in val_loader:
            # Move to device
            anchor = anchor.to(config.DEVICE)
            positive = positive.to(config.DEVICE)
            negative = negative.to(config.DEVICE)
            
            # Forward pass
            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)
            
            # Compute loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            running_loss += loss.item()
            
            # Compute distances
            pos_dist = torch.nn.functional.pairwise_distance(anchor_emb, positive_emb).mean()
            neg_dist = torch.nn.functional.pairwise_distance(anchor_emb, negative_emb).mean()
            running_pos_dist += pos_dist.item()
            running_neg_dist += neg_dist.item()
    
    # Validation statistics
    val_loss = running_loss / len(val_loader)
    val_pos_dist = running_pos_dist / len(val_loader)
    val_neg_dist = running_neg_dist / len(val_loader)
    
    # Log to tensorboard
    if writer:
        global_step = epoch * len(val_loader)
        writer.add_scalar('Loss/val', val_loss, global_step)
        writer.add_scalar('Distance_Val/positive', val_pos_dist, global_step)
        writer.add_scalar('Distance_Val/negative', val_neg_dist, global_step)
    
    print(f"  Val Loss: {val_loss:.4f} | "
          f"Pos Dist: {val_pos_dist:.3f} | "
          f"Neg Dist: {val_neg_dist:.3f}\n")
    
    return val_loss

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'embedding_dim': config.EMBEDDING_DIM,
            'backbone': config.MODEL_BACKBONE,
            'image_size': config.IMAGE_SIZE
        }
    }
    torch.save(checkpoint, filename)
    print(f"  💾 Checkpoint saved: {filename}")

def train():
    """Main training function"""
    print("\n" + "=" * 70)
    print("FACE RECOGNITION TRAINING - LFW DATASET")
    print("=" * 70)
    
    # Set random seed
    set_seed(config.SEED)
    
    # Create model
    model = get_model()
    model = model.to(config.DEVICE)
    
    # Create loss function
    criterion = TripletLoss(margin=config.MARGIN)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Load data
    train_loader, val_loader = get_data_loaders()
    
    # Tensorboard writer
    writer = None
    if config.USE_TENSORBOARD:
        log_dir = os.path.join(config.LOG_DIR, f'run_{time.strftime("%Y%m%d_%H%M%S")}')
        writer = SummaryWriter(log_dir)
        print(f"\n📊 TensorBoard logs: {log_dir}")
        print(f"   Run: tensorboard --logdir={config.LOG_DIR}\n")
    
    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING START")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{config.NUM_EPOCHS}]")
        print("-" * 70)
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch, writer)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, epoch, writer)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
        
        # Save checkpoint periodically
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = os.path.join(config.MODEL_DIR, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, 'final_model.pth')
    save_checkpoint(model, optimizer, config.NUM_EPOCHS - 1, train_loss, final_model_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {config.MODEL_DIR}")
    
    if writer:
        writer.close()
    
    print("\n🎯 Next steps:")
    print("  python FaceRecognition/src/evaluate.py  # Evaluate on LFW pairs")

if __name__ == '__main__':
    train()
