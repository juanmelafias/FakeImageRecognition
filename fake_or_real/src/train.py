"""
Training script for AI-Generated Image Detection
"""
import os
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from model import get_model
from dataset import get_data_loaders

def setup_logging():
    """Setup logging to both file and console"""
    # Create logs directory
    log_dir = os.path.join(config.PROJECT_ROOT, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    return logger

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    epoch_start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    epoch_time = time.time() - epoch_start_time
    
    logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN] - "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s")
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, logger):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]"
    pbar = tqdm(val_loader, desc=desc)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]   - "
                f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, val_acc, is_best, logger, filename='checkpoint.pth'):
    """Save model checkpoint"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    filepath = os.path.join(config.MODEL_DIR, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': {
            'model_name': config.MODEL_NAME,
            'num_classes': config.NUM_CLASSES,
            'image_size': config.IMAGE_SIZE,
        }
    }
    
    torch.save(checkpoint, filepath)
    
    if is_best:
        best_filepath = os.path.join(config.MODEL_DIR, 'best_model.pth')
        torch.save(checkpoint, best_filepath)
        message = f"💾 Saved best model with val_acc: {val_acc:.2f}%"
        print(message)
        logger.info(message)

def train():
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    
    print("\n" + "="*60)
    print("AI-Generated Image Detection - Training")
    print("="*60)
    
    logger.info("="*60)
    logger.info("Starting AI-Generated Image Detection Training")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    logger.info(f"Max epochs: {config.NUM_EPOCHS}")
    logger.info(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE}")
    logger.info("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders()
    logger.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = get_model()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # TensorBoard writer
    writer = SummaryWriter(config.TENSORBOARD_LOG_DIR)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH {epoch+1}/{config.NUM_EPOCHS}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, epoch, logger
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE, epoch, logger
        )
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            logger.info(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            improvement = val_acc - best_val_acc
            logger.info(f"✓ New best validation accuracy! "
                       f"Improved by {improvement:.2f}% ({best_val_acc:.2f}% → {val_acc:.2f}%)")
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
        
        if config.SAVE_BEST_ONLY:
            if is_best:
                save_checkpoint(model, optimizer, epoch, val_acc, is_best, logger)
        else:
            if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
                save_checkpoint(model, optimizer, epoch, val_acc, is_best, logger,
                              filename=f'checkpoint_epoch_{epoch+1}.pth')
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            message = f"⏹️  Early stopping triggered after {epoch+1} epochs - No improvement for {config.EARLY_STOPPING_PATIENCE} epochs"
            print(f"\n{message}")
            logger.info(message)
            break
        
        print("-" * 60)
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODEL_DIR}")
    
    logger.info("="*60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Total epochs: {epoch+1}")
    logger.info("="*60)
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    logger.info("\nFinal Test Set Evaluation:")
    
    # Load best model
    best_model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test evaluation
    model.eval()
    test_running_loss = 0.0
    test_correct = 0
    test_total = 0
    
    pbar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{test_running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*test_correct/test_total:.2f}%'
            })
    
    test_loss = test_running_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    
    logger.info(f"TEST RESULTS - Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")
    logger.info("="*60)
    
    writer.close()
    
    return model

if __name__ == "__main__":
    try:
        model = train()
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
