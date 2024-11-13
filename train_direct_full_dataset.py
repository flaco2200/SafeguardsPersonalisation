# train_full_dataset.py

import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn  # Import nn for loss function

from utils_direct import (
    calculate_metrics,
    evaluate_model,
    train_epoch,
    save_checkpoint,
    load_checkpoint
)

def main():
    # ============================
    # Configuration Parameters
    # ============================
    # Random Seeds for Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loading
    preprocessed_data_path = "preprocessed_datasets/preprocessed_direct_full_train_dataset.pt"
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_data_path}")
    preprocessed_data = torch.load(preprocessed_data_path)
    
    input_ids = preprocessed_data['input_ids']
    attention_mask = preprocessed_data['attention_mask']
    labels = preprocessed_data['labels'].float()  # Ensure labels are floats for BCEWithLogitsLoss
    
    # Training Hyperparameters
    total_epochs = 50
    batch_size = 32
    learning_rate = 1e-5
    validation_split = 0.1
    early_stopping_patience = 30
    checkpoint_path = "direct_full_dataset_checkpoint.pth"  # Path to save/load checkpoints
    
    # ============================
    # Model Initialization
    # ============================
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=1
    ).to(device)

    # Explicitly set problem type to single_label_classification
    model.config.problem_type = "single_label_classification"
    
    # ============================
    # Loss Function
    # ============================
    criterion = nn.BCEWithLogitsLoss()
    
    # ============================
    # Dataset Preparation
    # ============================
    dataset_size = len(input_ids)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    full_dataset = TensorDataset(input_ids, attention_mask, labels)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # ============================
    # Optimizer and Scheduler
    # ============================
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    num_training_steps = total_epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize GradScaler for Mixed Precision
    scaler = GradScaler()
    
    # ============================
    # Logging Setup
    # ============================
    writer = SummaryWriter(log_dir='runs/direct_roberta_classifier_full')
    
    # ============================
    # Checkpoint Loading
    # ============================
    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, scaler, training_history, best_val_f1, start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, filename=checkpoint_path
        )
        print(f"Resumed training from checkpoint: {checkpoint_path} at epoch {start_epoch}.")
        early_stopping_counter = 0  # Initialize early stopping counter
    else:
        training_history = {
            'train_loss': [], 'val_loss': [],
            'train_f1': [], 'val_f1': []
        }
        best_val_f1 = 0.0
        start_epoch = 1
        early_stopping_counter = 0  # Initialize early stopping counter
    
    # ============================
    # Training Loop
    # ============================
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            print(f"\nEpoch {epoch}/{total_epochs}")
            
            # Training Phase
            train_metrics, train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, scaler, epoch, total_epochs, criterion
            )
            
            # Validation Phase
            val_metrics, val_loss = evaluate_model(
                model, val_loader, device, criterion
            )
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('F1/Train', train_metrics['f1'], epoch)
            writer.add_scalar('F1/Validation', val_metrics['f1'], epoch)
            writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            
            # Print Epoch Results
            print(f"\nTraining Results:")
            print(f"Average Loss: {train_loss:.4f}")
            print(f"F1 Score: {train_metrics['f1']:.4f}")
            print(f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            print(f"\nValidation Results:")
            print(f"Average Loss: {val_loss:.4f}")
            print(f"F1 Score: {val_metrics['f1']:.4f}")
            print(f"Accuracy: {val_metrics['accuracy']:.4f}")
            print("\nDetailed Validation Report:")
            print(val_metrics['detailed_report'])
            
            # Check for Improvement
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'training_history': training_history,
                    'best_val_f1': best_val_f1
                }, filename="direct_roberta_weights_full_dataset_best.pt")
                print(f"New best model saved with validation F1: {best_val_f1:.4f}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(f"Validation F1 didn't improve for {early_stopping_counter} epoch(s).")
                
                if early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs.")
                    break
            
            # Update Training History
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['train_f1'].append(train_metrics['f1'])
            training_history['val_f1'].append(val_metrics['f1'])
            
            # Save Checkpoint
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'training_history': training_history,
                'best_val_f1': best_val_f1
            }, filename=checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    finally:
        # Save Final Model and Training History
        torch.save(model.state_dict(), "direct_roberta_weights_full_dataset_final.pt")
        torch.save(training_history, "direct_training_history_full.pt")
        writer.close()
        
        print("\nTraining completed!")
        print(f"Best Validation F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()
