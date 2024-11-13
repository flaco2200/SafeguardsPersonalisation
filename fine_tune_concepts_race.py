# fine_tune_race_group.py

import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from utils_concepts import (
    calculate_metrics,
    evaluate_model,
    train_epoch,
    freeze_model_layers,
    get_layer_number,
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
    
    # Label Definitions
    csv_labels = [
        "harmful_advice", "dangerous", "derogatory", "insensitive", "obscene",
        "personally_informative", "harmful_regulated_goods", "sexual",
        "generally_harmful", "religious_bias", "sexual_orientation_bias",
        "inherited_attributes_bias", "targeting_bias", "generally_bias",
        "misinformation", "politically_affiliated", "polarizing_topics",
        "endorsement", "generally_policy_breaking"
    ]

    #csv_labels = ["is_profane", "is_threat", "is_identity_attack", "is_insult", "is_sexual_harassment"]
    
    # Data Loading
    # Replace 'latinx' with the specific race group you want to fine-tune on
    race_group = 'white'
    preprocessed_data_path = f"preprocessed_datasets/preprocessed_{race_group}_train_dataset.pt"
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(f"Preprocessed data not found at {preprocessed_data_path}")
    preprocessed_data = torch.load(preprocessed_data_path)
    
    input_ids = preprocessed_data['input_ids']
    attention_mask = preprocessed_data['attention_mask']
    labels = preprocessed_data['labels']
    
    # Training Hyperparameters
    total_epochs = 50
    batch_size = 32
    learning_rate = 1e-5
    validation_split = 0.1
    early_stopping_patience = 10
    freeze_until_layer = 6  # Set to an integer to freeze up to that layer
    pretrained_weights_path = "concepts_weights/roberta_weights_full_dataset_best.pt"  # Path to your pre-trained weights
    checkpoint_path = f"{race_group}_checkpoint.pth"  # Path to save/load checkpoints
    unfreeze_epoch = None  # Set to an integer epoch to unfreeze layers
    
    # ============================
    # Model Initialization
    # ============================
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(csv_labels),
        problem_type="multi_label_classification"
    ).to(device)
    
    # Load Pretrained Weights
    if os.path.exists(pretrained_weights_path):
        checkpoint = torch.load(pretrained_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained weights from {pretrained_weights_path}.")
    else:
        raise FileNotFoundError(f"Pre-trained weights not found at {pretrained_weights_path}. Please train the full dataset model first.")
    
    # Freeze Layers if Required
    freeze_model_layers(model, freeze=True, freeze_until_layer=freeze_until_layer)
    print("Model layers frozen up to layer:", freeze_until_layer if freeze_until_layer else "All encoder layers")

    # ============================
    # Compute Class Weights
    # ============================
    # Sum over samples for each label to get label counts
    label_counts = labels.sum(dim=0)  # Shape: [num_labels]
    total_samples = labels.size(0)
    
    max_weight = 5.0  # Example maximum weight
    class_weights = total_samples / (label_counts + 1e-6)
    class_weights = torch.clamp(class_weights, max=max_weight)
    class_weights = class_weights.to(device)
    
    # ============================
    # Define Loss Function with Class Weights
    # ============================
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    
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
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
    
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
    writer = SummaryWriter(log_dir=f'runs/roberta_classifier_{race_group}')
    
    # ============================
    # Checkpoint Loading (Optional)
    # ============================
    if os.path.exists(checkpoint_path):
        model, optimizer, scheduler, scaler, training_history, best_val_f1, start_epoch = load_checkpoint(
            model, optimizer, scheduler, scaler, filename=checkpoint_path
        )
        print(f"Resumed training from checkpoint: {checkpoint_path} at epoch {start_epoch}.")
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
            
            # Optionally Unfreeze Layers at Specific Epoch
            if unfreeze_epoch and epoch == unfreeze_epoch:
                print("Unfreezing all encoder layers for fine-tuning.")
                freeze_model_layers(model, freeze=False)
                # Update optimizer to include newly unfrozen parameters
                optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
                # Reinitialize scheduler with new optimizer
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(0.1 * num_training_steps),
                    num_training_steps=num_training_steps
                )
            
            # Training Phase
            train_metrics, train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, device, scaler, epoch, total_epochs, criterion
            )
            
            # Validation Phase
            val_metrics, val_loss = evaluate_model(
                model, val_loader, device, criterion, label_names=csv_labels
            )
            
            # Logging
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('F1/Train', train_metrics['avg_f1'], epoch)
            writer.add_scalar('F1/Validation', val_metrics['avg_f1'], epoch)
            for idx, label in enumerate(csv_labels):
                writer.add_scalar(f'F1/{label}', val_metrics['f1_scores'][idx], epoch)
            
            # Print Epoch Results
            print(f"\nTraining Results:")
            print(f"Average Loss: {train_loss:.4f}")
            print(f"Average F1 Score: {train_metrics['avg_f1']:.4f}")
            
            print(f"\nValidation Results:")
            print(f"Average Loss: {val_loss:.4f}")
            print(f"Average F1 Score: {val_metrics['avg_f1']:.4f}")
            print("\nDetailed Validation Report:")
            print(val_metrics['detailed_report'])
            
            # Check for Improvement
            if val_metrics['avg_f1'] > best_val_f1:
                best_val_f1 = val_metrics['avg_f1']
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'training_history': training_history,
                    'best_val_f1': best_val_f1
                }, filename=f"roberta_weights_{race_group}_best.pt")
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
            training_history['train_f1'].append(train_metrics['avg_f1'])
            training_history['val_f1'].append(val_metrics['avg_f1'])
            
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
        torch.save(model.state_dict(), f"roberta_weights_{race_group}_final.pt")
        torch.save(training_history, f"training_history_{race_group}.pt")
        writer.close()
        
        print("\nTraining completed!")
        print(f"Best Validation F1: {best_val_f1:.4f}")

if __name__ == "__main__":
    main()