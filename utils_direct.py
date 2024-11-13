# utils_direct.py

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def calculate_metrics(predictions, labels):
    """Calculate classification metrics for binary classification."""
    # Move tensors to CPU
    predictions = predictions.cpu()
    labels = labels.cpu()
    
    # Binarize predictions
    predictions = (predictions >= 0.5).float()
    
    # Calculate metrics
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    detailed_report = classification_report(
        labels,
        predictions,
        target_names=['Not Offensive', 'Offensive'],
        zero_division=0
    )
    
    return {
        'f1': f1,
        'accuracy': accuracy,
        'detailed_report': detailed_report
    }

def evaluate_model(model, data_loader, device, criterion):
    """Evaluate the model on the given data loader."""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device)
            }
            labels = batch[2].to(device).squeeze(-1)
    
            with autocast():
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels.float())
                predictions = torch.sigmoid(logits)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / len(data_loader)
    
    return calculate_metrics(all_predictions, all_labels), avg_loss

def train_epoch(model, train_loader, optimizer, scheduler, device, scaler, epoch, total_epochs, criterion):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[1].to(device)
        }
        labels = batch[2].to(device).squeeze(-1)  # Ensure labels have correct shape

        with autocast():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels.float())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            predictions = torch.sigmoid(logits)
            all_predictions.append(predictions)
            all_labels.append(labels)
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = total_loss / len(train_loader)
    
    return calculate_metrics(all_predictions, all_labels), avg_loss

def freeze_model_layers(model, freeze=True, freeze_until_layer=None):
    """
    Freeze or unfreeze model layers.
    If freeze_until_layer is specified, freeze all layers up to that layer.
    """
    if freeze_until_layer is None:
        # Freeze all encoder layers
        for param in model.roberta.parameters():
            param.requires_grad = not freeze
    else:
        # Freeze up to a specific encoder layer
        for name, param in model.roberta.named_parameters():
            layer_num = get_layer_number(name)
            if layer_num < freeze_until_layer:
                param.requires_grad = not freeze
            else:
                param.requires_grad = freeze
    # Always keep the classifier trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

def get_layer_number(name):
    """
    Extract the layer number from parameter name.
    Assumes parameter names contain 'layer.{num}'.
    Returns -1 if not found.
    """
    import re
    match = re.search(r'layer\.(\d+)', name)
    return int(match.group(1)) if match else -1

def save_checkpoint(state, filename="checkpoint.pth"):
    """Save training checkpoint."""
    torch.save(state, filename)

def load_checkpoint(model, optimizer, scheduler, scaler, filename="checkpoint.pth"):
    """Load training checkpoint."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    training_history = checkpoint['training_history']
    best_val_f1 = checkpoint['best_val_f1']
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, scheduler, scaler, training_history, best_val_f1, start_epoch
