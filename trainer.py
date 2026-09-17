"""
Training Engine with Validation & Early Stopping Mechanism for Custom Raw BERT Classifier.
"""

import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import config


class EarlyStopping:
    """
    Early Stopping monitor to stop training when validation loss stops improving.
    Saves the best model checkpoint to disk.
    """
    def __init__(self, patience: int = config.PATIENCE, verbose: bool = True, save_path: str = config.MODEL_SAVE_PATH):
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_saved = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            if self.verbose:
                print(f"  [EarlyStopping] Validation loss improved ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model checkpoint to '{self.save_path}'...")
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.save_path)
            self.best_model_saved = True
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] Validation loss did not improve. Counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """
    Trains the Custom BERT Classifier for one single epoch.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="  Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        # Forward pass through Custom Raw BERT Classifier (returns logits directly)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Calculate CrossEntropyLoss explicitly
        loss = criterion(logits, labels)

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def eval_epoch(model, dataloader, criterion, device):
    """
    Evaluates the Custom BERT Classifier on the validation dataset.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy


def fit_model(
    model,
    train_loader,
    val_loader,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    patience: int = config.PATIENCE,
    device = config.DEVICE,
    save_path: str = config.MODEL_SAVE_PATH
):
    """
    Complete model training pipeline with optimizer setup, linear warmup, CrossEntropyLoss,
    validation tracking, early stopping, and metric history collection.
    """
    print("\n" + "=" * 60)
    print("        STARTING CUSTOM RAW BERT MODEL FINE-TUNING")
    print("=" * 60)
    
    # Loss function for multi-class/binary classification
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (AdamW with weight decay)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    early_stopper = EarlyStopping(patience=patience, verbose=True, save_path=save_path)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Results Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc * 100:.2f}%")
        
        # Check early stopping condition
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"\n[EARLY STOPPING TRIGGERED] Stopping training early at Epoch {epoch}.")
            break
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    
    # Load the best saved model state
    if early_stopper.best_model_saved:
        print(f"Restoring best model weights from '{save_path}' for evaluation...")
        model.load_state_dict(torch.load(save_path, map_location=device))
        
    return history, model
