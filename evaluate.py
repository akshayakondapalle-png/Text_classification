"""
Evaluation, Metrics Computation, Plotting, and Result Synthesis for BERT Sentiment Classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import config


def get_test_predictions(model, test_loader, device=config.DEVICE):
    """
    Runs model inference on the test dataset.
    
    Returns:
        tuple: (y_true, y_pred, y_probs)
    """
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    print("\nRunning inference on the Test Set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Testing", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def compute_metrics(y_true, y_pred):
    """
    Calculates detailed performance metrics: Accuracy, Precision, Recall, F1-Score,
    as well as macro/weighted averages and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Binary metrics (Positive class = 1)
    precision_binary = precision_score(y_true, y_pred, average='binary')
    recall_binary = recall_score(y_true, y_pred, average='binary')
    f1_binary = f1_score(y_true, y_pred, average='binary')

    # Macro & Weighted averages
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4)

    metrics = {
        'accuracy': accuracy,
        'precision_binary': precision_binary,
        'recall_binary': recall_binary,
        'f1_binary': f1_binary,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return metrics


def plot_training_curves(history, save_path='training_validation_curves.png'):
    """
    Plots and saves Training vs Validation Loss and Accuracy curves side-by-side.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Custom aesthetic styling
    sns.set_theme(style="whitegrid")

    # 1. Training vs Validation Loss
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    axes[0].set_title('Training vs Validation Loss', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Training vs Validation Accuracy
    axes[1].plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-o', label='Training Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-s', label='Validation Accuracy', linewidth=2, markersize=6)
    axes[1].set_title('Training vs Validation Accuracy (%)', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves successfully saved to '{save_path}'")


def plot_confusion_matrix(cm, class_names=config.CLASS_NAMES, save_path='confusion_matrix.png'):
    """
    Plots and saves an interactive Confusion Matrix Heatmap.
    """
    plt.figure(figsize=(6, 5))
    
    # Calculate counts and percentages for matrix annotations
    cm_sum = np.sum(cm)
    group_counts = [f"{value:,}" for value in cm.flatten()]
    group_percentages = [f"{value/cm_sum:.2%}" for value in cm.flatten()]
    labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap='Blues',
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title('IMDB Sentiment Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Sentiment', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix heatmap saved to '{save_path}'")


def analyze_model_fitting(history):
    """
    Analyzes training vs validation loss and accuracy curves to diagnose overfitting or underfitting.
    """
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    
    loss_gap = final_val_loss - final_train_loss
    acc_gap = final_train_acc - final_val_acc
    
    analysis_text = "\n" + "=" * 60 + "\n"
    analysis_text += "          MODEL FITTING DIAGNOSTIC ANALYSIS\n"
    analysis_text += "=" * 60 + "\n"
    analysis_text += f"Final Epoch Training Loss   : {final_train_loss:.4f}\n"
    analysis_text += f"Final Epoch Validation Loss : {final_val_loss:.4f}\n"
    analysis_text += f"Final Epoch Train Accuracy  : {final_train_acc * 100:.2f}%\n"
    analysis_text += f"Final Epoch Val Accuracy    : {final_val_acc * 100:.2f}%\n\n"

    if final_train_acc < 0.75 and final_val_acc < 0.75:
        fit_status = "UNDERFITTING"
        explanation = (
            "The model exhibits UNDERFITTING. Both training and validation accuracies remain relatively low, "
            "indicating that the model has not yet learned sufficient representations from the text data. "
            "Consider increasing the number of training epochs, unfreezing more transformer layers, or increasing learning rate."
        )
    elif loss_gap > 0.15 or acc_gap > 0.10:
        fit_status = "SLIGHT OVERFITTING"
        explanation = (
            "The model shows signs of SLIGHT OVERFITTING. The training loss is noticeably lower than the validation loss "
            f"(Loss Gap: {loss_gap:.4f}, Accuracy Gap: {acc_gap*100:.2f}%). While the model performs exceptionally well on the "
            "training data, generalization to unseen data starts to plateau. Early stopping prevented severe overfitting."
        )
    else:
        fit_status = "OPTIMALLY FITTED (WELL-BALANCED GENERALIZATION)"
        explanation = (
            "The model is OPTIMALLY FITTED. The training and validation loss curves decrease in sync, and validation "
            "accuracy tracks closely with training accuracy. The BERT pretrained weights provided effective inductive bias "
            "allowing robust generalization without overfitting."
        )

    analysis_text += f"Status: {fit_status}\n"
    analysis_text += f"Diagnosis:\n{explanation}\n"
    analysis_text += "=" * 60
    
    return fit_status, explanation, analysis_text


def print_final_summary(metrics, sample_mode: bool, sample_size: int, history):
    """
    Prints the required clear summary report.
    """
    print("\n" + "=" * 60)
    print("                FINAL MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model          : BERT ({config.BERT_MODEL_NAME})")
    print(f"Dataset        : IMDB Movie Reviews ({'Subset Sample: ' + str(sample_size) if sample_mode else 'Complete 50,000 Dataset'})")
    print(f"Task           : Binary Sentiment Classification")
    print(f"Test Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision      : {metrics['precision_binary']:.4f}")
    print(f"Recall         : {metrics['recall_binary']:.4f}")
    print(f"F1-Score       : {metrics['f1_binary']:.4f}")
    print("\n--- Classification Report ---")
    print(metrics['classification_report'])
    
    fit_status, explanation, analysis_text = analyze_model_fitting(history)
    print(analysis_text)
