"""
Main Executable Script for BERT Sentiment Analysis on IMDB Movie Reviews Dataset.

Run with default settings (Sample Mode for quick testing):
    python main.py

Run on full 50,000 dataset:
    python main.py --full

Customize parameters:
    python main.py --epochs 4 --batch_size 16 --lr 2e-5 --model bert-base-uncased
"""

import argparse
import sys
import config
from dataset import (
    load_imdb_data,
    perform_eda,
    prepare_data_splits,
    create_data_loaders
)
from model import get_tokenizer, build_bert_model
from trainer import fit_model
from evaluate import (
    get_test_predictions,
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    print_final_summary
)


def main():
    parser = argparse.ArgumentParser(description="BERT Sentiment Analysis on IMDB Dataset")
    parser.add_argument('--full', action='store_true', help="Run on full 50,000 IMDB dataset instead of sample subset")
    parser.add_argument('--sample_size', type=int, default=config.SAMPLE_SIZE, help="Sample size when running in sample mode")
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help="Batch size for training and evaluation")
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument('--model', type=str, default=config.BERT_MODEL_NAME, help="Pretrained BERT checkpoint")
    
    args = parser.parse_args()
    
    sample_mode = not args.full
    sample_size = args.sample_size
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    model_name = args.model
    
    print("=" * 60, flush=True)
    print("        BERT SENTIMENT ANALYSIS PIPELINE INITIALIZATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Mode          : {'Sample Mode (' + str(sample_size) + ' samples)' if sample_mode else 'Full Dataset Mode (50,000 samples)'}", flush=True)
    print(f"BERT Model    : {model_name}", flush=True)
    print(f"Batch Size    : {batch_size}", flush=True)
    print(f"Max Epochs    : {epochs}", flush=True)
    print(f"Learning Rate : {learning_rate}", flush=True)
    print(f"Device        : {config.DEVICE}", flush=True)
    print("=" * 60, flush=True)


    # Step 1: Load IMDB Data (Official Pre-Split Train & Test sets)
    raw_train_df, raw_test_df = load_imdb_data(sample_mode=sample_mode, sample_size=sample_size)
    
    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(raw_train_df, raw_test_df)
    
    # Step 3: Prepare Data Splits (Train, Validation, Test)
    train_df, val_df, test_df = prepare_data_splits(raw_train_df, raw_test_df)
    
    # Step 4: Tokenization & PyTorch DataLoaders
    tokenizer = get_tokenizer(model_name)
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size, max_len=config.MAX_LEN
    )
    
    # Step 5: Build BERT Model
    model = build_bert_model(model_name=model_name, num_classes=config.NUM_CLASSES, device=config.DEVICE)
    
    # Step 6: Fine-Tune Model with Validation & Early Stopping
    history, trained_model = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
        patience=config.PATIENCE,
        device=config.DEVICE,
        save_path=config.MODEL_SAVE_PATH
    )
    
    # Step 7: Plot Training & Validation Loss/Accuracy Curves
    plot_training_curves(history, save_path='training_validation_curves.png')
    
    # Step 8: Evaluate Final Model on Test Set
    y_true, y_pred, y_probs = get_test_predictions(trained_model, test_loader, device=config.DEVICE)
    metrics = compute_metrics(y_true, y_pred)
    
    # Step 9: Plot Confusion Matrix
    plot_confusion_matrix(metrics['confusion_matrix'], class_names=config.CLASS_NAMES, save_path='confusion_matrix.png')
    
    # Step 10: Print Final Structured Summary Report & Fitting Diagnosis
    print_final_summary(metrics, sample_mode=sample_mode, sample_size=sample_size, history=history)


if __name__ == '__main__':
    main()
