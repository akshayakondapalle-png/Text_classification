"""
Data Processing, EDA, Splitting, and Tokenization Module for BERT Sentiment Analysis.
"""

import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import config


def clean_text(text: str) -> str:
    """
    Cleans raw review text by removing HTML tags like <br /> and normalizing whitespace.
    Retains casing and punctuation as BERT benefits from contextual clues.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML break tags and other HTML markup
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_data(sample_mode: bool = config.SAMPLE_MODE, sample_size: int = config.SAMPLE_SIZE):
    """
    Loads official IMDB Movie Reviews dataset using pre-split Train (25,000) and Test (25,000) splits.
    
    Args:
        sample_mode (bool): If True, samples a subset of reviews for fast execution/testing.
        sample_size (int): Total number of reviews to sample when sample_mode=True.
        
    Returns:
        tuple: (train_df, test_df)
    """
    print("Loading IMDB Movie Reviews dataset...")
    raw_ds = load_dataset("imdb")
    
    # Load official pre-split Train (25,000) and Test (25,000) DataFrames
    train_df = pd.DataFrame(raw_ds['train'])
    test_df = pd.DataFrame(raw_ds['test'])
    
    # Clean review text
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)
    
    if sample_mode and sample_size < (len(train_df) + len(test_df)):
        print(f"[SAMPLE MODE ACTIVE] Sampling {sample_size} reviews from dataset...")
        train_sample = int(sample_size * 0.8)
        test_sample = sample_size - train_sample
        train_df = train_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=train_sample // 2, random_state=config.RANDOM_SEED)
        ).reset_index(drop=True)
        test_df = test_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=test_sample // 2, random_state=config.RANDOM_SEED)
        ).reset_index(drop=True)
    else:
        print(f"[FULL DATASET ACTIVE] Loaded official Train ({len(train_df):,}) and Test ({len(test_df):,}) splits.")
        
    return train_df, test_df


def perform_eda(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Explores the dataset and prints summary statistics:
    - Positive & Negative review counts across official Train & Test sets.
    """
    print("\n" + "=" * 60)
    print("           EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    total_train = len(train_df)
    total_test = len(test_df)
    total_reviews = total_train + total_test
    
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    test_pos = (test_df['label'] == 1).sum()
    test_neg = (test_df['label'] == 0).sum()
    
    print(f"Total Reviews Loaded        : {total_reviews:,}")
    print(f"Official Train Split (25k)  : {total_train:,} (Pos: {train_pos:,}, Neg: {train_neg:,})")
    print(f"Official Test Split (25k)   : {total_test:,} (Pos: {test_pos:,}, Neg: {test_neg:,})")
    print("=" * 60 + "\n")


def prepare_data_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, val_size: float = config.VAL_SIZE):
    """
    Uses official IMDB Train (25,000) and Test (25,000) datasets directly.
    Splits a validation set from Train for Early Stopping monitoring.
    """
    train_sub_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=config.RANDOM_SEED,
        stratify=train_df['label']
    )
    
    print(f"Data Split Summary (Official IMDB Pre-Split):")
    print(f"  Training Set   : {len(train_sub_df):,} samples ({len(train_sub_df)/len(train_df)*100:.1f}% of Train split)")
    print(f"  Validation Set : {len(val_df):,} samples ({len(val_df)/len(train_df)*100:.1f}% of Train split)")
    print(f"  Test Set       : {len(test_df):,} samples (100% Official Test split)")
    
    return train_sub_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class IMDBDataset(Dataset):
    """
    PyTorch Dataset for IMDB Movie Reviews.
    Tokenizes raw text sequences using BERT Tokenizer.
    """
    def __init__(self, texts, labels, tokenizer, max_len=config.MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # BERT Tokenization with truncation, padding, and attention mask creation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,       # Truncate or pad to max_len
            padding='max_length',          # Pad to max_len
            truncation=True,               # Truncate long sequences
            return_token_type_ids=False,   # Not required for simple text classification
            return_attention_mask=True,    # Generate attention mask (1 for token, 0 for padding)
            return_tensors='pt',           # Return PyTorch tensors
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def create_data_loaders(train_df, val_df, test_df, tokenizer, batch_size=config.BATCH_SIZE, max_len=config.MAX_LEN):
    """
    Creates PyTorch DataLoaders for train, validation, and test datasets.
    """
    train_dataset = IMDBDataset(train_df['text'], train_df['label'], tokenizer, max_len)
    val_dataset = IMDBDataset(val_df['text'], val_df['label'], tokenizer, max_len)
    test_dataset = IMDBDataset(test_df['text'], test_df['label'], tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
