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
    Loads the IMDB Movie Reviews dataset.
    
    Args:
        sample_mode (bool): If True, samples a subset of reviews for fast execution/testing.
        sample_size (int): Total number of reviews to sample when sample_mode=True.
        
    Returns:
        pd.DataFrame: Formatted DataFrame containing 'text' and 'label' columns.
    """
    print("Loading IMDB Movie Reviews dataset...")
    raw_ds = load_dataset("imdb")
    
    train_df = pd.DataFrame(raw_ds['train'])
    test_df = pd.DataFrame(raw_ds['test'])
    
    # Combine full dataset (50,000 reviews: 25k train, 25k test)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Clean review text
    full_df['text'] = full_df['text'].apply(clean_text)
    
    if sample_mode and sample_size < len(full_df):
        print(f"[SAMPLE MODE ACTIVE] Sampling {sample_size} reviews from full {len(full_df):,} dataset...")
        # Stratified sampling to maintain exact class balance
        full_df = full_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 2, random_state=config.RANDOM_SEED)
        ).reset_index(drop=True)
    else:
        print(f"[FULL DATASET ACTIVE] Loaded complete {len(full_df):,} reviews.")
        
    return full_df


def perform_eda(df: pd.DataFrame):
    """
    Explores the dataset and prints summary statistics:
    - Positive & Negative review counts
    - Text length (word count & character count) statistics
    - Class distribution check
    """
    print("\n" + "=" * 60)
    print("           EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    total_reviews = len(df)
    label_counts = df['label'].value_counts()
    pos_count = label_counts.get(1, 0)
    neg_count = label_counts.get(0, 0)
    
    print(f"Total Reviews Analyzed      : {total_reviews:,}")
    print(f"Number of Positive Reviews (1): {pos_count:,} ({pos_count / total_reviews * 100:.2f}%)")
    print(f"Number of Negative Reviews (0): {neg_count:,} ({neg_count / total_reviews * 100:.2f}%)")
    
    print("=" * 60 + "\n")


def prepare_data_splits(df: pd.DataFrame, test_size: float = config.TEST_SIZE, val_size: float = config.VAL_SIZE):
    """
    Splits the dataset into Training, Validation, and Test sets using stratified sampling.
    """
    # First split: Separate Test set from Train+Val
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=config.RANDOM_SEED,
        stratify=df['label']
    )
    
    # Second split: Separate Validation set from Training set
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=config.RANDOM_SEED,
        stratify=train_val_df['label']
    )
    
    print(f"Data Split Summary:")
    print(f"  Training Set   : {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation Set : {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test Set       : {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


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
