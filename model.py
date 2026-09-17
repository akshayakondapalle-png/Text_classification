"""
Raw BERT Model Architecture with Custom PyTorch Classification Head.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import config


class CustomBertClassifier(nn.Module):
    """
    Custom Sentiment Classifier built on top of raw pretrained BERT backbone.
    
    Architecture:
    Raw BERT Backbone (bert-base-uncased) -> [CLS] Pooler Vector (768 Dim) -> Dropout (0.3) -> Linear FC Layer (768 -> 2)
    """
    def __init__(self, model_name: str = config.BERT_MODEL_NAME, num_classes: int = config.NUM_CLASSES, dropout_rate: float = 0.3):
        super(CustomBertClassifier, self).__init__()
        
        # 1. Base Raw BERT Transformer Model
        print(f"Loading raw pretrained BERT backbone: '{model_name}'...")
        self.bert = BertModel.from_pretrained(model_name)
        
        # 2. Custom Classification Head Layers
        self.dropout = nn.Dropout(p=dropout_rate)
        # Fully Connected (FC) Linear Layer mapping BERT hidden size (768) to number of target classes (2)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through raw BERT backbone and custom classification head.
        """
        # Pass input tensors through base raw BERT model
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract the [CLS] representation (pooler_output tensor of shape [batch_size, 768])
        pooled_output = outputs.pooler_output
        
        # Apply Dropout for regularization
        x = self.dropout(pooled_output)
        
        # Pass through Fully Connected (FC) Linear classifier layer
        logits = self.fc(x)
        
        return logits


def get_tokenizer(model_name: str = config.BERT_MODEL_NAME):
    """
    Loads the pretrained BERT Tokenizer from Hugging Face.
    """
    print(f"Loading tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def build_bert_model(model_name: str = config.BERT_MODEL_NAME, num_classes: int = config.NUM_CLASSES, device=config.DEVICE):
    """
    Builds the custom raw BERT Classifier and transfers it to the compute device.
    """
    print(f"Initializing Custom Raw BERT Classifier on device: {device}...")
    model = CustomBertClassifier(model_name=model_name, num_classes=num_classes)
    model.to(device)
    return model
