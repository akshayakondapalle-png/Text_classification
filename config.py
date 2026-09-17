import torch
BERT_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
NUM_CLASSES = 2
CLASS_NAMES = ['Negative', 'Positive']
SAMPLE_MODE = True

# Number of total reviews to sample when SAMPLE_MODE = True
SAMPLE_SIZE = 2000

# Train/Val/Test split ratios
TEST_SIZE = 0.20  # 20% reserved for testing
VAL_SIZE = 0.15   # 15% of training split rserved for validation

# ==========================================
# 3. TRAINING HYPERPARAMETERS
# ==========================================
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Early stopping patience (stop training if val loss doesn't improve after N epochs)
PATIENCE = 2

# Reproducibility seed
RANDOM_SEED = 42

# Path to save the best performing model checkpoint
MODEL_SAVE_PATH = 'best_bert_sentiment_model.pt'

# Device selection (Apple Silicon MPS -> CUDA -> CPU)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

DEVICE = get_device()
