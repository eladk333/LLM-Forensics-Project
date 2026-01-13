import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis  # <--- NEW LIBRARY
from mingpt.model import GPT
from transformers import GPT2Tokenizer

# Model config
MODEL_SIZE = '124M'
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# Paths
BASE_FOLDER = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\models'
MODEL_PATH = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}')
OUTPUT_FILE = os.path.join(MODEL_PATH, 'final_frequency_dataset.csv')


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    conf = CONFIGS[MODEL_SIZE]
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = conf['n_layer']
    model_config.n_head = conf['n_head']
    model_config.n_embd = conf['n_embd']
    model_config.vocab_size = 50257
    model_config.block_size = 128

    model = GPT(model_config)
    ckpt_path = os.path.join(MODEL_PATH, 'final_model_1_epoch.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model not found at {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()
    return model, tokenizer

def get_weight_stats(model):
    # Basic Stats (Variance, Mean, Norms)
    embedding_matrix = model.transformer.wte.weight.detach().numpy()
    
    # 1. Existing Features
    l2_norm = np.linalg.norm(embedding_matrix, axis=1)
    variance = np.var(embedding_matrix, axis=1)
    mean = np.mean(embedding_matrix, axis=1)
    
    # 2. NEW: L1 Norm (Manhattan Distance)
    l1_norm = np.linalg.norm(embedding_matrix, ord=1, axis=1)
    
    # 3. NEW: Distance to Center (Euclidean distance to the average token)
    # This checks if the token is an "outlier" (frequent) or "average" (rare)
    global_mean_vector = np.mean(embedding_matrix, axis=0)
    dist_to_center = np.linalg.norm(embedding_matrix - global_mean_vector, axis=1)

    return l2_norm, variance, mean, l1_norm, dist_to_center

def get_advanced_stats(model):
    # Shape Statistics (Skew, Kurtosis)
    embedding_matrix = model.transformer.wte.weight.detach().numpy()
    
    # 4. NEW: Skewness (Asymmetry of the weight distribution)
    # Rare tokens ~ 0 (Symmetric). Frequent tokens != 0.
    skew_val = skew(embedding_matrix, axis=1)
    
    # 5. NEW: Kurtosis (Pointiness/Tail heaviness)
    kurt_val = kurtosis(embedding_matrix, axis=1)
    
    return skew_val, kurt_val

def get_logit_norms(model):
    weights = model.lm_head.weight.detach().numpy()
    return np.linalg.norm(weights, axis=1)

def extract_features():
    print(f"Loading {MODEL_SIZE} model...")
    model, tokenizer = load_model()

    # --- 6. NEW: Tokenizer Features (Zipf's Law) ---
    vocab_size = 50257
    token_strings = [tokenizer.decode([i]) for i in range(vocab_size)]
    
    # Calculate string length (shorter words are often more frequent)
    # We use 'strip' to ignore the leading space ' ' that GPT uses
    token_lengths = [len(s.strip()) if len(s.strip()) > 0 else 0 for s in token_strings]
    
    # Check if first letter is capital (Capitalized words are often rarer proper nouns)
    is_upper = [1 if (s.strip() and s.strip()[0].isupper()) else 0 for s in token_strings]

    df = pd.DataFrame({
        'token_id': np.arange(vocab_size),
        'token_str': token_strings,
        'token_len': token_lengths,  # New
        'is_upper': is_upper         # New
    })
    
    print("Extracting Weight Statistics (Norms, Var, Mean, L1, Dist)...")
    l2, var, mean, l1, dist = get_weight_stats(model)
    df['embedding_norm'] = l2
    df['weight_variance'] = var
    df['weight_mean'] = mean
    df['l1_norm'] = l1              # New
    df['dist_to_center'] = dist     # New

    print("Extracting Advanced Stats (Skew, Kurtosis)...")
    skew_val, kurt_val = get_advanced_stats(model)
    df['weight_skew'] = skew_val    # New
    df['weight_kurtosis'] = kurt_val # New

    print("Extracting Logit Norms...")
    df['logit_norm'] = get_logit_norms(model)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved expanded feature file: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_features()