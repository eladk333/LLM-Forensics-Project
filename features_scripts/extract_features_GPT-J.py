import os
import torch
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_weight_stats(embedding_matrix):
    l2_norm = np.linalg.norm(embedding_matrix, axis=1)
    variance = np.var(embedding_matrix, axis=1)
    mean = np.mean(embedding_matrix, axis=1)
    l1_norm = np.linalg.norm(embedding_matrix, ord=1, axis=1)
    global_mean_vector = np.mean(embedding_matrix, axis=0)
    dist_to_center = np.linalg.norm(embedding_matrix - global_mean_vector, axis=1)
    return l2_norm, variance, mean, l1_norm, dist_to_center

def get_advanced_stats(embedding_matrix):
    skew_val = skew(embedding_matrix, axis=1)
    kurt_val = kurtosis(embedding_matrix, axis=1)
    return skew_val, kurt_val

def extract_gptj_features():
    model_id = "EleutherAI/gpt-j-6B"
    
    print(f"{'='*40}")
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading model {model_id}...")
    # Using 'dtype' instead of 'torch_dtype' to keep the console clean
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    # 1. Get the ACTUAL dimensions from the weights themselves
    print("  Extracting embedding weights...")
    embeddings = model.get_input_embeddings().weight.detach().cpu().to(torch.float32).numpy()
    actual_rows = embeddings.shape[0] # This will be 50400
    tokenizer_vocab_size = tokenizer.vocab_size # This is 50257
    
    print(f"  Detected {actual_rows} rows in weights (Tokenizer size: {tokenizer_vocab_size})")

    # 2. Extract Token Information carefully
    print("  Extracting token strings...")
    token_strings = []
    for i in range(actual_rows):
        if i < tokenizer_vocab_size:
            token_strings.append(tokenizer.decode([i]))
        else:
            # Handle the padding/extra tokens
            token_strings.append(f"[PAD_{i}]")

    token_lengths = [len(s.strip()) if len(s.strip()) > 0 else 0 for s in token_strings]
    is_upper = [1 if (s.strip() and s.strip()[0].isupper()) else 0 for s in token_strings]
    
    df = pd.DataFrame({
        'token_id': np.arange(actual_rows),
        'token_str': token_strings,
        'token_len': token_lengths,
        'is_upper': is_upper
    })
    
    # 3. Calculate Stats
    print("  Extracting weight statistics...")
    l2, var, mean, l1, dist = get_weight_stats(embeddings)
    df['embedding_norm'] = l2
    df['weight_variance'] = var
    df['weight_mean'] = mean
    df['l1_norm'] = l1
    df['dist_to_center'] = dist
    
    print("  Extracting advanced stats...")
    skew_val, kurt_val = get_advanced_stats(embeddings)
    df['weight_skew'] = skew_val
    df['weight_kurtosis'] = kurt_val
    
    # 4. Get Logit Norms
    print("  Extracting logit norms...")
    lm_head_weights = model.get_output_embeddings().weight.detach().cpu().to(torch.float32).numpy()
    
    # Safety check: ensure LM Head also matches the 50400 size
    if lm_head_weights.shape[0] == actual_rows:
        df['logit_norm'] = np.linalg.norm(lm_head_weights, axis=1)
    else:
        # If for some reason they don't match, we pad with NaN
        logit_norms = np.linalg.norm(lm_head_weights, axis=1)
        full_logit_norms = np.full(actual_rows, np.nan)
        full_logit_norms[:len(logit_norms)] = logit_norms
        df['logit_norm'] = full_logit_norms

    output_file = "gptj_model_features.csv"
    df.to_csv(output_file, index=False)
    print(f"  ✅ Saved: {output_file}")

if __name__ == "__main__":
    extract_gptj_features()
    print("\nDone.")