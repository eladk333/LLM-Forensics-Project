import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import skew, kurtosis

# Configuration
MODELS = {
    'gptj_6B': 'EleutherAI/gpt-j-6B',
    'olmo_7B': 'allenai/OLMo-7B-hf'
}
OUTPUT_DIR = 'data/models'

def get_weight_stats(embedding_matrix):
    l2_norm = np.linalg.norm(embedding_matrix, axis=1)
    variance = np.var(embedding_matrix, axis=1)
    mean = np.mean(embedding_matrix, axis=1)
    l1_norm = np.linalg.norm(embedding_matrix, ord=1, axis=1)
    global_mean_vector = np.mean(embedding_matrix, axis=0)
    dist_to_center = np.linalg.norm(embedding_matrix - global_mean_vector, axis=1)
    return l2_norm, variance, mean, l1_norm, dist_to_center

def extract_for_model(model_id, hf_repo):
    print(f"\n{'='*40}\nProcessing {model_id} ({hf_repo})...")
    
    # Load with automatic device mapping to utilize multiple GPUs seamlessly
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    model = AutoModelForCausalLM.from_pretrained(
        hf_repo, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # Extract Embedding Matrix
    print("  Extracting Embeddings...")
    if 'gpt-j' in hf_repo.lower():
        embeddings = model.transformer.wte.weight.detach().float().cpu().numpy()
        logits = model.lm_head.weight.detach().float().cpu().numpy()
    elif 'olmo' in hf_repo.lower():
        embeddings = model.model.embed_tokens.weight.detach().float().cpu().numpy()
        logits = model.lm_head.weight.detach().float().cpu().numpy()
    else:
        raise ValueError("Unknown architecture layout.")

    vocab_size = embeddings.shape[0]
    
    # We decode in batches to avoid locking up the thread
    print("  Decoding vocabulary...")
    token_strings = []
    for i in range(vocab_size):
        try:
            token_strings.append(tokenizer.decode([i]))
        except Exception:
            token_strings.append("")

    token_lengths = [len(s.strip()) if len(s.strip()) > 0 else 0 for s in token_strings]
    is_upper = [1 if (s.strip() and s.strip()[0].isupper()) else 0 for s in token_strings]

    df = pd.DataFrame({
        'token_id': np.arange(vocab_size),
        'token_str': token_strings,
        'token_len': token_lengths,
        'is_upper': is_upper
    })

    print("  Calculating statistical features...")
    l2, var, mean, l1, dist = get_weight_stats(embeddings)
    df['embedding_norm'] = l2
    df['weight_variance'] = var
    df['weight_mean'] = mean
    df['l1_norm'] = l1
    df['dist_to_center'] = dist
    df['weight_skew'] = skew(embeddings, axis=1)
    df['weight_kurtosis'] = kurtosis(embeddings, axis=1)

    print("  Extracting logit norms...")
    df['logit_norm'] = np.linalg.norm(logits, axis=1)

    # Save
    save_folder = os.path.join(OUTPUT_DIR, model_id)
    os.makedirs(save_folder, exist_ok=True)
    output_file = os.path.join(save_folder, 'model_features.csv')
    df.to_csv(output_file, index=False)
    print(f"  ✅ Saved features to: {output_file}")
    
    # Free up VRAM before loading the next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for model_id, repo in MODELS.items():
        extract_for_model(model_id, repo)