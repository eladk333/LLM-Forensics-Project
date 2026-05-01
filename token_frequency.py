import os
import math
import pandas as pd
import torch
from transformers import GPT2Tokenizer

# Paths
CACHE_FOLDER = 'data/datasets'
INPUT_PT = os.path.join(CACHE_FOLDER, 'wiki_103_full_cache.pt')
OUTPUT_CSV = os.path.join(CACHE_FOLDER, 'wiki_103_full_cache_token_frequencies.csv')

def generate_frequency_of_tokens():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print(f"Loading cached tokens from {INPUT_PT}...")
    # Load the PyTorch tensor (mapping to CPU to avoid VRAM limits if saved on GPU)
    tensor_data = torch.load(INPUT_PT, map_location='cpu')

    # Ensure it's a flat 1D tensor, and cast to int64 for bincount
    if not isinstance(tensor_data, torch.Tensor):
        tokens_tensor = torch.tensor(tensor_data)
    else:
        tokens_tensor = tensor_data
        
    tokens_tensor = tokens_tensor.flatten().to(torch.int64)

    print("Counting token frequencies...")
    # torch.bincount is practically instantaneous compared to a Python for loop
    counts = torch.bincount(tokens_tensor)
    total_tokens = tokens_tensor.numel()

    print(f"Finished. Scanned {total_tokens:,} total tokens.")

    data = []
    # Find all token IDs that have a count > 0 (as_tuple=True safely returns a 1D list of indices)
    non_zero_indices = torch.nonzero(counts, as_tuple=True)[0]

    print("Decoding tokens and formatting data...")
    for token_id_tensor in non_zero_indices:
        token_id = token_id_tensor.item()
        count = counts[token_id].item()

        token_str = tokenizer.decode([token_id])
        log_freq = math.log10(count)
        
        data.append({
            'token_id': token_id,
            'token_str': token_str,
            'count': count,
            'log_count': log_freq
        })

    print("Creating DataFrame...")
    df = pd.DataFrame(data)
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    print(f"Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Preview of dataset:")
    print(df.head(10))

if __name__ == '__main__':
    generate_frequency_of_tokens()