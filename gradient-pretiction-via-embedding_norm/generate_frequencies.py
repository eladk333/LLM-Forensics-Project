import os
import torch
import math
import pandas as pd
from transformers import GPT2Tokenizer

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
BASE_PATH = os.getcwd()
DATASET_PATH = os.path.join(BASE_PATH, "data", "wiki", "wiki_103_full_cache.pt")
OUTPUT_CSV = os.path.join(BASE_PATH, "wiki_token_frequencies.csv")

def generate_frequency_of_tokens():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 

    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset not found at {DATASET_PATH}")
        return

    print("🚀 Loading dataset...")
    tokens = torch.load(DATASET_PATH)
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    
    print(f"✅ Loaded {len(tokens):,} tokens.")

    # 2. OPTIMIZATION: Use PyTorch native counting
    print("🧮 Counting token frequencies using PyTorch bincount...")
    counts = torch.bincount(tokens, minlength=50257) 
    
    # Filter out tokens that never appeared
    valid_token_ids = torch.nonzero(counts).squeeze()
    valid_counts = counts[valid_token_ids]

    data = []
    print("✍️  Decoding tokens and formatting data...")
    
    for token_id, count in zip(valid_token_ids.tolist(), valid_counts.tolist()):
        token_str = tokenizer.decode([token_id])
        log_freq = math.log10(count)
        
        data.append({
            'token_id': token_id,
            'token_str': token_str,
            'count': count,
            'log_count': log_freq
        })

    # 3. Create DataFrame and Sort
    df = pd.DataFrame(data)
    
    print("📊 Sorting data by frequency (Descending)...")
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    # 4. Save and Preview
    print(f"💾 Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n🎉 Done! Preview of the dataset:")
    print(df.head(5))

if __name__ == '__main__':
    generate_frequency_of_tokens()