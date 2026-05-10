import os
import math
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE_FOLDER = 'data/datasets'
os.makedirs(CACHE_FOLDER, exist_ok=True)

DATASETS_TO_PROCESS = [
    {
        'model_id': 'olmo_7B_0424',
        'tokenizer': 'allenai/OLMo-7B-0424-hf',    # Official HF tokenizer for OLMo 0424 7B
        'dataset': 'allenai/dolma',                # The Dolma dataset
        'split': 'train',
        'tokens_to_sample': float('inf'),          # Run until the dataset runs out
        'estimated_total_tokens': 2_000_000_000_000, # Approximately 2T tokens in Dolma
        'output': 'dolma_token_frequencies.csv',
        'batch_size': 1000                         # Documents per batch
    }
]

def generate_frequencies(config):
    print(f"\nProcessing frequencies for {config['model_id']}...")
    
    # Fast Rust-based tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], use_fast=True)
    vocab_size = tokenizer.vocab_size
    
    # Initialize the high-speed PyTorch counter on CPU, we use this list for the token count.
    global_counts = torch.zeros(vocab_size, dtype=torch.int64)

    # Stream the dataset, download the data as we see it.
    dataset = load_dataset(config['dataset'], split=config['split'], streaming=True, trust_remote_code=True)
    dataset_iter = iter(dataset)
    
    total_tokens = 0
    batch_size = config.get('batch_size', 1000)
    estimated_total = config.get('estimated_total_tokens', 1) # Prevent division by zero
    
    print(f"  Streaming, batching, and vector-counting...")
    print(f"  Estimated target: ~{estimated_total:,} tokens")
    
    while total_tokens < config['tokens_to_sample']: 
        # 1. Gather a batch of texts
        texts = []
        for _ in range(batch_size):
            try:
                row = next(dataset_iter)
                if row.get('text'):
                    texts.append(row['text'])
            except StopIteration:
                break # End of dataset
        
        if not texts:
            break # No more data to process

        # 2. Tokenize the entire batch at once
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
        
        # 3. Flatten into a single 1D numpy array
        flat_ids = np.concatenate(encoded['input_ids'])
        
        # 4. Convert to tensor and count
        tensor_ids = torch.tensor(flat_ids, dtype=torch.int64)
        batch_counts = torch.bincount(tensor_ids, minlength=vocab_size)
        
        # 5. Add to global odometer
        global_counts += batch_counts
        total_tokens += len(flat_ids)
        
        # Print updates roughly every 10M tokens with percentage
        if total_tokens % 10_000_000 < (batch_size * 1000): 
            pct = (total_tokens / estimated_total) * 100
            print(f"  ... {total_tokens:,} tokens processed (~{pct:.4f}%)")

    print("\n  Dataset stream complete!")
    print("  Aggregating and calculating Log Frequency...")
    
    # Convert back to standard Python list for Pandas
    counts_list = global_counts.tolist()
    data = []
    
    for token_id, count in enumerate(counts_list):
        try:
            token_str = tokenizer.decode([token_id])
        except Exception:
            token_str = ""
            
        data.append({
            'token_id': token_id,
            'token_str': token_str,
            'count': count,
            'log_count': math.log10(count) if count > 0 else 0
        })

    df = pd.DataFrame(data)
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)
    
    output_path = os.path.join(CACHE_FOLDER, config['output'])
    df.to_csv(output_path, index=False)
    print(f"  ✅ Saved to {output_path} (Final Token Count: {total_tokens:,})")

if __name__ == "__main__":
    for config in DATASETS_TO_PROCESS:
        generate_frequencies(config)