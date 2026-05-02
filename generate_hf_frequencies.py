import os
import math
import pandas as pd
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

CACHE_FOLDER = 'data/datasets'
os.makedirs(CACHE_FOLDER, exist_ok=True)

DATASETS_TO_PROCESS = [
    # {
    #     'model_id': 'gptj_6B',
    #     'tokenizer': 'EleutherAI/gpt-j-6B',
    #     'dataset': 'monology/pile-uncopyrighted',
    #     'split': 'train',
    #     'tokens_to_sample': 100_000_000, # Adjust sample size as needed
    #     'output': 'pile_token_frequencies.csv'
    # },
    {
        'model_id': 'olmo_7B',
        'tokenizer': 'allenai/OLMo-7B-hf',
        'dataset': 'allenai/dolma',
        'split': 'train',
        'tokens_to_sample': 100_000_000, 
        'output': 'dolma_token_frequencies.csv'
    }
]

def generate_frequencies(config):
    print(f"\nProcessing frequencies for {config['model_id']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    
    # Streaming prevents downloading the massive datasets to disk
   
    dataset = load_dataset(config['dataset'], split=config['split'], streaming=True, trust_remote_code=True)
    
    token_counts = Counter()
    total_tokens = 0
    
    print(f"  Streaming and tokenizing (Target: {config['tokens_to_sample']:,} tokens)...")
    for row in dataset:
        if total_tokens >= config['tokens_to_sample']:
            break
            
        text = row.get('text', '')
        if not text:
            continue
            
        tokens = tokenizer.encode(text)
        token_counts.update(tokens)
        total_tokens += len(tokens)
        
        if total_tokens % 10_000_000 < 10_000: # Print updates roughly every 10M tokens
            print(f"  ... {total_tokens:,} tokens processed")

    print("  Aggregating and calculating Log Frequency...")
    data = []
    for token_id, count in token_counts.items():
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
    print(f"  ✅ Saved to {output_path}")

if __name__ == "__main__":
    for config in DATASETS_TO_PROCESS:
        generate_frequencies(config)