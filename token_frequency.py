import os
import torch
import math
import pandas as pd
from transformers import GPT2Tokenizer
from collections import Counter

# Paths
DATASET_PATH = 'G:/My Drive/llm/wiki_103_full_cache.pt' # Path for the dataset
CACHE_FOLDER = 'G:/My Drive/llm/' # Path for the output
OUTPUT_CSV = os.path.join(CACHE_FOLDER, 'wiki_token_frequencies.csv')

def generate_frequency_of_tokens():

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # Loads our tokenizer

    # Loads the tokenize dataset into a list
    if os.path.exists(DATASET_PATH):
        tokens = torch.load(DATASET_PATH)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        print(f"Loaded {len(tokens):,} tokens.")
    else:
        print(f"No dataset was found at {DATASET_PATH}")
        return

    # Convert the list of tokens into a frequency dictionary
    token_counts = Counter(tokens)

    data = []


    # Creating the dataset
    for token_id, count in token_counts.items():
        token_str = tokenizer.decode([token_id])
        log_freq = math.log10(count)
        data.append({
            'token_id': token_id,
            'token_str': token_str,
            'count': count,
            'log_count': log_freq
        })

    df = pd.DataFrame(data)

    # # Sort for low frequancy tokens
    # initial_len = len(df)
    # df = df[df['count'] >= 5]


    # Sort by Frequency
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    # Save to path
    print(f"Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    # Preview
    print("Preview of dataset:")
    print(df.head(10))

if __name__ == '__main__':
    generate_frequency_of_tokens()