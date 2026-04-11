import os
import math
import pandas as pd
from transformers import GPT2Tokenizer
from datasets import load_dataset
from collections import Counter

# Paths
CACHE_FOLDER = 'G:/My Drive/llm/'
OUTPUT_CSV = os.path.join(CACHE_FOLDER, 'openwebtext_token_frequencies.csv')

# How many OpenWebText examples to scan (set to None to scan everything — slow!)
# OpenWebText has ~8M documents; 500k gives a solid frequency estimate quickly.
MAX_EXAMPLES = 500_000


def generate_frequency_of_tokens():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("Loading OpenWebText dataset (streaming)...")
    dataset = load_dataset("openwebtext", split="train", streaming=True)

    token_counts = Counter()
    processed = 0

    for example in dataset:
        tokens = tokenizer.encode(example['text'])
        token_counts.update(tokens)
        processed += 1

        if processed % 10_000 == 0:
            print(f"  Processed {processed:,} documents...", end="\r")

        if MAX_EXAMPLES is not None and processed >= MAX_EXAMPLES:
            break

    print(f"\nFinished. Scanned {processed:,} documents, "
          f"{sum(token_counts.values()):,} total tokens.")

    data = []
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
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    os.makedirs(CACHE_FOLDER, exist_ok=True)
    print(f"Saving to {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("Preview of dataset:")
    print(df.head(10))


if __name__ == '__main__':
    generate_frequency_of_tokens()