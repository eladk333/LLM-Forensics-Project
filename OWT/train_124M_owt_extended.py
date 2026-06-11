import os
import sys
import random
import hashlib
import glob
import re

# Go up one directory to find minGPT
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'minGPT'))

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


# We set const seed so the training will be deterministic
set_seed(3407)

# Model settings (124M config to match your Wikipedia run)
N_LAYER = 12
N_HEAD  = 12
N_EMBD  = 768
BATCH_SIZE = 32

# Target token count to perfectly match the size of WikiText-103
TARGET_TOKENS = 117_800_617  # Exact token count from your Wikipedia run

# Paths - Isolated folder for the new dataset with 3 epochs
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_FOLDER_PATH = os.path.join(MAIN_DIR, 'data', 'models', '124M_owt_3epoch')
DATA_CACHE_PATH = os.path.join(MAIN_DIR, 'data', 'datasets')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)

# Empirically chosen safety factor for character-based collection.
CHAR_SAFETY_FACTOR = 6


def _cache_path():
    key = f"owt_tokens={TARGET_TOKENS}_seed=3407"
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(DATA_CACHE_PATH, f"owt_cache_{tag}.pt")


# The dataset class modified for OpenWebText
class OWTDataset(Dataset):
    def __init__(self, split='train', block_size=128):
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = _cache_path()

        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path)
            assert len(self.tokens) == TARGET_TOKENS, (
                f"Cache token count mismatch: expected {TARGET_TOKENS:,}, "
                f"got {len(self.tokens):,}. Delete the cache file and re-run."
            )
            print(f"Loaded {len(self.tokens)} tokens.")

        else:
            print(f"Cache not found. Building OWT corpus ({TARGET_TOKENS:,} tokens)...")
            dataset = load_dataset("openwebtext", split="train")

            rng = random.Random(3407)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            char_budget = TARGET_TOKENS * CHAR_SAFETY_FACTOR
            collected = []
            total_chars = 0

            print("Collecting documents...")
            for pos, idx in enumerate(indices):
                article_text = dataset[idx]['text']
                if len(article_text) > 0:
                    collected.append(article_text)
                    total_chars += len(article_text) + 1
                    if total_chars >= char_budget:
                        print(f"Char budget reached after {pos+1:,} documents.")
                        break

            print("Tokenizing data...")
            text_data = "\n".join(collected)
            all_tokens = self.tokenizer.encode(text_data)
            print(f"Total tokens after encode: {len(all_tokens):,}")

            if len(all_tokens) < TARGET_TOKENS:
                raise RuntimeError(
                    f"Only {len(all_tokens):,} tokens produced; need {TARGET_TOKENS:,}."
                )

            self.tokens = all_tokens[:TARGET_TOKENS]
            torch.save(self.tokens, cache_path)

        print(f"Total tokens in dataset: {len(self.tokens)}")

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        if start_idx + self.block_size + 1 > len(self.tokens):
            start_idx = len(self.tokens) - self.block_size - 1

        chunk = self.tokens[start_idx : start_idx + self.block_size + 1]
        dix = torch.tensor(chunk, dtype=torch.long)
        x = dix[:-1]
        y = dix[1:]
        return x, y


BEST_LOSS = float('inf')


if __name__ == '__main__':
    print("Loading dataset")
    train_dataset = OWTDataset('train', block_size=128)

    total_samples = len(train_dataset)
    iters_for_one_epoch = total_samples // BATCH_SIZE
    
    # Calculate total maximum iterations for 3 full epochs
    max_total_iters = iters_for_one_epoch * 3

    # Checking for existing checkpoints (handles standard and epoch-ended filenames)
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))

    start_global_step = 0
    latest_ckpt_path = None

    if checkpoint_files:
        # Robust regex extraction that handles both 'ckpt_step_X.pt' and 'ckpt_step_X_epoch_Y.pt'
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(re.search(r'ckpt_step_(\d+)(?:_epoch_\d+)?\.pt', x).group(1)))
        start_global_step = int(re.search(r'ckpt_step_(\d+)(?:_epoch_\d+)?\.pt', latest_ckpt_path).group(1))

        print(f"Found checkpoint: {latest_ckpt_path}")
        print(f"Resuming from GLOBAL step: {start_global_step}")
    else:
        print("No checkpoints found. Starting from 0.")

    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    print(f"Model setup: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}")
    model = GPT(model_config)

    if latest_ckpt_path:
        print(f"Loading weights into model")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 0.0006
    train_config.max_iters = max_total_iters
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    optimizer = model.configure_optimizers(train_config)
    model.to(train_config.device)
    model.train()

    print(f"Starting the training (Global Step {start_global_step} to {max_total_iters})...")

    # Track continuous global step timeline across epochs
    global_step = 0

    # Explicit loop across 3 separate epochs to maintain data alignment
    for epoch in range(1, 4):
        print(f"\n--- Starting Epoch {epoch}/3 ---")
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Strict barrier to stop exactly when 1 epoch data allocation ends
            if batch_idx >= iters_for_one_epoch:
                break
                
            global_step += 1

            # Defensive skip for accurate tracking when resuming from mid-training state
            if global_step <= start_global_step and start_global_step > 0:
                continue

            x = x.to(train_config.device)
            y = y.to(train_config.device)

            logits, loss = model(x, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if loss.item() < BEST_LOSS:
                BEST_LOSS = loss.item()

            if global_step % 10 == 0:
                percent = (global_step / max_total_iters) * 100
                print(f"Global step {global_step}/{max_total_iters} ({percent:.1f}%): loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")

            # Condition A: Standard step intervals (every 500 global steps)
            # Excludes exact epoch boundary matches to prevent double saving
            if global_step % 500 == 0 and batch_idx != (iters_for_one_epoch - 1):
                ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f"Standard checkpoint saved: {ckpt_path}")

        # Condition B: End of Epoch checkpoint saving logic (evaluated immediately as the loader loop finishes)
        if global_step == iters_for_one_epoch * epoch:
            epoch_ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}_epoch_{epoch}.pt')
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"Epoch completion checkpoint saved: {epoch_ckpt_path}")

    print(f"\nTraining completed successfully for 3 full epochs.")