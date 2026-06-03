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

# Paths - Isolated folder for the new dataset
# Go up one directory to save in the main data folder
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_FOLDER_PATH = os.path.join(MAIN_DIR, 'data', 'models', '124M_owt')
DATA_CACHE_PATH = os.path.join(MAIN_DIR, 'data', 'datasets')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)

# Empirically chosen safety factor for character-based collection.
# If insufficient, a RuntimeError is raised before any training begins.
CHAR_SAFETY_FACTOR = 6


# ─────────────────────────────────────────────────────────────────────────────
# Cache path — encodes every parameter that affects the token stream.
# Prevents silent stale-cache bugs if TARGET_TOKENS ever changes.
# ─────────────────────────────────────────────────────────────────────────────
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

        # If we already tokenized and saved the dataset, just load it
        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path)
            assert len(self.tokens) == TARGET_TOKENS, (
                f"Cache token count mismatch: expected {TARGET_TOKENS:,}, "
                f"got {len(self.tokens):,}. Delete the cache file and re-run."
            )
            print(f"Loaded {len(self.tokens)} tokens.")

        # If not, extract exactly the target amount from OpenWebText
        else:
            print(f"Cache not found. Building OWT corpus ({TARGET_TOKENS:,} tokens)...")
            dataset = load_dataset("openwebtext", split="train")

            # Shuffle document order with fixed seed for reproducibility.
            # NOTE: Wikipedia uses the dataset's natural order because it is a
            # complete, fixed corpus. OWT requires a shuffle because we are
            # taking a subset (~1.3%) of a much larger corpus, and the natural
            # document order of OWT may reflect crawl order (temporal/source
            # bias). The shuffle produces a representative sample.
            rng = random.Random(3407)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            # Collect documents until we have enough characters for the encode.
            # We do NOT tokenize per-document because GPT-2 BPE is context-
            # sensitive at document boundaries: encoding documents separately
            # produces more tokens than encoding the joined string once, making
            # the token count incomparable with Wikipedia. Instead we collect
            # by character budget and tokenize the full concatenation once —
            # exactly mirroring the Wikipedia procedure.
            char_budget = TARGET_TOKENS * CHAR_SAFETY_FACTOR
            collected = []
            total_chars = 0

            print("Collecting documents...")
            for pos, idx in enumerate(indices):
                article_text = dataset[idx]['text']  # No .strip() — matches Wikipedia exactly
                if len(article_text) > 0:
                    collected.append(article_text)
                    total_chars += len(article_text) + 1  # +1 for '\n' separator
                    if total_chars >= char_budget:
                        print(f"Char budget reached after {pos+1:,} documents.")
                        break

            # Single encode call — identical procedure to Wikipedia:
            #   Wikipedia: "\n".join(articles) → tokenizer.encode(text_data)
            #   OWT:       "\n".join(articles) → tokenizer.encode(text_data)
            print("Tokenizing data...")
            text_data = "\n".join(collected)
            all_tokens = self.tokenizer.encode(text_data)
            print(f"Total tokens after encode: {len(all_tokens):,}")

            if len(all_tokens) < TARGET_TOKENS:
                raise RuntimeError(
                    f"Only {len(all_tokens):,} tokens produced; "
                    f"need {TARGET_TOKENS:,}. "
                    f"Increase CHAR_SAFETY_FACTOR (currently {CHAR_SAFETY_FACTOR})."
                )

            # Trim to exactly match Wikipedia token count
            self.tokens = all_tokens[:TARGET_TOKENS]
            torch.save(self.tokens, cache_path)

        print(f"Total tokens in dataset: {len(self.tokens)}")

    # Number of sequences we need for the model to see the entire dataset (1 epoch)
    # Identical to WikiDataset.__len__
    def __len__(self):
        return len(self.tokens) // self.block_size

    # Identical to WikiDataset.__getitem__
    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        if start_idx + self.block_size + 1 > len(self.tokens):
            start_idx = len(self.tokens) - self.block_size - 1

        chunk = self.tokens[start_idx : start_idx + self.block_size + 1]
        dix = torch.tensor(chunk, dtype=torch.long)
        x = dix[:-1]
        y = dix[1:]
        return x, y


# To track the lowest loss achieved during training
BEST_LOSS = float('inf')


if __name__ == '__main__':
    print("Loading dataset")
    train_dataset = OWTDataset('train', block_size=128)

    # Calculation of total epoch size
    total_samples = len(train_dataset)
    iters_for_one_epoch = total_samples // BATCH_SIZE

    # Checking for existing checkpoints
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))

    start_iter = 0
    latest_ckpt_path = None

    if checkpoint_files:
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(re.search(r'ckpt_step_(\d+).pt', x).group(1)))
        start_iter = int(re.search(r'ckpt_step_(\d+).pt', latest_ckpt_path).group(1))

        print(f"Found checkpoint: {latest_ckpt_path}")
        print(f"Resuming from GLOBAL step: {start_iter}")
    else:
        print("No checkpoints found. Starting from 0.")

    # Setup of the model config
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    print(f"Model setup: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}")
    model = GPT(model_config)

    # If resuming load weights
    if latest_ckpt_path:
        print(f"Loading weights into model")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    # Training config
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 0.0006
    train_config.max_iters = iters_for_one_epoch
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Exactly matching Wikipedia DataLoader settings
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

    print(f"Starting the training (Step {start_iter} to {iters_for_one_epoch})...")

    for batch_idx, (x, y) in enumerate(train_loader):

        # For when we resume training
        if batch_idx <= start_iter and start_iter > 0:
            continue

        # To stop exactly when we reach the end of 1 epoch
        if batch_idx >= iters_for_one_epoch:
            break

        x = x.to(train_config.device)
        y = y.to(train_config.device)

        logits, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if loss.item() < BEST_LOSS:
            BEST_LOSS = loss.item()

        if batch_idx % 10 == 0:
            percent = (batch_idx / iters_for_one_epoch) * 100
            print(f"step {batch_idx}/{iters_for_one_epoch} ({percent:.1f}%): loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")

        if batch_idx % 500 == 0:
            ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{batch_idx}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_FOLDER_PATH, 'final_model_1_epoch.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
