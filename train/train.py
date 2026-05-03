import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import math
import glob
import re
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


# We set const seed so the training will be deterministic
set_seed(3407)

# Model settings — 7M config
# N_LAYER = 4
# N_HEAD  = 4
# N_EMBD  = 128

# Model settings — 30M config
N_LAYER = 6   # Layers
N_HEAD  = 6   # Attention heads
N_EMBD  = 384  # Embedding Dimension
BATCH_SIZE = 32

# # Model settings — 124M config
# N_LAYER = 12   # Increased from 6
# N_HEAD  = 12   # Increased from 6
# N_EMBD  = 768  # Increased from 384
# Target token count: 7M params * (30/100) ratio = ~21M tokens
# We use 24M as a round number slightly above that
TARGET_TOKENS = 420_000_000

# Paths
CHECKPOINT_FOLDER_PATH = os.path.join(os.getcwd(), 'data', 'models', '30M_owt_420M')
DATA_CACHE_PATH = os.path.join(os.getcwd(), 'data', 'datasets')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)


class OpenWebTextDataset(Dataset):
    def __init__(self, split='train', block_size=128, target_tokens=TARGET_TOKENS, seed=3):
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, f'owt_{target_tokens // 1_000_000}M_cache.pt')

        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path)
            print(f"Loaded {len(self.tokens):,} tokens.")
        else:
            print(f"Cache not found. Loading OpenWebText ({split})...")
            # OpenWebText is train-only; no official val split
            dataset = load_dataset("openwebtext", split="train")

            print(f"Dataset has {len(dataset):,} documents. Randomly sampling to reach {target_tokens:,} tokens...")

            # Shuffle document order with fixed seed for reproducibility
            rng = random.Random(seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            all_tokens = []
            for idx in indices:
                text = dataset[idx]['text'].strip()
                if not text:
                    continue
                chunk_tokens = self.tokenizer.encode(text)
                all_tokens.extend(chunk_tokens)

                if len(all_tokens) >= target_tokens:
                    break

            # Trim to exactly target_tokens
            all_tokens = all_tokens[:target_tokens]
            self.tokens = all_tokens

            print(f"Saving {len(self.tokens):,} tokens to cache...")
            torch.save(self.tokens, cache_path)

        print(f"Total tokens in dataset: {len(self.tokens):,}")

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
    print("Loading training dataset (OpenWebText, ~100M tokens)")
    train_dataset = OpenWebTextDataset(split='train', block_size=128, target_tokens=TARGET_TOKENS)

    # For validation we take the last 1% of the sampled tokens as a held-out set.
    # Since OWT has no official val split, we carve it out manually from the cache.
    val_split_idx = int(len(train_dataset.tokens) * 0.99)
    val_tokens = train_dataset.tokens[val_split_idx:]
    train_dataset.tokens = train_dataset.tokens[:val_split_idx]
    print(f"Train tokens: {len(train_dataset.tokens):,} | Val tokens: {len(val_tokens):,}")

    class _SliceDataset(Dataset):
        def __init__(self, tokens, block_size=128):
            self.tokens = tokens
            self.block_size = block_size
        def __len__(self):
            return len(self.tokens) // self.block_size
        def __getitem__(self, idx):
            start = idx * self.block_size
            if start + self.block_size + 1 > len(self.tokens):
                start = len(self.tokens) - self.block_size - 1
            chunk = self.tokens[start : start + self.block_size + 1]
            dix = torch.tensor(chunk, dtype=torch.long)
            return dix[:-1], dix[1:]

    val_dataset = _SliceDataset(val_tokens, block_size=128)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,
                            batch_size=BATCH_SIZE, num_workers=0)

    total_samples = len(train_dataset)
    iters_for_one_epoch = total_samples // BATCH_SIZE
    print(f"Steps for 1 epoch: {iters_for_one_epoch:,}")

    # Resume from checkpoint if available
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))
    start_iter = 0
    latest_ckpt_path = None

    if checkpoint_files:
        latest_ckpt_path = max(checkpoint_files,
                               key=lambda x: int(re.search(r'ckpt_step_(\d+).pt', x).group(1)))
        start_iter = int(re.search(r'ckpt_step_(\d+).pt', latest_ckpt_path).group(1))
        print(f"Found checkpoint: {latest_ckpt_path}")
        print(f"Resuming from step: {start_iter}")
    else:
        print("No checkpoints found. Starting from step 0.")

    # Model config — 7M
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    print(f"Model: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}")
    model = GPT(model_config)

    if latest_ckpt_path:
        print("Loading weights from checkpoint...")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 0.0006
    train_config.max_iters = iters_for_one_epoch
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                              batch_size=BATCH_SIZE, num_workers=0)

    optimizer = model.configure_optimizers(train_config)
    model.to(train_config.device)
    model.train()

    print(f"Training on {train_config.device} | Steps {start_iter} → {iters_for_one_epoch}")

    for batch_idx, (x, y) in enumerate(train_loader):

        if batch_idx <= start_iter and start_iter > 0:
            continue

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
            print(f"step {batch_idx}/{iters_for_one_epoch} ({percent:.1f}%): "
                  f"loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")

        if batch_idx % 100 == 0 and batch_idx > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_step, (val_x, val_y) in enumerate(val_loader):
                    val_x = val_x.to(train_config.device)
                    val_y = val_y.to(train_config.device)
                    _, val_loss = model(val_x, val_y)
                    val_losses.append(val_loss.item())
                    if val_step >= 20:
                        break
            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f"--- Validation at Step {batch_idx}: avg_val_loss {avg_val_loss:.4f} ---")
            model.train()

        # # Save checkpoint every 500 steps
        # if batch_idx % 500 == 0 and batch_idx > 0:
        #     ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{batch_idx}.pt')
        #     torch.save(model.state_dict(), ckpt_path)
        #     print(f"Checkpoint saved: {ckpt_path}")
    
    # Generate a sample sentence after training
    print("\n--- Sample generation after training ---")
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    prompt = "The history of science"
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(train_config.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50, do_sample=True, top_k=40)
    generated = tokenizer.decode(output[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
    print("---\n")

    final_path = os.path.join(CHECKPOINT_FOLDER_PATH, 'final_model_1_epoch.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")