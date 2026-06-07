import os
import sys
import random
import hashlib
import glob
import re

# Go up one directory to find minGPT reference
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'minGPT'))

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

# Set permanent seed for training determinism
set_seed(3407)

# Model configuration architecture constraints for 124M parameter setup
N_LAYER = 12
N_HEAD  = 12
N_EMBD  = 768
BATCH_SIZE = 32

# Multi-epoch target definition
TOTAL_EPOCHS = 3

# Target token count matching absolute size of WikiText-103
TARGET_TOKENS = 117_800_617  

# Direct path tracking mapping
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_FOLDER_PATH = os.path.join(MAIN_DIR, 'data', 'models', '124M_owt')
DATA_CACHE_PATH = os.path.join(MAIN_DIR, 'data', 'datasets')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)

CHAR_SAFETY_FACTOR = 6

def _cache_path():
    key = f"owt_tokens={TARGET_TOKENS}_seed=3407"
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    return os.path.join(DATA_CACHE_PATH, f"owt_cache_{tag}.pt")

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
            assert len(self.tokens) == TARGET_TOKENS, "Cache token count mismatch."
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
            
            if len(all_tokens) < TARGET_TOKENS:
                raise RuntimeError("Insufficient tokens generated.")

            self.tokens = all_tokens[:TARGET_TOKENS]
            torch.save(self.tokens, cache_path)

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
    print("Loading OWT dataset...")
    train_dataset = OWTDataset('train', block_size=128)

    # Calculate baseline mathematical epoch steps
    total_samples = len(train_dataset)
    iters_for_one_epoch = total_samples // BATCH_SIZE
    total_iters = iters_for_one_epoch * TOTAL_EPOCHS

    # Locate latest available step parameters
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))
    start_global_step = 0
    latest_ckpt_path = None

    if checkpoint_files:
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(re.search(r'ckpt_step_(\d+).pt', x).group(1)))
        start_global_step = int(re.search(r'ckpt_step_(\d+).pt', latest_ckpt_path).group(1))

    # Synchronize tracking to resume perfectly from the end of epoch 1 anchor file
    final_1_epoch_path = os.path.join(CHECKPOINT_FOLDER_PATH, 'final_model_1_epoch.pt')
    if os.path.exists(final_1_epoch_path) and start_global_step < (iters_for_one_epoch - 1):
        latest_ckpt_path = final_1_epoch_path
        start_global_step = iters_for_one_epoch - 1

    if latest_ckpt_path:
        print(f"Resuming from architecture anchor: {latest_ckpt_path}")
        print(f"Synchronized global step timeline starts at: {start_global_step}")
    else:
        print("No historical checkpoints detected. Starting from global baseline 0.")

    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    model = GPT(model_config)

    if latest_ckpt_path:
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 0.0006
    train_config.max_iters = total_iters
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=BATCH_SIZE, num_workers=0)

    optimizer = model.configure_optimizers(train_config)
    model.to(train_config.device)
    model.train()

    print(f"Trajectory bounds set: Step {start_global_step} to {total_iters}")

    global_step = 0
    for epoch in range(TOTAL_EPOCHS):
        print(f"\n--- Current Processing Pipeline: Epoch {epoch + 1}/{TOTAL_EPOCHS} ---")
        trained_in_this_epoch = False
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Fast-forward past structural history to prevent dual-training bias
            if global_step <= start_global_step and start_global_step > 0:
                global_step += 1
                continue

            if global_step >= total_iters:
                break

            trained_in_this_epoch = True
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
                percent = (global_step / total_iters) * 100
                print(f"Global Step {global_step}/{total_iters} ({percent:.1f}%): loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")

            # Continuous execution saves strictly synchronized with actual timeline integer scale
            if global_step % 500 == 0 and global_step > 0:
                ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved sequentially: {ckpt_path}")

            global_step += 1

        # Isolate step completion boundaries to append macro milestones safely
        if trained_in_this_epoch:
            epoch_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'final_model_{epoch + 1}_epochs.pt')
            torch.save(model.state_dict(), epoch_path)
            print(f"*** Saved explicit epoch milestone marker: {epoch_path} ***")

    print(f"Extended sequence training fully completed across {TOTAL_EPOCHS} epochs.")