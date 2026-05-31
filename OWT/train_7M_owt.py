import os
import sys
# Go up one directory to find minGPT
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'minGPT'))
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

# Model settings (7M config)
N_LAYER = 4
N_HEAD  = 4
N_EMBD  = 128
BATCH_SIZE = 32
TARGET_TOKENS = 117_800_617 # Exact token count from your Wikipedia run

# Paths - Isolated folder for the new dataset
# Go up one directory to save in the main data folder
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHECKPOINT_FOLDER_PATH = os.path.join(MAIN_DIR, 'data', 'models', '7M_owt') 
DATA_CACHE_PATH = os.path.join(MAIN_DIR, 'data', 'datasets')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True) 

# The dataset class modified for OpenWebText
class OWTDataset(Dataset):
    def __init__(self, split='train', block_size=128, target_tokens=TARGET_TOKENS, seed=3):
        self.block_size = block_size 
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, 'owt_117M_full_cache.pt') 

        # If we already tokenized and saved the dataset, just load it
        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path) 
            print(f"Loaded {len(self.tokens):,} tokens.")
        # If not, extract exactly the target amount from OpenWebText
        else:
            print(f"Cache not found. Loading OpenWebText to extract ~103M tokens...")
            dataset = load_dataset("openwebtext", split="train") 

            # Shuffle document order with fixed seed for perfect reproducibility
            rng = random.Random(seed)
            indices = list(range(len(dataset)))
            rng.shuffle(indices)

            print("Tokenizing data...")
            all_tokens = []
            
            for idx in indices:
                article_text = dataset[idx]['text'].strip()
                if len(article_text) > 0:
                    chunk_tokens = self.tokenizer.encode(article_text + "\n")
                    all_tokens.extend(chunk_tokens)
                
                # Stop when we reach the exact size of Wikipedia
                if len(all_tokens) >= target_tokens:
                    break
            
            # Trim to perfectly match the requested size
            self.tokens = all_tokens[:target_tokens]
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
    print("Loading dataset")
    train_dataset = OWTDataset('train', block_size=128) 

    total_samples = len(train_dataset) 
    iters_for_one_epoch = total_samples // BATCH_SIZE 

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
    train_config.max_iters = iters_for_one_epoch
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Exactly matching your original DataLoader settings
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
            print(f"step {batch_idx}/{iters_for_one_epoch} ({percent:.1f}%): loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")

        # Keeps your original interval for Forensics extraction
        if batch_idx % 500 == 0:
            ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{batch_idx}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(CHECKPOINT_FOLDER_PATH, 'final_model_1_epoch.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")