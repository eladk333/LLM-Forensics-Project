import os
import torch
import glob
import re
import sys
sys.path.append('/home/nlp/katzela4/LLM-Forensics-Project/minGPT')
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed

set_seed(3407)

# --- CONFIGURATION ---
N_LAYER = 12   
N_HEAD  = 16   
N_EMBD  = 1024 
BATCH_SIZE = 128  # Increased for A100 80GB
BLOCK_SIZE = 128
TARGET_TOKENS = 420_000_000

# USE ABSOLUTE PATH TO BE SAFE
DATA_FILE_PATH = '/home/nlp/katzela4/LLM-Forensics-Project/data/datasets/owt_420M_cache.pt'
CHECKPOINT_FOLDER_PATH = '/home/nlp/katzela4/LLM-Forensics-Project/data/models/200M_owt_420M'
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)

class OpenWebTextDataset(Dataset):
    def __init__(self, block_size=128):
        self.block_size = block_size
        if os.path.exists(DATA_FILE_PATH):
            print(f"Loading cached tokens from {DATA_FILE_PATH}...")
            self.tokens = torch.load(DATA_FILE_PATH)
        else:
            raise FileNotFoundError(f"Could not find {DATA_FILE_PATH}. Check your path!")

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        if start_idx + self.block_size + 1 > len(self.tokens):
            start_idx = len(self.tokens) - self.block_size - 1
        chunk = self.tokens[start_idx : start_idx + self.block_size + 1]
        dix = torch.tensor(chunk, dtype=torch.long)
        return dix[:-1], dix[1:]

if __name__ == '__main__':
    # 1. Load Data
    full_dataset = OpenWebTextDataset(block_size=BLOCK_SIZE)
    
    # 2. Split Data (99% Train, 1% Val)
    val_split_idx = int(len(full_dataset.tokens) * 0.99)
    train_tokens = full_dataset.tokens[:val_split_idx]
    val_tokens = full_dataset.tokens[val_split_idx:]
    
    # Update main dataset to only be train
    full_dataset.tokens = train_tokens
    
    # Manual helper for val slice
    class SliceDataset(Dataset):
        def __init__(self, tokens, b_size):
            self.tokens = tokens
            self.b_size = b_size
        def __len__(self): return len(self.tokens) // self.b_size
        def __getitem__(self, idx):
            s = idx * self.b_size
            if s + self.b_size + 1 > len(self.tokens): s = len(self.tokens) - self.b_size - 1
            d = torch.tensor(self.tokens[s : s + self.b_size + 1], dtype=torch.long)
            return d[:-1], d[1:]

    val_dataset = SliceDataset(val_tokens, BLOCK_SIZE)
    
    # 3. Model Setup
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = BLOCK_SIZE
    
    model = GPT(model_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # MOVE TO DEVICE BEFORE OPTIMIZER
    model.to(device)

    # 4. Trainer Setup
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 6e-4
    train_config.max_iters = len(full_dataset) // BATCH_SIZE
    train_config.batch_size = BATCH_SIZE
    train_config.device = device

    optimizer = model.configure_optimizers(train_config)
    train_loader = DataLoader(full_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # 5. Training Loop
    print(f"Starting Training: 203M Model | Device: {device}")
    model.train()
    best_loss = float('inf')

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f"Step {batch_idx}: Loss {loss.item():.4f}")

        # # Save Checkpoint Logic
        # if batch_idx % 500 == 0 and batch_idx > 0:
        #     torch.save(model.state_dict(), os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_{batch_idx}.pt'))