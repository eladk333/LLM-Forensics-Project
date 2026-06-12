import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import math
import glob
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


# We set const seed so the training will be deterministic
set_seed(3407)

# Model settings (Currently set to 124M)
N_LAYER = 6  # Increased from 6
N_HEAD  = 6   # Increased from 6
N_EMBD  = 384  # Increased from 384
BATCH_SIZE = 32 # Same for all models

# Paths - Updated to isolate the 3-epoch run
CHECKPOINT_FOLDER_PATH = os.path.join(os.getcwd(), 'data', 'models', '30M_wiki_3epoch') 
DATA_CACHE_PATH = os.path.join(os.getcwd(), 'data', 'wiki')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True) 

# The dataset class
class WikiDataset(Dataset):
    def __init__(self, split='train', block_size=128):
        self.block_size = block_size 
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, 'wiki_103_full_cache.pt') 

        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path) 
            print(f"Loaded {len(self.tokens)} tokens.")
        else:
            print(f"Cache not found. Loading WikiText-103 ({split})...")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=split) 

            print("Tokenizing data...")
            cleaned_articles = [] 

            for x in dataset:
                article_text = x['text'] 
                if len(article_text) > 0:
                    cleaned_articles.append(article_text)
            text_data = "\n".join(cleaned_articles) 
            self.tokens = self.tokenizer.encode(text_data) 
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
    train_dataset = WikiDataset('train', block_size=128) 

    total_samples = len(train_dataset) 
    iters_for_one_epoch = total_samples // BATCH_SIZE 
    
    # NEW: Calculate total iterations for 3 epochs
    max_total_iters = iters_for_one_epoch * 3

    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))

    start_global_step = 0
    latest_ckpt_path = None

    if checkpoint_files:
        # NEW: Robust regex extraction to handle 'ckpt_step_X.pt' and 'ckpt_step_X_epoch_Y.pt'
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
    train_config.max_iters = max_total_iters  # NEW: Target is now 3 epochs worth of steps
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

    # NEW: Global step tracking
    global_step = 0

    # NEW: Outer loop for 3 epochs
    for epoch in range(1, 4):
        print(f"\n--- Starting Epoch {epoch}/3 ---")
        
        for batch_idx, (x, y) in enumerate(train_loader):

            # Strict barrier: stop exactly when 1 epoch ends
            if batch_idx >= iters_for_one_epoch:
                break
                
            global_step += 1

            # Resume logic using global_step
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

            # Condition A: Standard checkpoint every 500 steps (excludes exact epoch boundaries)
            if global_step % 500 == 0 and batch_idx != (iters_for_one_epoch - 1):
                ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f"Standard checkpoint saved: {ckpt_path}")

        # Condition B: End of Epoch checkpoint
        if global_step == iters_for_one_epoch * epoch:
            epoch_ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}_epoch_{epoch}.pt')
            torch.save(model.state_dict(), epoch_ckpt_path)
            print(f"Epoch completion checkpoint saved: {epoch_ckpt_path}")

    print(f"\nTraining completed successfully for 3 full epochs.")