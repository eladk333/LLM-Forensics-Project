import os
import sys

# Ensure minGPT is in the path
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

# Set permanent seed for deterministic dataset shuffling and initialization
set_seed(3407)

# ==========================================
# CRITICAL UPDATE: Multi-Epoch Configuration
# ==========================================
TOTAL_EPOCHS = 3

# Model architecture settings (124M variant)
N_LAYER = 12
N_HEAD  = 12
N_EMBD  = 768
BATCH_SIZE = 32

# Path configurations targeting original Wiki storage
CHECKPOINT_FOLDER_PATH = os.path.join(os.getcwd(), 'data', 'models', '124M')
DATA_CACHE_PATH = os.path.join(os.getcwd(), 'data', 'wiki')
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True)

class WikiDataset(Dataset):
    def __init__(self, split='train', block_size=128):
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, 'wiki_103_full_cache.pt')

        # Load cached dataset to maintain strict token consistency
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

# Track the lowest loss globally across all resumed sessions
BEST_LOSS = float('inf')

if __name__ == '__main__':
    print("Loading Wikipedia dataset...")
    train_dataset = WikiDataset('train', block_size=128)

    # Compute sequence trajectories based on total epochs required
    total_samples = len(train_dataset)
    iters_for_one_epoch = total_samples // BATCH_SIZE
    total_iters = iters_for_one_epoch * TOTAL_EPOCHS

    # Locate existing checkpoints to establish a resumption anchor
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))
    start_global_step = 0
    latest_ckpt_path = None

    if checkpoint_files:
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(re.search(r'ckpt_step_(\d+).pt', x).group(1)))
        start_global_step = int(re.search(r'ckpt_step_(\d+).pt', latest_ckpt_path).group(1))

    # Dynamically verify if a finalized macro-epoch supersedes standard interval checkpoints
    for completed_epoch in range(1, TOTAL_EPOCHS):
        epoch_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'final_model_{completed_epoch}_epoch.pt')
        if not os.path.exists(epoch_path):
            epoch_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'final_model_{completed_epoch}_epochs.pt')
            
        epoch_boundary_step = (iters_for_one_epoch * completed_epoch) - 1
        
        if os.path.exists(epoch_path) and start_global_step < epoch_boundary_step:
            latest_ckpt_path = epoch_path
            start_global_step = epoch_boundary_step

    if latest_ckpt_path:
        print(f"Resuming from architecture anchor: {latest_ckpt_path}")
        print(f"Synchronized global step timeline starts at: {start_global_step}")
    else:
        print("No historical checkpoints detected. Starting from global baseline 0.")

    # Initialize standard minGPT structure matching forensic requirements
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    print(f"Model setup: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}")
    model = GPT(model_config)

    # Load previously trained spatial weights (Note: Adam moments undergo warm restart)
    if latest_ckpt_path:
        print(f"Loading spatial weights into model...")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 0.0006
    train_config.max_iters = total_iters
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

    print(f"Trajectory bounds set: Step {start_global_step} to {total_iters}")

    # Synchronized temporal loop across multiple physical epochs
    global_step = 0
    for epoch in range(TOTAL_EPOCHS):
        print(f"\n--- Current Processing Pipeline: Epoch {epoch + 1}/{TOTAL_EPOCHS} ---")
        trained_in_this_epoch = False
        
        for batch_idx, (x, y) in enumerate(train_loader):
            
            # Non-destructive fast-forwarding to align physical state with timeline
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

            # Persist intermediate network states without overwriting legacy step integers
            if global_step % 500 == 0 and global_step > 0:
                ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{global_step}.pt')
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved sequentially: {ckpt_path}")

            global_step += 1

        # Macro-state preservation at absolute epoch boundaries
        if trained_in_this_epoch:
            epoch_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'final_model_{epoch + 1}_epochs.pt')
            torch.save(model.state_dict(), epoch_path)
            print(f"*** Saved explicit epoch milestone marker: {epoch_path} ***")

    print(f"Extended sequence training fully completed across {TOTAL_EPOCHS} epochs.")