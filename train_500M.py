import os
import sys
import math

# --- RESTRICT TO TWO GPUs ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# Dynamic path: Adds the current folder's 'minGPT' subdirectory to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'minGPT'))

import torch
import torch.nn.functional as F
import glob
import re
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT, CausalSelfAttention
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
import mingpt.model

# --- 1. MEMORY PATCH: FLASH ATTENTION ---
# Bypasses the O(T^2) memory bottleneck, preventing GPU 0 from OOMing
def flash_attention_forward(self, x):
    B, T, C = x.size()
    
    # Calculate query, key, values for all heads in batch
    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

    # 1. Safely find the attention dropout probability
    attn_pdrop = 0.1 # Standard minGPT default fallback
    if hasattr(self, 'attn_drop'): attn_pdrop = self.attn_drop.p
    elif hasattr(self, 'attn_dropout'): attn_pdrop = self.attn_dropout.p

    # Native PyTorch 2.0+ Flash Attention
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=None, 
        dropout_p=attn_pdrop if self.training else 0.0, 
        is_causal=True
    )
    
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    
    # 2. Output projection
    y = self.c_proj(y)
    
    # 3. Safely apply the residual dropout
    if hasattr(self, 'resid_drop'): 
        y = self.resid_drop(y)
    elif hasattr(self, 'resid_dropout'): 
        y = self.resid_dropout(y)
        
    return y

mingpt.model.CausalSelfAttention.forward = flash_attention_forward

# --- 2. MULTI-GPU BLOCK PATCH ---
class MultiGPUBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = mingpt.model.CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            mingpt.model.NewGELU(),
            torch.nn.Linear(4 * config.n_embd, config.n_embd),
            torch.nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

mingpt.model.Block = MultiGPUBlock

# --- 3. CONFIGURATION ---
set_seed(3407)

NUM_EPOCHS = 1

# Model Specs (~535M Parameters)
N_LAYER = 24    
N_HEAD  = 16    
N_EMBD  = 1280  
BLOCK_SIZE = 1024 

# Gradient Accumulation Logic
MICRO_BATCH_SIZE = 16  # Fits safely in VRAM across 2 GPUs
ACCUM_STEPS = 4        # 16 * 4 = 64 Effective Batch Size

# Server Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'data', 'models', '500M_Context1024_G')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 4. DATASET ---
class LargeTextDataset(Dataset):
    def __init__(self, split='train', block_size=1024):
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading OpenWebText ({split}) lazily to save RAM...")
        self.dataset = load_dataset("openwebtext", split=split)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.block_size + 1, 
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        
        return tokens[:-1], tokens[1:]

# --- HELPER: FIND LATEST CHECKPOINT ---
def get_latest_checkpoint(ckpt_dir):
    list_of_files = glob.glob(os.path.join(ckpt_dir, '*.pt'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getmtime)

if __name__ == '__main__':
    print(f"Running on Server Mode.")
    print(f"Checkpoints:  {CHECKPOINT_DIR}")

    # 1. RESUME LOGIC & METADATA
    latest_ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
    start_epoch = 0
    start_step = -1
    BEST_LOSS = float('inf')
    checkpoint_data = None
    is_legacy_ckpt = False

    if latest_ckpt_path:
        print(f"✅ Found latest checkpoint: {latest_ckpt_path}")
        checkpoint_data = torch.load(latest_ckpt_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0)
            start_step = checkpoint_data.get('step', -1)
            BEST_LOSS = checkpoint_data.get('best_loss', float('inf'))
            print(f"   Resuming from Epoch {start_epoch+1}, Step {start_step+1}.")
        else:
            is_legacy_ckpt = True
            match = re.search(r'epoch_(\d+)_step_(\d+)', latest_ckpt_path)
            if match:
                start_epoch = int(match.group(1)) - 1  
                start_step = int(match.group(2))
            print(f"   [Legacy Checkpoint Detected] Guessing Epoch {start_epoch+1}, Step {start_step}.")
    else:
        print("   Starting training from SCRATCH.")

    # 2. SETUP DATA (With Deterministic Seeding)
    train_dataset = LargeTextDataset('train', BLOCK_SIZE)
    
    g = torch.Generator()
    g.manual_seed(3407 + start_epoch) # Ensures shuffling is identical if resuming
    
    train_loader = DataLoader(
        train_dataset, 
        shuffle=True, 
        pin_memory=True, 
        batch_size=MICRO_BATCH_SIZE, 
        num_workers=4, # Reduced to prevent CPU thrashing
        generator=g
    )
    
    # 3. SETUP MODEL
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER; model_config.n_head = N_HEAD; model_config.n_embd = N_EMBD
    model_config.vocab_size = 50257; model_config.block_size = BLOCK_SIZE
    model = GPT(model_config)

    if checkpoint_data:
        if not is_legacy_ckpt:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)

    # 4. MULTI-GPU & AMP SETUP
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_count = torch.cuda.device_count()
    print(f"🚀 Hardware: {gpu_count} GPUs detected.")
    
    model.to(device)
    if gpu_count > 1:
        print(f"   Activating DataParallel on {gpu_count} GPUs!")
        model = torch.nn.DataParallel(model)

    model.train()
    
    optimizer = model.module.configure_optimizers(Trainer.get_default_config()) if hasattr(model, "module") else model.configure_optimizers(Trainer.get_default_config())
    scaler = torch.amp.GradScaler('cuda')

    if checkpoint_data and not is_legacy_ckpt:
        if 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print("   Optimizer state loaded.")
        if 'scaler_state_dict' in checkpoint_data:
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            print("   AMP Scaler state loaded.")
            
    del checkpoint_data
    torch.cuda.empty_cache()

    # 5. TRAINING LOOP
    print(f"\nStarting Training: Epoch {start_epoch+1} to {NUM_EPOCHS}")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ===")
        
        # Reset generator seed per epoch
        g.manual_seed(3407 + epoch)
        
        # Initial zero grad before loop starts
        model.zero_grad(set_to_none=True)
        
        for batch_idx, (x, y) in enumerate(train_loader):
            
            if epoch == start_epoch and batch_idx <= start_step:
                if batch_idx % 100 == 0:
                    print(f"Fast-forwarding... Skipping batch {batch_idx}/{start_step}", end="\r")
                continue

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
                if gpu_count > 1: 
                    loss = loss.mean()
                
                # Scale loss down for accumulation
                loss = loss / ACCUM_STEPS

            # Backward pass
            scaler.scale(loss).backward()

            # Optimizer Step logic (only step when accumulation target is reached)
            if ((batch_idx + 1) % ACCUM_STEPS == 0) or (batch_idx + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)

            # Display un-scaled loss for accurate tracking
            real_loss = loss.item() * ACCUM_STEPS
            if real_loss < BEST_LOSS: 
                BEST_LOSS = real_loss

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {real_loss:.4f}", end="\r")

            # Save Safety Checkpoint every 2000 steps
            if batch_idx % 2000 == 0 and batch_idx > 0:
                safe_path = os.path.join(CHECKPOINT_DIR, f'ckpt_epoch_{epoch+1}_step_{batch_idx}.pt')
                save_model = model.module if hasattr(model, "module") else model
                
                torch.save({
                    'epoch': epoch,
                    'step': batch_idx,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': BEST_LOSS
                }, safe_path)

        # SAVE EPOCH MODEL
        final_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        save_model = model.module if hasattr(model, "module") else model
        
        torch.save({
            'epoch': epoch,
            'step': batch_idx,
            'model_state_dict': save_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': BEST_LOSS
        }, final_path)
        print(f"\n✅ SAVED: {final_path}")