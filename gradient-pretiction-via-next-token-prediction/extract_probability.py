import os
import sys
import torch
import glob
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
BASE_PATH = os.getcwd() 

# Path structure based on your friend's code
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki", "wiki_103_full_cache.pt")
CHECKPOINT_BASE_DIR = os.path.join(BASE_PATH, "data", "models")

# Add minGPT
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: minGPT not found.")
    sys.exit(1)

set_seed(3407)

CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    # '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128}, # Commented out (Colab)
}

BATCH_SIZE = 64      
BLOCK_SIZE = 128
MAX_BATCHES = 200    
NUM_WORKERS = 4      

# ==========================================
# 2. DATASET
# ==========================================
class WikiDataset(Dataset):
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from: {cache_path}")
            self.tokens = torch.load(cache_path)
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens):,} tokens.")
        else:
            print(f"❌ Error: Cache not found at {cache_path}")
            self.tokens = torch.empty(0, dtype=torch.long)

    def __len__(self):
        if len(self.tokens) == 0: return 0
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ==========================================
# 3. LOGIC (FIXED BATCHES + SMART CHECKPOINTS)
# ==========================================
def get_fixed_batches(loader, max_batches):
    print(f"🔒 Freezing {max_batches} batches...")
    fixed_data = []
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            fixed_data.append(next(iterator))
    except StopIteration:
        pass
    print(f"✅ Locked {len(fixed_data)} batches in RAM.")
    return fixed_data

def calculate_stats_on_fixed_data(model, fixed_batches, device):
    model.eval()
    all_probs = []
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            for x, y in fixed_batches:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
                y_expanded = y.unsqueeze(-1)
                correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
                all_probs.append(correct_probs.mean().item())
    if not all_probs: return 0.0
    return np.mean(all_probs)

def get_all_checkpoints_smart(folder_path):
    """
    CRITICAL FIX: Handles 'final_model_1_epoch.pt' correctly.
    Ignores the '1' in '1_epoch' and places final model at the end.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.pt"))
    numbered = []
    final_file = None
    
    for f in all_files:
        fname = os.path.basename(f)
        
        # Check for Final Model
        if "final_model" in fname or "full_epoch" in fname:
            final_file = f
            continue
            
        # Check for Numbered Steps
        parts = fname.replace('.pt', '').split('_')
        for p in parts:
            if p.isdigit():
                # FIX: Ignore '1' if it comes from '1_epoch'
                if p == '1' and 'epoch' in fname: continue
                numbered.append((int(p), f))
                break
    
    numbered.sort(key=lambda x: x[0])
    
    # Assign Final Model to (Max Step + 500)
    if final_file:
        if numbered:
            max_step = numbered[-1][0]
            simulated_step = max_step + 500
        else:
            simulated_step = 500
            
        print(f"ℹ️  Identified Final Model: {os.path.basename(final_file)} -> Step {simulated_step}")
        numbered.append((simulated_step, final_file))
        
    return numbered

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running Experiment on: {device}")
    
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0: return

    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    fixed_batches = get_fixed_batches(loader, MAX_BATCHES)
    
    all_results = []

    for model_size, conf in CONFIGS.items():
        print(f"\n{'='*40}\n🔎 Processing: {model_size}\n{'='*40}")
        
        ckpt_folder = os.path.join(CHECKPOINT_BASE_DIR, model_size)
        
        if not os.path.exists(ckpt_folder):
            print(f"⚠️  Folder not found at: {ckpt_folder}")
            # Optional fallback
            alt_path = os.path.join(BASE_PATH, f"MinGPT_Checkpoints_{model_size}")
            if os.path.exists(alt_path):
                print(f"   -> Found alternative: {alt_path}")
                ckpt_folder = alt_path
            else:
                continue

        # Init Model
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        model = GPT(model_config).to(device)

        # USE SMART DETECTION HERE
        valid_ckpts = get_all_checkpoints_smart(ckpt_folder)
        print(f"ℹ️  Found {len(valid_ckpts)} checkpoints to analyze.")

        for step, ckpt_path in valid_ckpts:
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                print(f"   ⏳ Analyzing Step {step}...", end="\r")
                avg_prob = calculate_stats_on_fixed_data(model, fixed_batches, device)
                print(f"   ✅ Step {step}: Avg Confidence = {avg_prob:.5f}")
                
                all_results.append({"Model": f"Model {model_size}", "Step": step, "Avg_Probability": avg_prob})
            except Exception as e:
                print(f"\n   ❌ Error: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = os.path.join(BASE_PATH, "probability_results_server.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 Saved results to: {output_csv}")

if __name__ == "__main__":
    main()