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
BASE_PATH = "G:/My Drive/LLM_erez_property/" 
DATA_CACHE_PATH = os.path.join(BASE_PATH, "Data/wiki", "wiki_103_full_cache.pt")

sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: minGPT not found.")
    sys.exit(1)

# Fixed seed ensures the "Random Shuffle" is exactly the same every time we run the script
set_seed(3407)

CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    # '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128}, 
}

BATCH_SIZE = 64      
BLOCK_SIZE = 128
MAX_BATCHES = 200    
NUM_WORKERS = 4      

# ==========================================
# 2. DATASET CLASS
# ==========================================
class WikiDataset(Dataset):
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from cache: {cache_path}")
            self.tokens = torch.load(cache_path)
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens):,} tokens into memory.")
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
# 3. CORE LOGIC
# ==========================================
def get_fixed_batches(loader, max_batches):
    """
    Extracts a FIXED list of batches from the loader.
    This ensures every model/checkpoint sees exactly the same data.
    """
    print(f"🔒 Freezing {max_batches} batches for consistent evaluation...")
    fixed_data = []
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            batch = next(iterator)
            fixed_data.append(batch)
    except StopIteration:
        pass
    print(f"✅ Locked {len(fixed_data)} batches in memory.")
    return fixed_data

def calculate_stats_on_fixed_data(model, fixed_batches, device):
    """
    Runs inference on the pre-loaded fixed batches.
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            # Iterate over the LIST of batches, not the loader
            for x, y in fixed_batches:
                x, y = x.to(device), y.to(device)
                
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
                
                y_expanded = y.unsqueeze(-1)
                correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
                all_probs.append(correct_probs.mean().item())
            
    if len(all_probs) == 0: return 0.0
    return np.mean(all_probs)

def get_step_number(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.replace('.pt', '').split('_')
        for part in parts:
            if part.isdigit(): return int(part)
        return -1
    except: return -1

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running Experiment on: {device}")
    
    # 1. Initialize Dataset & Loader
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0: return

    # Shuffle is TRUE here, but we will run it ONCE to get the fixed batches
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    # 2. FREEZE THE DATA (The Critical Step)
    fixed_batches = get_fixed_batches(loader, MAX_BATCHES)
    
    all_results = []

    # 3. Iterate Models
    for model_size, conf in CONFIGS.items():
        print(f"\n{'='*40}\n🔎 Model: {model_size}\n{'='*40}")
        
        ckpt_folder = os.path.join(BASE_PATH, f"MinGPT_Checkpoints_{model_size}")
        if not os.path.exists(ckpt_folder): continue

        # Init Model
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        
        model = GPT(model_config).to(device)

        # Checkpoints
        files = glob.glob(os.path.join(ckpt_folder, "*.pt"))
        valid_ckpts = sorted([f for f in files if get_step_number(f) >= 0], key=get_step_number)
        
        print(f"ℹ️  Found {len(valid_ckpts)} checkpoints.")

        for ckpt_path in valid_ckpts:
            step = get_step_number(ckpt_path)
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                print(f"   ⏳ Analyzing Step {step}...", end="\r")
                
                # USE FIXED DATA
                avg_prob = calculate_stats_on_fixed_data(model, fixed_batches, device)
                
                print(f"   ✅ Step {step}: Avg Confidence = {avg_prob:.5f}")
                all_results.append({"Model": f"Model {model_size}", "Step": step, "Avg_Probability": avg_prob})
                
            except Exception as e:
                print(f"\n   ❌ Error: {e}")

    # Save
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(BASE_PATH, "probability_results_server.csv"), index=False)
        print("\n🎉 Done!")

if __name__ == "__main__":
    main()