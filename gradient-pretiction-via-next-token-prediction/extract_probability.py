import os
import sys
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
# שנה את הנתיב הזה לנתיב של הפרויקט בשרת שלך!
BASE_PATH = "/home/erez/llm_project" 

# נתיב לקובץ ה-Cache של הטוקנים (WikiText-103)
# חשוב: וודא שזה הקובץ שמכיל את ה-Training Set
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki_103_full_cache.pt")

# הגדרת נתיבי minGPT (אם התיקייה נמצאת בתוך הפרויקט)
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

from mingpt.model import GPT
from mingpt.utils import set_seed

set_seed(3407)

# הגדרות המודלים
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# הגדרות הרצה
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_BATCHES = 150  # המספר מהמסמך שלך (מבטיח סטטיסטיקה טובה)

# ==========================================
# 2. DATASET CLASS
# ==========================================
class WikiDataset(Dataset):
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from: {cache_path}")
            self.tokens = torch.load(cache_path)
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens)} tokens.")
        else:
            print(f"❌ Cache not found at {cache_path}")
            self.tokens = []

    def __len__(self):
        # We don't limit dataset size here, we limit by MAX_BATCHES in the loop
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ==========================================
# 3. CORE LOGIC (The Experiment)
# ==========================================
def calculate_checkpoint_stats(model, loader, device, max_batches=150):
    """
    Calculates the Average Probability assigned to the TRUE next token.
    Methodology: Softmax -> Gather(Ground Truth) -> Mean
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            
            x, y = x.to(device), y.to(device)
            
            # 1. Forward Pass
            logits, _ = model(x)
            
            # 2. Softmax (Logits -> Probabilities)
            probs = F.softmax(logits, dim=-1)
            
            # 3. Gather (Extract probability of the specific True Token)
            # y shape: [B, T] -> y_expanded: [B, T, 1]
            y_expanded = y.unsqueeze(-1)
            correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
            
            # 4. Append Mean of this batch
            all_probs.append(correct_probs.mean().item())
            
    # Return Global Mean
    return np.mean(all_probs)

def get_step_number(filepath):
    try:
        return int(filepath.split('_')[-1].split('.')[0])
    except:
        return -1

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Experiment on: {device}")
    
    # 1. Load Dataset (Once for all models)
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
    
    all_results = []

    # 2. Loop Over Models
    for model_size, conf in CONFIGS.items():
        print(f"\n==========================================")
        print(f"🔎 Processing Model: {model_size}")
        print(f"==========================================")
        
        # Define Checkpoint Path
        ckpt_folder = os.path.join(BASE_PATH, f"MinGPT_Checkpoints_{model_size}")
        if not os.path.exists(ckpt_folder):
            print(f"⚠️ Folder not found: {ckpt_folder}. Skipping.")
            continue

        # Init Model
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        
        model = GPT(model_config)
        model.to(device)

        # Find Checkpoints
        files = glob.glob(os.path.join(ckpt_folder, "*.pt"))
        valid_ckpts = [f for f in files if get_step_number(f) >= 0]
        sorted_ckpts = sorted(valid_ckpts, key=get_step_number)
        
        print(f"   Found {len(sorted_ckpts)} checkpoints.")

        # Loop Over Checkpoints
        for ckpt_path in sorted_ckpts:
            step = get_step_number(ckpt_path)
            
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                # RUN EXPERIMENT
                avg_prob = calculate_checkpoint_stats(model, loader, device, max_batches=MAX_BATCHES)
                
                print(f"   -> Step {step}: Avg Prob = {avg_prob:.5f}")
                
                all_results.append({
                    "Model": f"Model {model_size}",
                    "Step": step,
                    "Avg_Probability": avg_prob
                })
                
            except Exception as e:
                print(f"   ❌ Error reading {ckpt_path}: {e}")

    # 3. Save Final Results
    df = pd.DataFrame(all_results)
    output_csv = os.path.join(BASE_PATH, "probability_results_full.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ SUCCESS! All results saved to: {output_csv}")

if __name__ == "__main__":
    main()