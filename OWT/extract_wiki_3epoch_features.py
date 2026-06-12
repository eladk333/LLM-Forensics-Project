import os
import sys
import torch
import glob
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import re

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Wikipedia Paths
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki", "wiki_103_full_cache.pt")
CHECKPOINT_BASE_DIR = os.path.join(BASE_PATH, "data", "models")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "wiki_3epoch_features.csv")

sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: minGPT not found.")
    sys.exit(1)

# CRITICAL: Same seed for deterministic batch freezing
set_seed(3407)

CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128}, 
}

# CRITICAL: Same dimensions as baseline experiments
BATCH_SIZE = 64      
BLOCK_SIZE = 128
MAX_BATCHES = 1000 
NUM_WORKERS = 4      

# ==========================================
# 2. DATASET (Wikipedia)
# ==========================================
class WikiDataset(Dataset):
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading Wikipedia tokens from: {cache_path}")
            self.tokens = torch.load(cache_path)
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens):,} tokens.")
        else:
            print(f"❌ Error: Wikipedia Cache not found at {cache_path}")
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
# 3. UTILS & LOGIC
# ==========================================
def get_fixed_batches(loader, max_batches):
    print(f"🔒 Freezing {max_batches} batches of Wikipedia Text...")
    fixed_data = []
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            fixed_data.append(next(iterator))
    except StopIteration:
        pass
    print(f"✅ Locked {len(fixed_data)} batches in RAM.")
    return fixed_data

def get_step_from_filename(filename):
    # Robust regex that extracts the global step number from both standard and epoch formats
    match = re.search(r'ckpt_step_(\d+)(?:_epoch_\d+)?\.pt', filename)
    if match:
        return int(match.group(1))
    return None

def get_all_checkpoints_sorted(folder_path):
    ckpt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    ckpt_files_sorted = sorted(ckpt_files, key=lambda x: get_step_from_filename(x) or -1)
    
    numbered = []
    for f in ckpt_files_sorted:
        step = get_step_from_filename(f)
        if step is not None:
            numbered.append((step, os.path.join(folder_path, f)))
    return numbered

# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================
def extract_all_features(model, fixed_batches, device):
    model.eval()
    
    with torch.no_grad():
        wte_weight = model.transformer.wte.weight
        lm_head_weight = model.lm_head.weight
        
        # 1. Embedding Norm (Global Frobenius of WTE)
        emb_norm = torch.norm(wte_weight, p='fro').item()
        
        # 2. L1 Norm of WTE
        l1_norm = torch.norm(wte_weight, p=1).item()
        
        # 3. Weight Variance of WTE
        weight_variance = torch.var(wte_weight).item()
        
        # 4. Logit Norm (Global Frobenius of LM Head)
        logit_norm = torch.norm(lm_head_weight, p='fro').item()
    
    # 5. Average Probability over frozen Wiki batches
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
                
    avg_probability = np.mean(all_probs) if all_probs else 0.0

    return emb_norm, logit_norm, l1_norm, weight_variance, avg_probability

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running Wiki 3-Epoch Extraction on: {device}")
    
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0: return

    # Using shuffle=True prior to freezing ensures a representative linguistic subset
    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    fixed_batches = get_fixed_batches(loader, MAX_BATCHES)
    
    master_results = []

    for model_size, conf in CONFIGS.items():
        # Matches the folder nomenclature used during your extended Wiki runs
        folder_name = f"{model_size}_wiki_3epoch"
        print(f"\n{'='*40}\n🔎 Processing Wiki: {folder_name}\n{'='*40}")
        
        ckpt_folder = os.path.join(CHECKPOINT_BASE_DIR, folder_name)
        if not os.path.exists(ckpt_folder):
            print(f"⚠️  Folder not found at: {ckpt_folder}. Skipping...")
            continue

        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        model = GPT(model_config).to(device)

        valid_ckpts = get_all_checkpoints_sorted(ckpt_folder)
        print(f"ℹ️  Found {len(valid_ckpts)} checkpoints to analyze.")

        for step, ckpt_path in valid_ckpts:
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                print(f"   ⏳ Extracting Features from Step {step}...", end="\r")
                
                emb_norm, logit_norm, l1_norm, variance, avg_prob = extract_all_features(model, fixed_batches, device)
                
                master_results.append({
                    "Dataset": "Wiki",
                    "Model": f"Model {model_size}",
                    "Step": step,
                    "embedding_norm": emb_norm,
                    "logit_norm": logit_norm,
                    "l1_norm": l1_norm,
                    "weight_variance": variance,
                    "Avg_Probability": avg_prob
                })
                
                print(f"   ✅ Global Step {step:<5} | Emb: {emb_norm:.1f} | L1: {l1_norm:.1f} | Var: {variance:.5f} | Logit: {logit_norm:.1f} | Prob: {avg_prob:.4f}")
                
            except Exception as e:
                print(f"\n   ❌ Error at step {step}: {e}")

    if master_results:
        df = pd.DataFrame(master_results)
        df.sort_values(by=['Model', 'Step'], inplace=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 SUCCESS! Extended Wiki Master dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()