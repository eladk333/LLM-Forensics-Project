import os
import sys
import torch
import glob
import re
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ==========================================
# 1. SETUP & PATHS (ADAPTED TO YOUR LS -R)
# ==========================================
BASE_PATH = os.getcwd() # Should be LLM-Forensics-Project

# Add minGPT to path
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: Could not import minGPT. Ensure you are running from 'LLM-Forensics-Project' root.")
    sys.exit(1)

# Device Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 128
BATCH_SIZE = 32   
MAX_BATCHES = 50  # Sampling for speed

# Paths based on your ls -R
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki", "wiki_103_full_cache.pt")
MODELS_ROOT_DIR = os.path.join(BASE_PATH, "data", "models")

# Configs matching your folders
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_step(filename):
    """Extract step number from filename like 'ckpt_step_1000.pt'"""
    match = re.search(r'step_(\d+)', filename)
    if match: return int(match.group(1))
    if 'final' in filename: return 999999
    return -1

def get_advanced_weight_stats(model):
    """ COMBINED LOGIC: Yours (Norm) + Friend's (Skew/Kurtosis) """
    wte = model.transformer.wte.weight.detach().cpu()
    
    # 1. Your Feature: Norm
    norm = torch.norm(wte, p=2).item()
    
    # 2. Friend's Features: Distribution shape
    flat_wte = wte.numpy().flatten()
    var = np.var(flat_wte)
    mean = np.mean(flat_wte)
    skw = skew(flat_wte)
    krt = kurtosis(flat_wte)
    
    return norm, var, mean, skw, krt

def calculate_avg_probability(model, x_batches, y_batches):
    """ YOUR LOGIC: Dynamic Next-Token Probability """
    model.eval()
    total_prob = 0
    count = 0
    
    with torch.no_grad():
        for x, y in zip(x_batches, y_batches):
            logits, _ = model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            probs = torch.softmax(shift_logits, dim=-1)
            gathered = torch.gather(probs, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            total_prob += gathered.mean().item()
            count += 1
            
    return total_prob / count if count > 0 else 0

# ==========================================
# 3. MAIN
# ==========================================
def main():
    print(f"🔌 Running on: {DEVICE}")
    print(f"📂 Root Path: {BASE_PATH}")
    
    # 1. LOAD WIKITEXT
    if not os.path.exists(DATA_CACHE_PATH):
        print(f"❌ Error: Data file not found at {DATA_CACHE_PATH}")
        return
    
    print("⏳ Loading WikiText Cache...")
    data_tensor = torch.load(DATA_CACHE_PATH)
    n_data = len(data_tensor)
    
    # Pre-generate fixed batches
    fixed_x, fixed_y = [], []
    for i in range(MAX_BATCHES):
        ix = torch.randint(n_data - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data_tensor[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data_tensor[i+1:i+BLOCK_SIZE+1] for i in ix])
        fixed_x.append(x.to(DEVICE))
        fixed_y.append(y.to(DEVICE))
        
    all_data = []

    # 2. ITERATE MODELS
    for size, conf in CONFIGS.items():
        # Correct path based on your LS -R: ./data/models/7M
        model_dir = os.path.join(MODELS_ROOT_DIR, size)
        
        if not os.path.exists(model_dir):
            print(f"⚠️ Skipping {size} (Folder not found: {model_dir})")
            continue
            
        print(f"\n🔍 Processing {size} checkpoints...")
        
        # Init Model
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        model = GPT(model_config).to(DEVICE)
        
        checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
        print(f"   Found {len(checkpoints)} files.")
        
        for ckpt in checkpoints:
            step = extract_step(os.path.basename(ckpt))
            if step == -1: continue
            
            try:
                state_dict = torch.load(ckpt, map_location=DEVICE)
                model.load_state_dict(state_dict)
                
                # A. Static Features (Yours + Friend's)
                norm, var, mean, skw, krt = get_advanced_weight_stats(model)
                
                # B. Dynamic Features (Yours)
                prob = calculate_avg_probability(model, fixed_x, fixed_y)
                
                print(f"   Step {step:<6} | Norm: {norm:.2f} | Skew: {skw:.2f} | Prob: {prob:.4f}")
                
                all_data.append({
                    "Model_Size": size,
                    "Step": step,
                    "Embedding_Norm": norm,
                    "Weight_Variance": var,
                    "Weight_Skew": skw,
                    "Weight_Kurtosis": krt,
                    "Avg_Next_Token_Prob": prob
                })
                
            except Exception as e:
                print(f"   ❌ Error {os.path.basename(ckpt)}: {e}")

    # 3. SAVE
    if all_data:
        df = pd.DataFrame(all_data)
        df.sort_values(by=["Model_Size", "Step"], inplace=True)
        output_csv = os.path.join(BASE_PATH, "forensic_merged_features.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n✅ SUCCESS! Data saved to: {output_csv}")
    else:
        print("\n❌ No data extracted.")

if __name__ == "__main__":
    main()