import os
import sys
import torch
import glob
import re
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Assuming this script runs inside 'combine_features/'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Add minGPT to path
sys.path.append(os.path.join(ROOT_DIR, 'minGPT'))

try:
    from mingpt.model import GPT
except ImportError:
    print("❌ Error: minGPT module not found. Check your folder structure.")
    sys.exit(1)

MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "forensic_features.csv")

# Model Architectures
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# ==========================================
# 2. FEATURE EXTRACTION LOGIC
# ==========================================
def get_step_from_filename(filename):
    """Extracts step number from checkpoint filename."""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    if 'final' in filename:
        return 999999 # Place final model at the end
    return -1

def extract_global_stats(model):
    """
    Extracts global statistical features from the model's embedding matrix.
    Based on the logic from the original extract_features.py but applied globally.
    """
    # 1. Embedding Weights stats
    wte = model.transformer.wte.weight.detach().cpu().numpy()
    flat_wte = wte.flatten()
    
    # 2. Output Head stats (Logit Norm)
    lm_head = model.lm_head.weight.detach().cpu().numpy()
    
    return {
        'embedding_norm': np.linalg.norm(wte),          # Total L2 Norm
        'l1_norm':        np.linalg.norm(wte, ord=1),   # Total L1 Norm
        'weight_variance': np.var(flat_wte),            # Global Variance
        'weight_skew':    skew(flat_wte),               # Skewness
        'weight_kurtosis': kurtosis(flat_wte),          # Kurtosis
        'logit_norm':     np.linalg.norm(lm_head)       # Logit/De-embedding Norm
    }

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    print(f"🚀 Starting Forensic Feature Extraction...")
    print(f"📂 Reading models from: {MODELS_DIR}")
    
    all_records = []

    for size, conf in CONFIGS.items():
        model_folder = os.path.join(MODELS_DIR, size)
        if not os.path.exists(model_folder):
            print(f"⚠️  Skipping {size} (Folder not found)")
            continue
            
        print(f"⚙️  Processing {size}...")
        
        # Initialize Architecture
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = 128
        model = GPT(model_config)

        # Iterate Checkpoints
        checkpoints = glob.glob(os.path.join(model_folder, "*.pt"))
        print(f"   Found {len(checkpoints)} checkpoints.")

        for i, ckpt_path in enumerate(checkpoints):
            step = get_step_from_filename(os.path.basename(ckpt_path))
            if step == -1: continue

            if i % 10 == 0: print(f"   ⏳ Scanning Step {step}...", end='\r')

            try:
                state_dict = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(state_dict)
                
                features = extract_global_stats(model)
                features['Model'] = f"Model {size}" # Naming convention: "Model 7M"
                features['Step'] = step
                all_records.append(features)

            except Exception as e:
                print(f"\n   ❌ Failed to load {os.path.basename(ckpt_path)}: {e}")

        print(f"\n   ✅ Finished {size}.")

    # Save Results
    if all_records:
        df = pd.DataFrame(all_records)
        df.sort_values(by=['Model', 'Step'], inplace=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Success! Forensic features saved to: {OUTPUT_CSV}")
    else:
        print("\n❌ No data extracted. Check your model paths.")

if __name__ == "__main__":
    main()