import os
import sys
import torch
import glob
import re
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ==========================================
# 1. SETUP & PATHS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = CURRENT_DIR

while not os.path.exists(os.path.join(ROOT_DIR, 'minGPT')):
    parent = os.path.dirname(ROOT_DIR)
    if parent == ROOT_DIR: sys.exit(1)
    ROOT_DIR = parent

sys.path.append(os.path.join(ROOT_DIR, 'minGPT'))
try:
    from mingpt.model import GPT
except ImportError:
    sys.exit(1)

# Input files
NORM_CSV = os.path.join(CURRENT_DIR, "gradient_norm_results.csv")
PROB_CSV = os.path.join(CURRENT_DIR, "probability_results_server.csv") # המעודכן שלך
MODELS_ROOT_DIR = os.path.join(ROOT_DIR, "data", "models")
OUTPUT_CSV = os.path.join(CURRENT_DIR, "forensic_merged_features.csv")

CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

def extract_step(filename):
    match = re.search(r'step_(\d+)', filename)
    if match: return int(match.group(1))
    if 'final' in filename: return 999999
    return -1

def get_enhanced_stats(model):
    """ Matches friend's feature set: Embedded Norm, Logit Norm, Variance, L1 """
    # 1. Embedded Layer (wte)
    wte = model.transformer.wte.weight.detach().cpu()
    flat_wte = wte.numpy().flatten()
    
    # 2. De-embedded Layer (lm_head)
    # In minGPT, lm_head usually shares weights with wte, but let's check
    lm_head = model.lm_head.weight.detach().cpu()
    
    return {
        'Embedded_Norm': torch.norm(wte, p=2).item(),
        'Logit_Norm': torch.norm(lm_head, p=2).item(),
        'L1_Norm': torch.norm(wte, p=1).item(),
        'Weight_Variance': np.var(flat_wte),
        'Weight_Skew': skew(flat_wte)
    }

def main():
    print("📂 Loading base data...")
    df_norm = pd.read_csv(NORM_CSV)
    df_prob = pd.read_csv(PROB_CSV)
    
    df_norm['Model_Size'] = df_norm['Model'].astype(str).str.replace('Model ', '')
    df_prob['Model_Size'] = df_prob['Model'].astype(str).str.replace('Model ', '')
    
    df_merged = pd.merge(df_norm, df_prob, on=['Model_Size', 'Step'], how='inner')
    df_merged.rename(columns={'Avg_Probability': 'Next_Token_Prob'}, inplace=True)
    
    new_stats = []
    for size, conf in CONFIGS.items():
        model_dir = os.path.join(MODELS_ROOT_DIR, size)
        relevant_steps = df_merged[df_merged['Model_Size'] == size]['Step'].values
        
        if len(relevant_steps) == 0: continue
            
        print(f"🧪 Enriching {size} with friend's feature set...")
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = 128
        model = GPT(model_config)
        
        for ckpt in glob.glob(os.path.join(model_dir, "*.pt")):
            step = extract_step(os.path.basename(ckpt))
            if step in relevant_steps:
                state_dict = torch.load(ckpt, map_location='cpu')
                model.load_state_dict(state_dict)
                stats = get_enhanced_stats(model)
                stats['Model_Size'], stats['Step'] = size, step
                new_stats.append(stats)

    df_final = pd.merge(df_merged, pd.DataFrame(new_stats), on=['Model_Size', 'Step'], how='inner')
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Enhanced dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()