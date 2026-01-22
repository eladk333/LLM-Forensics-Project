import os
import sys
import torch
import glob
import re
import pandas as pd
import numpy as np

# Set paths based on your Windows environment
ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, "data", "models")
NORM_CSV = "gradient_norm_results.csv"
PROB_CSV = "probability_results_server.csv"
OUTPUT_CSV = "forensic_merged_features.csv"

# Add minGPT path
sys.path.append(os.path.join(ROOT_DIR, 'minGPT'))
from mingpt.model import GPT

CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

def get_step(filename):
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1)) if match else (999999 if 'final' in filename else -1)

def extract_weight_features(model):
    """Extracts features matching the friends' methodology."""
    wte = model.transformer.wte.weight.detach().cpu()
    lm_head = model.lm_head.weight.detach().cpu()
    flat_wte = wte.numpy().flatten()
    
    return {
        'Logit_Norm': torch.norm(lm_head, p=2).item(),
        'L1_Norm': torch.norm(wte, p=1).item(),
        'Weight_Variance': np.var(flat_wte)
    }

def main():
    print("Loading existing experimental data...")
    df_norm = pd.read_csv(NORM_CSV)
    df_prob = pd.read_csv(PROB_CSV)
    
    # Unify model names
    df_norm['Model_Size'] = df_norm['Model'].str.replace('Model ', '')
    df_prob['Model_Size'] = df_prob['Model'].str.replace('Model ', '')
    df_merged = pd.merge(df_norm, df_prob, on=['Model_Size', 'Step'], how='inner')
    
    enriched_results = []
    
    for size, conf in CONFIGS.items():
        model_path = os.path.join(MODELS_DIR, size)
        if not os.path.exists(model_path): continue
        
        print(f"Enriching features for model: {size}")
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer, model_config.n_head, model_config.n_embd = conf['n_layer'], conf['n_head'], conf['n_embd']
        model_config.vocab_size, model_config.block_size = 50257, 128
        model = GPT(model_config)

        target_steps = df_merged[df_merged['Model_Size'] == size]['Step'].tolist()
        
        for ckpt in glob.glob(os.path.join(model_path, "*.pt")):
            step = get_step(os.path.basename(ckpt))
            if step in target_steps:
                model.load_state_dict(torch.load(ckpt, map_location='cpu'))
                stats = extract_weight_features(model)
                stats.update({'Model_Size': size, 'Step': step})
                enriched_results.append(stats)

    final_df = pd.merge(df_merged, pd.DataFrame(enriched_results), on=['Model_Size', 'Step'])
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Merged dataset created: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()