import os
import sys
import torch
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Define paths relative to the current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Add minGPT directory to sys.path to import the architecture
sys.path.append(os.path.join(PARENT_DIR, 'minGPT'))
from mingpt.model import GPT

MODELS_DIR = os.path.join(PARENT_DIR, 'data', 'models')
OUTPUT_CSV = os.path.join(CURRENT_DIR, 'extracted_rest_features.csv')
EDA_DIR = os.path.join(CURRENT_DIR, 'eda_graphs')

# Create directory for the EDA graphs if it doesn't exist
os.makedirs(EDA_DIR, exist_ok=True)

# Model architecture configurations (required to initialize an empty model)
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

def get_step_from_filename(filename):
    """
    Extracts the step number from the checkpoint filename using regex.
    """
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    if 'final' in filename:
        return 29000  # Default to the last step of epoch 1 based on previous data
    return None

def plot_eda_graphs(df):
    """
    Generates Exploratory Data Analysis (EDA) scatter plots for each extracted feature.
    Saves the plots in the designated EDA directory.
    """
    print(f"\n📊 Generating EDA graphs in: {EDA_DIR}")
    
    features_to_plot = ['l1_norm', 'weight_variance', 'logit_norm']
    colors = {'Model 7M': 'red', 'Model 30M': 'green', 'Model 124M': 'blue'}
    
    for feature in features_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Plot each model's data points with a specific color
        for model_name, color in colors.items():
            subset = df[df['Model'] == model_name]
            if not subset.empty:
                plt.scatter(subset[feature], subset['Step'], label=model_name, color=color, alpha=0.7)
        
        plt.title(f"EDA: {feature} vs. Gradient Updates")
        plt.xlabel(f"{feature} Value")
        plt.ylabel("Gradient Updates (Steps)")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the figure
        plot_path = os.path.join(EDA_DIR, f"eda_{feature}_vs_steps.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
    print("✅ EDA graphs generated successfully.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Starting extraction using device: {device}")
    
    results = []

    # Iterate over the 3 architectures
    for model_size in ['7M', '30M', '124M']:
        model_path = os.path.join(MODELS_DIR, model_size)
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Directory not found -> {model_path}")
            continue

        print(f"\n========================================")
        print(f"📂 Processing Architecture: {model_size}")
        print(f"========================================")

        # Create an empty model based on the exact configuration
        conf = CONFIGS[model_size]
        m_conf = GPT.get_default_config()
        m_conf.model_type = None
        m_conf.n_layer = conf['n_layer']
        m_conf.n_head = conf['n_head']
        m_conf.n_embd = conf['n_embd']
        m_conf.vocab_size = 50257
        m_conf.block_size = 128
        
        model = GPT(m_conf).to(device)
        model.eval()

        # Scan checkpoint files and sort by step number
        ckpt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        ckpt_files_sorted = sorted(ckpt_files, key=lambda x: get_step_from_filename(x) or -1)

        for fname in ckpt_files_sorted:
            step = get_step_from_filename(fname)
            if step is None:
                continue
                
            ckpt_full_path = os.path.join(model_path, fname)
            
            try:
                # Load the weights (state_dict) from the file
                state_dict = torch.load(ckpt_full_path, map_location=device)
                model.load_state_dict(state_dict)
                
                with torch.no_grad():
                    # --- Calculate the 3 new features ---
                    
                    # 1. Input embedding matrix
                    wte_weight = model.transformer.wte.weight
                    
                    # L1 Norm of the input embeddings
                    l1_norm = torch.norm(wte_weight, p=1).item()
                    
                    # Weight Variance of the input embeddings
                    weight_var = torch.var(wte_weight).item()
                    
                    # 2. Output un-embedding matrix (LM Head)
                    lm_head_weight = model.lm_head.weight
                    
                    # Logit Norm (Global L2 / Frobenius norm of the output matrix)
                    logit_norm = torch.norm(lm_head_weight, p='fro').item()

                print(f"  ✅ [Step {step:5d}] Extracted -> L1: {l1_norm:.1f} | Var: {weight_var:.5f} | Logit Norm: {logit_norm:.1f}")
                
                results.append({
                    'Model': f"Model {model_size}",
                    'Step': step,
                    'l1_norm': l1_norm,
                    'weight_variance': weight_var,
                    'logit_norm': logit_norm
                })
                
            except Exception as e:
                print(f"  ❌ Error processing {fname}: {e}")

    # Save results and generate plots
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Success! Missing features saved to:\n{OUTPUT_CSV}")
        
        # Call the plotting function
        plot_eda_graphs(df)
    else:
        print("\n⚠️ No features were extracted. Check your file paths.")

if __name__ == "__main__":
    main()