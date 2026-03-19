import torch
import os
import glob
import re
import pandas as pd
import numpy as np

# --- ML Imports for Cross Validation ---
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Cross-validation will be skipped.")

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
BASE_PATH = os.getcwd() 
CHECKPOINT_BASE_DIR = os.path.join(BASE_PATH, "data", "models")
FREQUENCY_CSV = os.path.join(BASE_PATH, "wiki_token_frequencies.csv")
OUTPUT_CSV = os.path.join(BASE_PATH, "embedding_norms_empirical.csv")

CONFIGS = {
    '124M': os.path.join(CHECKPOINT_BASE_DIR, '124M'),
    '30M':  os.path.join(CHECKPOINT_BASE_DIR, '30M'),
    '7M':   os.path.join(CHECKPOINT_BASE_DIR, '7M'), 
}

BIN_SIZE = 1000 

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_step_number(filename):
    match = re.search(r'step_(\d+)', filename)
    if match: return int(match.group(1))
    return -1

def get_model_step_map(files):
    step_map = {}
    steps = []
    for f in files:
        basename = os.path.basename(f)
        if "step_" in basename:
            s = extract_step_number(basename)
            if s != -1:
                step_map[f] = s
                steps.append(s)
    
    if not steps: return {} 
    max_step = max(steps)
    steps.sort()
    interval = 500
    if len(steps) > 1:
        interval = steps[1] - steps[0] 
        if interval <= 0: interval = 500

    for f in files:
        if "final_model" in os.path.basename(f) and f not in step_map:
            step_map[f] = max_step + interval
            
    return step_map

def process_folder(model_name, folder_path, sorted_token_ids):
    print(f"\n🔎 Empirical Extraction for {model_name}...")
    
    if not os.path.exists(folder_path):
        return []

    all_files = glob.glob(os.path.join(folder_path, "*.pt"))
    file_step_map = get_model_step_map(all_files)
    
    # Pre-generate random indices for the "Random Bins" approach
    num_tokens = len(sorted_token_ids)
    random_indices = np.random.permutation(sorted_token_ids)
    
    folder_data = []

    for file_path, step in file_step_map.items():
        try:
            state_dict = torch.load(file_path, map_location='cpu')
            wte = state_dict.get('transformer.wte.weight') or state_dict.get('wte.weight')
                
            if wte is not None:
                token_norms = torch.norm(wte, p=2, dim=1)
                
                # --- A. Frequency Bins (The primary hypothesis) ---
                for bin_id, start_idx in enumerate(range(0, num_tokens, BIN_SIZE)):
                    end_idx = min(start_idx + BIN_SIZE, num_tokens)
                    f_bin_ids = sorted_token_ids[start_idx:end_idx]
                    folder_data.append({
                        "Model": model_name, "Step": step, "Method": "Frequency",
                        "Bin_ID": bin_id, "Feature_Value": token_norms[f_bin_ids].mean().item()
                    })
                
                # --- B. Random Bins (The control group) ---
                for bin_id, start_idx in enumerate(range(0, num_tokens, BIN_SIZE)):
                    end_idx = min(start_idx + BIN_SIZE, num_tokens)
                    r_bin_ids = random_indices[start_idx:end_idx]
                    folder_data.append({
                        "Model": model_name, "Step": step, "Method": "Random",
                        "Bin_ID": bin_id, "Feature_Value": token_norms[r_bin_ids].mean().item()
                    })

                # --- C. Statistical Moments (Global Matrix Features) ---
                folder_data.append({"Model": model_name, "Step": step, "Method": "Stat", "Bin_ID": "Mean", "Feature_Value": token_norms.mean().item()})
                folder_data.append({"Model": model_name, "Step": step, "Method": "Stat", "Bin_ID": "Std", "Feature_Value": token_norms.std().item()})
                folder_data.append({"Model": model_name, "Step": step, "Method": "Stat", "Bin_ID": "Max", "Feature_Value": token_norms.max().item()})

        except Exception as e:
            print(f"   ❌ Error at {os.path.basename(file_path)}: {e}")

    return folder_data

# ==========================================
# 3. MAIN
# ==========================================
def main():
    if not os.path.exists(FREQUENCY_CSV):
        print("❌ Run 1_generate_frequencies.py first!")
        return

    df_freq = pd.read_csv(FREQUENCY_CSV)
    sorted_token_ids = df_freq['token_id'].tolist()
    
    all_results = []
    for name, path in CONFIGS.items():
        all_results.extend(process_folder(name, path, sorted_token_ids))

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Success! Empirical data saved to: {OUTPUT_CSV}")

        # --- NEW: Cross-Validation Step ---
        if SKLEARN_AVAILABLE:
            print("\n" + "="*50)
            print("🧠 RUNNING 5-FOLD CROSS-VALIDATION ON BASELINE (60 Points)")
            print("="*50)
            for name in CONFIGS.keys():
                model_data = df_results[(df_results['Model'] == name) & 
                                        (df_results['Method'] == 'Stat') & 
                                        (df_results['Bin_ID'] == 'Mean')]
                if len(model_data) > 0:
                    X = model_data[['Feature_Value']].values
                    y = model_data['Step'].values
                    
                    kf = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(LinearRegression(), X, y, cv=kf, scoring='r2')
                    
                    print(f"✅ Model {name:4s} | Baseline CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            print("="*50 + "\n")

if __name__ == "__main__":
    main()