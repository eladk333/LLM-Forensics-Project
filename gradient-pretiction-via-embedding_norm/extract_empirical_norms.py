import torch
import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. SERVER CONFIGURATION & PATHS
# ==========================================
BASE_PATH = os.getcwd() 
WORK_DIR = os.path.join(BASE_PATH, "gradient-pretiction-via-embedding_norm")

CHECKPOINT_BASE_DIR = os.path.join(BASE_PATH, "data", "models")
FREQUENCY_CSV = os.path.join(BASE_PATH, "wiki_token_frequencies.csv")

# Save outputs inside the specific sub-folder
OUTPUT_CSV = os.path.join(WORK_DIR, "embedding_norms_empirical.csv")
PLOTS_DIR = os.path.join(WORK_DIR, "plots_empirical_ml")
os.makedirs(PLOTS_DIR, exist_ok=True)

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
    print(f"\n🔎 Extracting Data for {model_name}...")
    
    if not os.path.exists(folder_path):
        return []

    all_files = glob.glob(os.path.join(folder_path, "*.pt"))
    file_step_map = get_model_step_map(all_files)
    
    num_tokens = len(sorted_token_ids)
    random_indices = np.random.permutation(sorted_token_ids)
    
    folder_data = []

    for file_path, step in file_step_map.items():
        try:
            state_dict = torch.load(file_path, map_location='cpu')
            
            if 'transformer.wte.weight' in state_dict:
                wte = state_dict['transformer.wte.weight']
            elif 'wte.weight' in state_dict:
                wte = state_dict['wte.weight']
            else:
                wte = None
                
            if wte is not None:
                # --- A. Global Frobenius Norm (The 60 original points) ---
                global_norm = torch.norm(wte).item()
                folder_data.append({
                    "Model": model_name, "Step": step, "Method": "Global",
                    "Bin_ID": "-1", "Feature_Value": global_norm
                })

                # --- Calculate per-token norms for Bins ---
                token_norms = torch.norm(wte, p=2, dim=1)
                
                # --- B. Frequency Bins (~3000 samples) ---
                for bin_id, start_idx in enumerate(range(0, num_tokens, BIN_SIZE)):
                    end_idx = min(start_idx + BIN_SIZE, num_tokens)
                    f_bin_ids = sorted_token_ids[start_idx:end_idx]
                    folder_data.append({
                        "Model": model_name, "Step": step, "Method": "Frequency",
                        "Bin_ID": str(bin_id), "Feature_Value": token_norms[f_bin_ids].mean().item()
                    })
                
                # --- C. Random Bins Control Group (~3000 samples) ---
                for bin_id, start_idx in enumerate(range(0, num_tokens, BIN_SIZE)):
                    end_idx = min(start_idx + BIN_SIZE, num_tokens)
                    r_bin_ids = random_indices[start_idx:end_idx]
                    folder_data.append({
                        "Model": model_name, "Step": step, "Method": "Random",
                        "Bin_ID": str(bin_id), "Feature_Value": token_norms[r_bin_ids].mean().item()
                    })

        except Exception as e:
            print(f"   ❌ Error at {os.path.basename(file_path)}: {e}")

    return folder_data

# ==========================================
# 3. MACHINE LEARNING & GRAPHING PIPELINE
# ==========================================
def train_and_evaluate(df, model_name):
    print(f"\n" + "="*50)
    print(f"🧠 ML RESULTS & PLOTS FOR MODEL: {model_name}")
    print("="*50)

    model_df = df[df['Model'] == model_name]

    # ---------------------------------------------------------
    # TASK 1: Cross-Validation & Raw Plot on Global Norm (60 Points)
    # ---------------------------------------------------------
    global_df = model_df[model_df['Method'] == 'Global'].sort_values(by='Step')
    if not global_df.empty:
        X_global = global_df[['Feature_Value']].values
        y_global = global_df['Step'].values
        
        # CV Training
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(LinearRegression(), X_global, y_global, cv=kf, scoring='r2')
        print(f"✅ 1. Global Baseline (CV on {len(global_df)} samples) | R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # PLOT: Separate Raw Global Baseline
        plt.figure(figsize=(8, 6))
        plt.scatter(global_df['Step'], global_df['Feature_Value'], color='black', label='Raw Global Norm')
        
        # Add a simple trendline
        reg = LinearRegression().fit(global_df[['Step']].values, global_df['Feature_Value'].values)
        plt.plot(global_df['Step'], reg.predict(global_df[['Step']].values), color='red', linestyle='--', label='Trendline')
        
        plt.title(f"{model_name} - Raw Global Embedding Norm (The 60 Points)", fontsize=12)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Frobenius Norm", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_1_Raw_Global_Baseline.png"), bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # TASK 2: 80/20 Split & Raw Dynamics on Frequency Bins
    # ---------------------------------------------------------
    freq_df = model_df[model_df['Method'] == 'Frequency']
    if not freq_df.empty:
        X_freq = freq_df[['Feature_Value', 'Bin_ID']].values
        y_freq = freq_df['Step'].values
        
        X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_freq, y_freq, test_size=0.2, random_state=42)
        reg_freq = LinearRegression().fit(X_train_f, y_train_f)
        y_pred_f = reg_freq.predict(X_test_f)
        
        print(f"✅ 2. Frequency Bins ML (80/20 Split) | R²: {r2_score(y_test_f, y_pred_f):.4f} | MAE: {mean_absolute_error(y_test_f, y_pred_f):.2f}")

        # PLOT: ML Predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_f, y_pred_f, alpha=0.6, color='blue', label='Predictions')
        plt.plot([y_test_f.min(), y_test_f.max()], [y_test_f.min(), y_test_f.max()], color='black', linestyle='--', label='Perfect Prediction')
        plt.title(f"{model_name} - Frequency Bins ML: True vs Predicted Steps", fontsize=12)
        plt.xlabel("True Step", fontsize=12)
        plt.ylabel("Predicted Step", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_2_ML_Frequency_Predictions.png"), bbox_inches='tight')
        plt.close()

        # PLOT: Raw Dynamics (The Cloud)
        bin0 = freq_df[freq_df['Bin_ID'] == '0'].sort_values(by='Step')
        bin40 = freq_df[freq_df['Bin_ID'] == '40'].sort_values(by='Step')
        
        plt.figure(figsize=(8, 6))
        plt.scatter(bin0['Step'], bin0['Feature_Value'], color='blue', alpha=0.7, label='Most Frequent (Bin 0)')
        if not bin40.empty:
            plt.scatter(bin40['Step'], bin40['Feature_Value'], color='red', alpha=0.7, label='Rare (Bin 40)')
        plt.title(f"{model_name} - Raw Dynamics: Frequent vs Rare Tokens", fontsize=12)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Average L2 Norm", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_3_Raw_Bin_Dynamics.png"), bbox_inches='tight')
        plt.close()

    # ---------------------------------------------------------
    # TASK 3: 80/20 Split on Random Bins (Control)
    # ---------------------------------------------------------
    rand_df = model_df[model_df['Method'] == 'Random']
    if not rand_df.empty:
        X_rand = rand_df[['Feature_Value', 'Bin_ID']].values
        y_rand = rand_df['Step'].values
        
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rand, y_rand, test_size=0.2, random_state=42)
        reg_rand = LinearRegression().fit(X_train_r, y_train_r)
        y_pred_r = reg_rand.predict(X_test_r)
        
        print(f"✅ 3. Random Bins ML (80/20 Split)  | R²: {r2_score(y_test_r, y_pred_r):.4f} | MAE: {mean_absolute_error(y_test_r, y_pred_r):.2f}")

# ==========================================
# 4. COMBINED GRAPHS (ALL MODELS)
# ==========================================
def generate_combined_graphs(df):
    print("\n🎨 Generating Combined Cross-Architecture Graphs...")
    
    # COMBINED PLOT: The 60 Global Points for all models
    global_df = df[df['Method'] == 'Global']
    
    plt.figure(figsize=(10, 6))
    colors = {'7M': 'blue', '30M': 'green', '124M': 'red'}
    markers = {'7M': 'o', '30M': 's', '124M': '^'}
    
    for model in ['7M', '30M', '124M']:
        m_df = global_df[global_df['Model'] == model].sort_values(by='Step')
        if not m_df.empty:
            # We normalize the values (Min-Max scale) just for the plot so they fit nicely on one axis
            # since 124M matrix norm is naturally much larger than 7M
            min_val = m_df['Feature_Value'].min()
            max_val = m_df['Feature_Value'].max()
            normalized_vals = (m_df['Feature_Value'] - min_val) / (max_val - min_val)
            
            plt.plot(m_df['Step'], normalized_vals, color=colors[model], marker=markers[model], 
                     linestyle='-', alpha=0.7, label=f'Model {model} (Normalized)')

    plt.title("Combined Global Baseline: Embedding Norm vs Training Step", fontsize=14)
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Frobenius Norm (Min-Max Scaled)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    combined_path = os.path.join(PLOTS_DIR, "ALL_MODELS_Combined_Global_Baseline.png")
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved combined graph: {combined_path}")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(FREQUENCY_CSV):
        print("❌ Run 1_generate_frequencies.py first!")
        return

    df_freq = pd.read_csv(FREQUENCY_CSV)
    sorted_token_ids = df_freq['token_id'].tolist()
    
    # Step 1: Extract all features from Checkpoints
    all_results = []
    for name, path in CONFIGS.items():
        all_results.extend(process_folder(name, path, sorted_token_ids))

    if all_results:
        # Step 2: Save to CSV in the sub-directory
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Extraction Success! Data saved to: {OUTPUT_CSV}")
        
        # Step 3: Run ML & Graphing for each model individually
        for name in CONFIGS.keys():
            train_and_evaluate(df_results, name)
            
        # Step 4: Generate the requested combined graphs
        generate_combined_graphs(df_results)
            
        print(f"\n📊 Process Complete. All graphs saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()