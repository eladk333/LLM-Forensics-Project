import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION & ARCHITECTURE DATA
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
MATRICES = {
    'Original': os.path.join(CURRENT_DIR, 'step2_master_matrix.csv'),
    'Interaction': os.path.join(CURRENT_DIR, 'step2_interaction_matrix.csv')
}

# Graph directories
DIRS = {
    'Original': os.path.join(CURRENT_DIR, 'micro_graphs-step2_original'),
    'Interaction': os.path.join(CURRENT_DIR, 'micro_graphs-step2_interaction')
}

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}

def setup_graphs_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created graphs directory at: {path}")

def plot_actual_vs_predicted(y_true, y_pred, model_name, test_type, filename, save_dir):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color='royalblue', edgecolors='k', 
                label='Step Prediction')
    
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Ideal Identity Line')
    
    plt.title(f"Architecture: {model_name} | {test_type}", fontsize=14)
    plt.xlabel('Ground Truth (Actual Training Steps)', fontsize=12)
    plt.ylabel('Model Prediction (Estimated Training Steps)', fontsize=12)
    
    stats_text = f"Accuracy Metrics:\n------------------\nR² Score: {r2:.4f}\nMAE: +/- {mae:.1f} steps"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

def get_mlp():
    return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=100, early_stopping=True)

def run_all_micro_tests():
    comparison_results = {'Original': {}, 'Interaction': {}}

    for dataset_name, matrix_path in MATRICES.items():
        if not os.path.exists(matrix_path):
            print(f"❌ Error: Matrix not found at {matrix_path}")
            continue

        print("\n" + "#"*85)
        print(f"{f' RUNNING TESTS ON: {dataset_name.upper()} MATRIX ':^85}")
        print("#"*85)

        save_dir = DIRS[dataset_name]
        setup_graphs_dir(save_dir)
        df = pd.read_csv(matrix_path)
        
        # Dynamically grab features (excluding Model and Step)
        feature_cols = [col for col in df.columns if col not in ['Model', 'Step']]
        bin_cols = [col for col in df.columns if col.startswith('Bin_')]
        models = ['7M', '30M', '124M']

        # TEST 2: GROUPED 80/20 SPLIT
        print("\n--- TEST 2: GROUPED BY CHECKPOINT ---")
        for model_name in models:
            df_model = df[df['Model'] == model_name].copy()
            if df_model.empty: continue
            X, Y = df_model[feature_cols].values, df_model['Step'].values
            groups = df_model['Step'].values
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, Y, groups))
            
            scaler = StandardScaler()
            mlp = get_mlp()
            mlp.fit(scaler.fit_transform(X[train_idx]), Y[train_idx])
            y_pred = mlp.predict(scaler.transform(X[test_idx]))
            
            r2 = r2_score(Y[test_idx], y_pred)
            print(f"{model_name:<10} | R2: {r2:.4f}")
            comparison_results[dataset_name][f'Grouped_{model_name}'] = r2
            
            plot_actual_vs_predicted(Y[test_idx], y_pred, model_name, 
                                     "Clean Internal Evaluation", 
                                     f"test2_clean_grouped_{model_name}.png", save_dir)

        # TEST 5: HONEST CROSS-MODEL
        print("\n--- TEST 5: HONEST CROSS-ARCHITECTURE (124M Zero-Shot) ---")
        df_honest = df.copy()
        for model_name, n_embd in EMBED_DIM_MAP.items():
            mask = df_honest['Model'] == model_name
            if mask.sum() > 0:
                df_honest.loc[mask, bin_cols] = df_honest.loc[mask, bin_cols] / np.sqrt(n_embd)
                
        df_train_h = df_honest[df_honest['Model'].isin(['7M', '30M'])]
        df_test_h = df_honest[df_honest['Model'] == '124M']
        
        if not df_train_h.empty and not df_test_h.empty:
            scaler = StandardScaler()
            mlp_h = get_mlp()
            mlp_h.fit(scaler.fit_transform(df_train_h[feature_cols].values), df_train_h['Step'].values)
            y_pred_h = mlp_h.predict(scaler.transform(df_test_h[feature_cols].values))
            
            r2_h = r2_score(df_test_h['Step'].values, y_pred_h)
            print(f"{'124M (Honest)':<10} | R2: {r2_h:.4f}")
            comparison_results[dataset_name]['ZeroShot_124M'] = r2_h
            
            plot_actual_vs_predicted(df_test_h['Step'].values, y_pred_h, "124M (Zero-Shot)", 
                                     "Train: 7M & 30M -> Predict: 124M", 
                                     "test5_honest_zero_shot_124m.png", save_dir)

    # =====================================================================
    # COMPARISON SUMMARY OUTPUT
    # =====================================================================
    print("\n\n" + "="*85)
    print(f"{' A/B TEST RESULTS SUMMARY (STEP 2: NOISY BATCH DATA) ':^85}")
    print("="*85)
    print(f"{'Metric':<25} | {'Original Setup (R²)':<20} | {'Interaction Setup (R²)':<20}")
    print("-" * 85)
    
    keys = list(comparison_results['Original'].keys())
    for k in keys:
        orig_val = comparison_results['Original'].get(k, 0)
        int_val = comparison_results['Interaction'].get(k, 0)
        winner = "✨ Original" if orig_val > int_val else "✨ Interaction"
        if abs(orig_val - int_val) < 0.001: winner = "Tie"
        print(f"{k:<25} | {orig_val:<20.4f} | {int_val:<20.4f} | Winner: {winner}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_all_micro_tests()