import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Safe backend for saving plots without GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MATRICES = {
    'Original': os.path.join(CURRENT_DIR, 'step2s_master_matrix.csv'),
    'Interaction': os.path.join(CURRENT_DIR, 'step2s_interaction_matrix.csv')
}

DIRS = {
    'Original': os.path.join(CURRENT_DIR, 'micro_graphs-step2s_original'),
    'Interaction': os.path.join(CURRENT_DIR, 'micro_graphs-step2s_interaction')
}

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}

def setup_graphs_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created graphs directory at: {path}")

def get_algorithms():
    return {
        'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=False),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name, algo_name, test_type, filename, save_dir):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkorange', edgecolors='k', s=60, label=f'{algo_name} Prediction')
    
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Accuracy')
    
    plt.title(f"Architecture: {model_name} | Algo: {algo_name}\n{test_type}", fontsize=13)
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

def run_step2s_tests():
    # Store results for final summary [Dataset][Algo][Metric] = R2
    summary = {}

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
        
        feature_cols = [col for col in df.columns if col not in ['Model', 'Step']]
        bin_cols = [col for col in df.columns if col.startswith('Bin_')]
        models = ['7M', '30M', '124M']

        # -----------------------------------------------------------------
        # TEST 2b: 5-FOLD CV EDITION
        # -----------------------------------------------------------------
        print("\n--- TEST 2b: 5-FOLD CV EDITION ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # We will track 124M CV results for summary
        for model_name in models:
            df_model = df[df['Model'] == model_name].copy()
            if df_model.empty: continue
            X, Y = df_model[feature_cols].values, df_model['Step'].values
            algorithms = get_algorithms()
            
            for algo_name, algo in algorithms.items():
                y_pred_cv = np.zeros_like(Y, dtype=float)
                
                # Scaler remains inside the fold loop because train_idx changes per fold
                for train_idx, test_idx in kf.split(X):
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X[train_idx])
                    X_test_scaled = scaler.transform(X[test_idx])
                    algo.fit(X_train_scaled, Y[train_idx])
                    y_pred_cv[test_idx] = algo.predict(X_test_scaled)
                
                r2 = r2_score(Y, y_pred_cv)
                if model_name == '124M': 
                    # Safe dictionary assignment
                    summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['CV_124M'] = r2
                
                plot_actual_vs_predicted(Y, y_pred_cv, model_name, algo_name, 
                                         "Internal Architecture Prediction (5-Fold CV)", 
                                         f"test2b_CV_{model_name}_{algo_name}.png", save_dir)

        # -----------------------------------------------------------------
        # TEST 5: HONEST ZERO-SHOT CROSS-ARCHITECTURE
        # -----------------------------------------------------------------
        print("\n--- TEST 5: HONEST ZERO-SHOT CROSS-ARCHITECTURE ---")
        df_honest = df.copy()
        for model_name, n_embd in EMBED_DIM_MAP.items():
            mask = df_honest['Model'] == model_name
            if mask.sum() > 0:
                df_honest.loc[mask, bin_cols] = df_honest.loc[mask, bin_cols] / np.sqrt(n_embd)
                
        df_train_h = df_honest[df_honest['Model'].isin(['7M', '30M'])]
        df_test_h = df_honest[df_honest['Model'] == '124M']
        
        if not df_train_h.empty and not df_test_h.empty:
            X_train_h, Y_train_h = df_train_h[feature_cols].values, df_train_h['Step'].values
            X_test_h, Y_test_h = df_test_h[feature_cols].values, df_test_h['Step'].values
            
            # Moved Scaler OUTSIDE the algorithm loop
            scaler = StandardScaler()
            X_train_h_scaled = scaler.fit_transform(X_train_h)
            X_test_h_scaled = scaler.transform(X_test_h)
            
            algorithms = get_algorithms()
            for algo_name, algo in algorithms.items():
                algo.fit(X_train_h_scaled, Y_train_h)
                y_pred_h = algo.predict(X_test_h_scaled)
                r2_h = r2_score(Y_test_h, y_pred_h)
                
                # Safe dictionary assignment
                summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['ZeroShot_124M'] = r2_h
                print(f"{algo_name:<15} | Zero-Shot R2: {r2_h:.4f}")
                
                plot_actual_vs_predicted(Y_test_h, y_pred_h, "124M (Zero-Shot)", algo_name, 
                                         "Train on 7M & 30M -> Tested on 124M", 
                                         f"test5s_zeros_shot_124m_{algo_name}.png", save_dir)

    # =====================================================================
    # COMPARISON SUMMARY OUTPUT
    # =====================================================================
    print("\n\n" + "="*85)
    print(f"{' A/B TEST RESULTS SUMMARY (STEP 2s: CLEAN AVERAGED DATA) ':^85}")
    print("="*85)
    print(f"{'Algorithm':<15} | {'Test Metric':<15} | {'Original R²':<15} | {'Interaction R²':<15} | {'Winner'}")
    print("-" * 85)
    
    for algo in ['Ridge', 'RandomForest', 'MLP']:
        for metric in ['CV_124M', 'ZeroShot_124M']:
            orig_val = summary.get('Original', {}).get(algo, {}).get(metric, 0)
            int_val = summary.get('Interaction', {}).get(algo, {}).get(metric, 0)
            winner = "✨ Original" if orig_val > int_val else "✨ Interaction"
            if abs(orig_val - int_val) < 0.001: winner = "Tie"
            print(f"{algo:<15} | {metric:<15} | {orig_val:<15.4f} | {int_val:<15.4f} | {winner}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_step2s_tests()