import pandas as pd
import numpy as np
import os
import time
import datetime
import matplotlib
matplotlib.use("Agg")  # Safe backend for server execution
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# SERVER CONFIGURATION & PATHS (ROBUST VERSION)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MATRICES = {
    'Original': os.path.join(BASE_DIR, 'step2_master_matrix.csv'),
    'Interaction': os.path.join(BASE_DIR, 'step2_interaction_matrix.csv')
}

DIRS = {
    'Original': os.path.join(BASE_DIR, 'micro_graphs-step2_original'),
    'Interaction': os.path.join(BASE_DIR, 'micro_graphs-step2_interaction')
}

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}

def setup_graphs_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created graphs directory at: {path}")

def get_algorithms():
    return {
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=100, early_stopping=True),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=8)
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name, algo_name, test_type, filename, save_dir):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.4, color='royalblue', edgecolors='k', 
                label=f'{algo_name} Prediction')
    
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Ideal Identity Line')
    
    plt.title(f"Architecture: {model_name} | Algo: {algo_name}\n{test_type}", fontsize=14)
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

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"{'Model':<12} | {'Algorithm':<15} | {'R²':>8} | {'MAE':>12}")
    print(f"{'-'*55}")

def print_result_row(model_name, algo_name, r2, mae):
    print(f"{model_name:<12} | {algo_name:<15} | {r2:>8.4f} | {mae:>12.1f}")

def run_all_micro_tests():
    summary = {}
    start_time = time.time()
    print(f"🚀 Script started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 BASE_DIR set to: {BASE_DIR}\n")

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
        # TEST 1: RANDOM 80/20 SPLIT
        # -----------------------------------------------------------------
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] --- TEST 1: RANDOM 80/20 SPLIT (DATA LEAKAGE EXPECTED) ---")
        print_test_header("TEST 1: Random 80/20 Split")
        for model_name in models:
            df_model = df[df['Model'] == model_name].copy()
            if df_model.empty: continue
            X, Y = df_model[feature_cols].values, df_model['Step'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            algorithms = get_algorithms()
            for algo_name, algo in algorithms.items():
                algo.fit(X_train_scaled, y_train)
                y_pred = algo.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print_result_row(model_name, algo_name, r2, mae)

                if model_name == '124M':
                    summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['Test1_Random_124M'] = r2
                
                plot_actual_vs_predicted(y_test, y_pred, model_name, algo_name,
                                         "Random Split (Leakage Expected)", 
                                         f"test1_random_{model_name}_{algo_name}.png", save_dir)

        # -----------------------------------------------------------------
        # TEST 2: GROUPED 80/20 SPLIT
        # -----------------------------------------------------------------
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] --- TEST 2: GROUPED BY CHECKPOINT (CLEAN EVALUATION) ---")
        print_test_header("TEST 2: Grouped by Checkpoint")
        for model_name in models:
            df_model = df[df['Model'] == model_name].copy()
            if df_model.empty: continue
            X, Y = df_model[feature_cols].values, df_model['Step'].values
            groups = df_model['Step'].values
            
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(gss.split(X, Y, groups))
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X[train_idx])
            X_test_scaled = scaler.transform(X[test_idx])
            
            algorithms = get_algorithms()
            for algo_name, algo in algorithms.items():
                algo.fit(X_train_scaled, Y[train_idx])
                y_pred = algo.predict(X_test_scaled)
                r2 = r2_score(Y[test_idx], y_pred)
                mae = mean_absolute_error(Y[test_idx], y_pred)

                print_result_row(model_name, algo_name, r2, mae)

                if model_name == '124M': 
                    summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['Test2_Grouped_124M'] = r2
                
                plot_actual_vs_predicted(Y[test_idx], y_pred, model_name, algo_name,
                                         "Clean Internal Evaluation (Grouped)", 
                                         f"test2_grouped_{model_name}_{algo_name}.png", save_dir)

        # -----------------------------------------------------------------
        # TEST 5: HONEST CROSS-ARCHITECTURE
        # -----------------------------------------------------------------
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] --- TEST 5: HONEST CROSS-ARCHITECTURE (Zero-Shot) ---")
        print_test_header("TEST 5: Zero-Shot Cross-Architecture (Train: 7M+30M → Test: 124M)")
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
            
            scaler = StandardScaler()
            X_train_h_scaled = scaler.fit_transform(X_train_h)
            X_test_h_scaled = scaler.transform(X_test_h)
            
            algorithms = get_algorithms()
            for algo_name, algo in algorithms.items():
                algo.fit(X_train_h_scaled, Y_train_h)
                y_pred_h = algo.predict(X_test_h_scaled)
                r2_h = r2_score(Y_test_h, y_pred_h)
                mae_h = mean_absolute_error(Y_test_h, y_pred_h)

                print_result_row("124M (ZS)", algo_name, r2_h, mae_h)

                summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['Test5_ZeroShot_124M'] = r2_h
                
                plot_actual_vs_predicted(Y_test_h, y_pred_h, "124M (Zero-Shot)", algo_name,
                                         "Train: 7M & 30M -> Predict: 124M", 
                                         f"test5_honest_zero_shot_124m_{algo_name}.png", save_dir)

    # =====================================================================
    # COMPARISON SUMMARY OUTPUT
    # =====================================================================
    print("\n\n" + "="*85)
    print(f"{' A/B TEST RESULTS SUMMARY (STEP 2: NOISY BATCH DATA) ':^85}")
    print("="*85)
    print(f"{'Algorithm':<15} | {'Test Metric':<20} | {'Original R²':<15} | {'Interaction R²':<15} | {'Winner'}")
    print("-" * 85)
    
    for algo in ['Ridge', 'RandomForest', 'MLP']:
        for metric in ['Test1_Random_124M', 'Test2_Grouped_124M', 'Test5_ZeroShot_124M']:
            orig_val = summary.get('Original', {}).get(algo, {}).get(metric, 0)
            int_val = summary.get('Interaction', {}).get(algo, {}).get(metric, 0)
            winner = "✨ Original" if orig_val > int_val else "✨ Interaction"
            if abs(orig_val - int_val) < 0.001: winner = "Tie"
            print(f"{algo:<15} | {metric:<20} | {orig_val:<15.4f} | {int_val:<15.4f} | {winner}")
    print("="*85)
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ All tests completed successfully in {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    run_all_micro_tests()