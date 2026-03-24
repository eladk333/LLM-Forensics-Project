import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress MLP convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_PATH = os.path.join(CURRENT_DIR, 'step2s_master_matrix.csv')
GRAPHS_DIR = os.path.join(CURRENT_DIR, 'micro_graphs-step2s')

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}

def setup_graphs_dir():
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
        print(f"📁 Created graphs directory at: {GRAPHS_DIR}")

def get_algorithms():
    """
    Returns a dictionary of the models we want to compare.
    """
    return {
        'MLP': MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42, early_stopping=False),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

def plot_actual_vs_predicted(y_true, y_pred, model_name, algo_name, test_type, filename):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color='darkorange', edgecolors='k', s=60,
                label=f'{algo_name} Prediction')
    
    # Ideal identity line
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
    plt.savefig(os.path.join(GRAPHS_DIR, filename), dpi=300)
    plt.close()

def check_feature_importance(X, Y, feature_cols):
    """
    Analyzes the contribution of Mean_Eval_Probability vs the Embedding Bins using Random Forest.
    """
    print("\n" + "="*85)
    print(f"{'FEATURE IMPORTANCE ANALYSIS (Random Forest)':^85}")
    print("="*85)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, Y)
    
    importances = rf.feature_importances_
    
    prob_idx = feature_cols.index('Mean_Eval_Probability')
    prob_importance = importances[prob_idx]
    
    bin_indices = [i for i, col in enumerate(feature_cols) if col.startswith('Bin_')]
    bins_importance = np.sum(importances[bin_indices])
    
    print(f"Contribution of Mean_Eval_Probability: {prob_importance*100:.2f}%")
    print(f"Contribution of all Checkpoint Bins combined: {bins_importance*100:.2f}%")
    
    print(f"\nTop 5 most important individual features:")
    feat_imp_pairs = [(feature_cols[i], importances[i]) for i in range(len(feature_cols))]
    feat_imp_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i in range(5):
        print(f"  {i+1}. {feat_imp_pairs[i][0]}: {feat_imp_pairs[i][1]*100:.2f}%")

def run_step2s_tests():
    if not os.path.exists(MATRIX_PATH):
        print(f"❌ Error: Master matrix not found at {MATRIX_PATH}")
        return

    setup_graphs_dir()
    df = pd.read_csv(MATRIX_PATH)
    
    feature_cols = ['Mean_Eval_Probability'] + [col for col in df.columns if col.startswith('Bin_')]
    bin_cols = [col for col in df.columns if col.startswith('Bin_')]
    models = ['7M', '30M', '124M']
    algorithms = get_algorithms()

    # =====================================================================
    # TEST 2: STANDARD EVALUATION (CLEAN SPLIT PER ARCHITECTURE)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 2: STANDARD EVALUATION (80/20 SPLIT PER MODEL)':^85}")
    print("="*85)
    
    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        if df_model.empty: continue
        
        print(f"\nTarget Architecture: {model_name}")
        print("-" * 40)
        
        X = df_model[feature_cols].values
        Y = df_model['Step'].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for algo_name, algo in algorithms.items():
            algo.fit(X_train_scaled, y_train)
            y_pred = algo.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            print(f"  -> {algo_name:<15} | R2: {r2:+.4f} | MAE: +/- {mae:.1f} steps")
            
            filename = f"test2s_internal_{model_name}_{algo_name}.png"
            plot_actual_vs_predicted(y_test, y_pred, model_name, algo_name, 
                                     "Internal Architecture Prediction (80/20)", filename)

    # =====================================================================
    # TEST 2b: 5-FOLD CV EDITION
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 2b: 5-FOLD CV EDITION PER MODEL':^85}")
    print("="*85)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        if df_model.empty: continue
        
        print(f"\nTarget Architecture: {model_name}")
        print("-" * 40)
        
        X = df_model[feature_cols].values
        Y = df_model['Step'].values
        
        for algo_name, algo in algorithms.items():
            y_pred_cv = np.zeros_like(Y, dtype=float)
            
            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = Y[train_idx], Y[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                algo.fit(X_train_scaled, y_train)
                y_pred_cv[test_idx] = algo.predict(X_test_scaled)
            
            r2 = r2_score(Y, y_pred_cv)
            mae = mean_absolute_error(Y, y_pred_cv)
            print(f"  -> {algo_name:<15} | R2: {r2:+.4f} | MAE: +/- {mae:.1f} steps")
            
            filename = f"test2b_CV_{model_name}_{algo_name}.png"
            plot_actual_vs_predicted(Y, y_pred_cv, model_name, algo_name, 
                                     "Internal Architecture Prediction (5-Fold CV)", filename)

    # =====================================================================
    # TEST 5: HONEST ZERO-SHOT CROSS-ARCHITECTURE (Scaling by sqrt(N_EMBD))
    # =====================================================================
    print("\n\n" + "="*85)
    print(f"{'TEST 5: HONEST ZERO-SHOT (Train: 7M+30M -> Test: 124M)':^85}")
    print("="*85)
    
    df_honest = df.copy()
    
    for model_name, n_embd in EMBED_DIM_MAP.items():
        mask = df_honest['Model'] == model_name
        if mask.sum() > 0:
            df_honest.loc[mask, bin_cols] = df_honest.loc[mask, bin_cols] / np.sqrt(n_embd)
            
    df_train_h = df_honest[df_honest['Model'].isin(['7M', '30M'])]
    df_test_h = df_honest[df_honest['Model'] == '124M']
    
    if not df_train_h.empty and not df_test_h.empty:
        print(f"Evaluating Zero-Shot on unseen 124M Architecture:")
        print("-" * 55)
        
        X_train_h = df_train_h[feature_cols].values
        Y_train_h = df_train_h['Step'].values
        X_test_h = df_test_h[feature_cols].values
        Y_test_h = df_test_h['Step'].values
        
        scaler = StandardScaler()
        X_train_h_scaled = scaler.fit_transform(X_train_h)
        X_test_h_scaled = scaler.transform(X_test_h)
        
        for algo_name, algo in algorithms.items():
            algo.fit(X_train_h_scaled, Y_train_h)
            y_pred_h = algo.predict(X_test_h_scaled)
            
            r2_h = r2_score(Y_test_h, y_pred_h)
            mae_h = mean_absolute_error(Y_test_h, y_pred_h)
            print(f"  -> {algo_name:<15} | R2: {r2_h:+.4f} | MAE: +/- {mae_h:.1f} steps")
            
            filename = f"test5s_zeros_shot_124m_{algo_name}.png"
            plot_actual_vs_predicted(Y_test_h, y_pred_h, "124M (Zero-Shot)", algo_name, 
                                     "Trained on 7M & 30M -> Tested on 124M", filename)

    # =====================================================================
    # FEATURE IMPORTANCE (Running on the scaled 7M+30M training set)
    # =====================================================================
    check_feature_importance(df_train_h[feature_cols].values, df_train_h['Step'].values, feature_cols)
    
    print("-" * 85 + "\n✅ Step 2s Tests Complete. Graphs saved to folder.\n")

if __name__ == "__main__":
    run_step2s_tests()