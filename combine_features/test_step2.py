import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION & ARCHITECTURE DATA
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_PATH = os.path.join(CURRENT_DIR, 'step2_master_matrix.csv')
GRAPHS_DIR = os.path.join(CURRENT_DIR, 'micro_graphs-embed_norm_and_prob')

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}
LAYER_MAP = {'7M': 4, '30M': 6, '124M': 12}

def setup_graphs_dir():
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)
        print(f"📁 Created graphs directory at: {GRAPHS_DIR}")

def plot_actual_vs_predicted(y_true, y_pred, model_name, test_type, filename):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    # Updated Label per your request
    plt.scatter(y_true, y_pred, alpha=0.4, color='royalblue', edgecolors='k', 
                label='Step Prediction (per Batch Probability + Checkpoint Bins)')
    
    # Ideal line
    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Ideal Identity Line (Perfect Accuracy)')
    
    plt.title(f"Architecture: {model_name} | {test_type}\nMethod: 80/20 Split (Grouped by Checkpoint - Zero Leakage)", fontsize=14)
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

def plot_zero_shot_comparison(zero_shot_results):
    plt.figure(figsize=(11, 7))
    names = list(zero_shot_results.keys())
    scores = list(zero_shot_results.values())
    
    colors = ['#ff9999', '#99cc99'] 
    
    bars = plt.bar(names, scores, color=colors, edgecolor='black', width=0.6)
    plt.axhline(0, color='black', lw=1.2)
    plt.ylim(-1.2, 1.2)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05 if yval > 0 else yval - 0.1, 
                 f'R² = {yval:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.ylabel('R² Score', fontsize=12)
    plt.title('124M Zero-Shot Performance (Train on 7M+30M -> Test on 124M)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'zero_shot_comparison_bar.png'), dpi=300)
    plt.close()

def check_feature_importance(df, feature_cols):
    """Bonus function to answer: How much does the Probability feature actually help?"""
    print("\n" + "="*85)
    print(f"{'FEATURE IMPORTANCE ANALYSIS (Random Forest)':^85}")
    print("="*85)
    df_7m = df[df['Model'] == '7M'].copy()
    if df_7m.empty: return
    
    X, Y = df_7m[feature_cols].values, df_7m['Step'].values
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, Y)
    
    importances = rf.feature_importances_
    prob_importance = importances[0] # Assuming Batch_Probability is the first column
    bins_importance = np.sum(importances[1:])
    
    print(f"Contribution of Batch_Probability: {prob_importance*100:.2f}%")
    print(f"Contribution of all 50 Bins combined: {bins_importance*100:.2f}%")
    print(f"Top 3 most important individual Bins:")
    
    bin_imp_pairs = [(feature_cols[i], importances[i]) for i in range(1, len(feature_cols))]
    bin_imp_pairs.sort(key=lambda x: x[1], reverse=True)
    for i in range(3):
        print(f"  - {bin_imp_pairs[i][0]}: {bin_imp_pairs[i][1]*100:.2f}%")

def run_all_micro_tests():
    if not os.path.exists(MATRIX_PATH):
        print(f"Error: Master matrix not found at {MATRIX_PATH}")
        return

    setup_graphs_dir()
    df = pd.read_csv(MATRIX_PATH)
    
    feature_cols = ['Batch_Probability'] + [col for col in df.columns if col.startswith('Bin_')]
    bin_cols = [col for col in df.columns if col.startswith('Bin_')]
    models = ['7M', '30M', '124M']
    
    zero_shot_r2_scores = {}
    
    def get_mlp():
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=100, early_stopping=True)

    # =====================================================================
    # TEST 1: RANDOM 80/20 SPLIT
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 1: RANDOM 80/20 SPLIT (DATA LEAKAGE EXPECTED)':^85}")
    print("="*85)
    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        if df_model.empty: continue
        X, Y = df_model[feature_cols].values, df_model['Step'].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        mlp = get_mlp()
        mlp.fit(scaler.fit_transform(X_train), y_train)
        y_pred = mlp.predict(scaler.transform(X_test))
        print(f"{model_name:<10} | R2: {r2_score(y_test, y_pred):.4f}")

    # =====================================================================
    # TEST 2: GROUPED 80/20 SPLIT (Clean)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 2: GROUPED BY CHECKPOINT (CLEAN EVALUATION)':^85}")
    print("="*85)
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
        
        print(f"{model_name:<10} | R2: {r2_score(Y[test_idx], y_pred):.4f} | MAE: +/- {mean_absolute_error(Y[test_idx], y_pred):.1f}")
        
        plot_actual_vs_predicted(Y[test_idx], y_pred, model_name, 
                                 "Clean Internal Evaluation (Unseen Checkpoints)", 
                                 f"test2_clean_grouped_{model_name}.png")

    # =====================================================================
    # TEST 3: UNNORMALIZED (Commented out per request)
    # =====================================================================
    """
    df_train, df_test = df[df['Model'].isin(['7M', '30M'])].copy(), df[df['Model'] == '124M'].copy()
    if not df_train.empty and not df_test.empty:
        scaler = StandardScaler()
        mlp = get_mlp()
        mlp.fit(scaler.fit_transform(df_train[feature_cols].values), df_train['Step'].values)
        y_pred = mlp.predict(scaler.transform(df_test[feature_cols].values))
        r2 = r2_score(df_test['Step'].values, y_pred)
        zero_shot_r2_scores['1. Unnormalized (Scale Mismatch)'] = max(r2, -1.0)
    """

    # =====================================================================
    # TEST 5: HONEST CROSS-MODEL (Scaling by sqrt(N_EMBD))
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 5: HONEST CROSS-ARCHITECTURE (Scaling by sqrt(N_EMBD))':^85}")
    print("="*85)
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
        mae_h = mean_absolute_error(df_test_h['Step'].values, y_pred_h)
        
        # *** THIS IS THE PRINT STATEMENT THAT WAS MISSING ***
        print(f"{'124M (Honest)':<10} | R2: {r2_h:.4f} | MAE: +/- {mae_h:.1f}")
        
        zero_shot_r2_scores['2. Honest N_EMBD (Physics-based)'] = r2_h
        
        plot_actual_vs_predicted(df_test_h['Step'].values, y_pred_h, "124M (Zero-Shot Prediction)", 
                                 "Train: 7M & 30M Models -> Predict: 124M Model", 
                                 "test5_honest_zero_shot_124m.png")

    print("-" * 85 + "\nAll Tests Complete.\n")
    
    # Run the feature importance check
    check_feature_importance(df, feature_cols)
    
    if len(zero_shot_r2_scores) >= 2:
        plot_zero_shot_comparison(zero_shot_r2_scores)
    print(f"\n📉 Detailed scatter plots saved to: {GRAPHS_DIR}")

if __name__ == "__main__":
    run_all_micro_tests()