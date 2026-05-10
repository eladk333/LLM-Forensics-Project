import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR           = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

CORE_MODELS = ['7M', '30M', '124M', '500M']

MODEL_FOLDER_MAP = {
    '7M':   'MinGPT_Checkpoints_7M',
    '30M':  'MinGPT_Checkpoints_30M',
    '124M': 'MinGPT_Checkpoints_124M',
    '500M': 'MinGPT_Checkpoints_500M',
}

FREQ_FILE_MAP = {
    '7M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M': os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M': os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
}

# The two feature sets to compare (logit_norm is the difference)
FEATURES_5 = ['embedding_norm', 'logit_norm', 'weight_variance', 'weight_mean', 'l1_norm']
FEATURES_4 = ['embedding_norm', 'weight_variance', 'weight_mean', 'l1_norm']

# Chart Colors
C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#7b9dd4'

# ──────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ──────────────────────────────────────────────────────────────

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return ((rankdata(arr) - 1) / max(len(arr) - 1, 1)).astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j])
        out[:, j] = (r - 1) / max(len(r) - 1, 1)
    return out

def load_model_data(size: str, freq_cache: dict):
    folder    = MODEL_FOLDER_MAP[size]
    feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
    freq_path = FREQ_FILE_MAP[size]

    if not os.path.exists(feat_file):
        print(f"  ⚠ Feature file missing for {size}")
        return None
    if freq_path not in freq_cache:
        print(f"  ⚠ Frequency file missing for {size}")
        return None

    header  = pd.read_csv(feat_file, nrows=0).columns.tolist()
    valid   = [f for f in FEATURES_5 if f in header]
    usecols = ['token_id'] + valid
    df_feat = pd.read_csv(feat_file, usecols=usecols, low_memory=False)

    for df_tmp in [df_feat]:
        df_tmp['token_id'] = pd.to_numeric(df_tmp['token_id'], errors='coerce').astype('Int64')
        df_tmp.dropna(subset=['token_id'], inplace=True)
        df_tmp['token_id'] = df_tmp['token_id'].astype(np.int64)

    df_freq = freq_cache[freq_path].copy()
    df_freq['token_id'] = pd.to_numeric(df_freq['token_id'], errors='coerce').astype('Int64')
    df_freq.dropna(subset=['token_id'], inplace=True)
    df_freq['token_id'] = df_freq['token_id'].astype(np.int64)

    df    = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    y_raw = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

    df = df[valid + (['log_count'] if 'log_count' in df.columns else [])].copy()
    df['__y__'] = y_raw
    df.dropna(inplace=True)
    y_raw = df.pop('__y__')
    
    X_np = df[valid].values.astype(np.float32)
    y_np = y_raw.values.astype(np.float32)
    
    del df, df_feat, df_freq, y_raw
    gc.collect()

    return {'X': X_np, 'y': y_np, 'valid_features': valid}

# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION (TRAIN ON 1, TEST ON ALL 4) - COMPARING 4 vs 5
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── 1. Cache frequency files ──────────────────────────────
    print("Loading frequency files...")
    needed_paths = set(FREQ_FILE_MAP[m] for m in CORE_MODELS)
    freq_cache   = {}
    for path in needed_paths:
        if not os.path.exists(path):
            continue
        freq_header = pd.read_csv(path, nrows=0).columns.tolist()
        freq_cols   = ['token_id', 'log_count'] if 'log_count' in freq_header else ['token_id', 'count']
        freq_cache[path] = pd.read_csv(path, usecols=freq_cols)

    # ── 2. Load and Normalize all models ──────────────────────
    print("\nLoading and normalizing all models...")
    all_data = {}
    for size in CORE_MODELS:
        raw_data = load_model_data(size, freq_cache)
        if raw_data:
            idx_5 = [raw_data['valid_features'].index(f) for f in FEATURES_5]
            
            all_data[size] = {
                'X': rank_normalize(raw_data['X'][:, idx_5]),
                'y': rank_normalize(raw_data['y'])
            }
            print(f"  ✅ {size} loaded and rank-normalized.")
            
    del freq_cache
    gc.collect()

    if len(all_data) != len(CORE_MODELS):
        print("Failed to load all required models. Exiting.")
        exit(1)

    print(f"\n{'='*75}\n--- Running 1-to-All Cross-Model Evaluation (4 vs 5 Features) ---\n{'='*75}")
    
    idx_4f = [FEATURES_5.index(f) for f in FEATURES_4]

    results_4f = {m: {'Linear': [], 'MLP': [], 'Random Forest': []} for m in CORE_MODELS}
    results_5f = {m: {'Linear': [], 'MLP': [], 'Random Forest': []} for m in CORE_MODELS}

    # ── 3. The 1-to-All Training Loop ─────────────────────────
    for train_model in CORE_MODELS:
        print(f"\n[Iteration] Training on: {train_model}  |  Testing on: {CORE_MODELS}")
        
        y_train = all_data[train_model]['y']
        X_train_5f = all_data[train_model]['X']
        X_train_4f = X_train_5f[:, idx_4f]

        scaler_4f = StandardScaler()
        X_train_scaled_4f = scaler_4f.fit_transform(X_train_4f)
        
        scaler_5f = StandardScaler()
        X_train_scaled_5f = scaler_5f.fit_transform(X_train_5f)

        # ── Fit 4-Feature Models ──
        lin_4f = LinearRegression().fit(X_train_scaled_4f, y_train)
        mlp_4f = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu',
            alpha=0.01, max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            learning_rate='adaptive', random_state=42
        ).fit(X_train_scaled_4f, y_train)
        rf_4f = RandomForestRegressor(
            n_estimators=100, max_features='sqrt',
            min_samples_leaf=5, n_jobs=4, random_state=42
        ).fit(X_train_scaled_4f, y_train)

        # ── Fit 5-Feature Models ──
        lin_5f = LinearRegression().fit(X_train_scaled_5f, y_train)
        mlp_5f = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu',
            alpha=0.01, max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            learning_rate='adaptive', random_state=42
        ).fit(X_train_scaled_5f, y_train)
        rf_5f = RandomForestRegressor(
            n_estimators=100, max_features='sqrt',
            min_samples_leaf=5, n_jobs=4, random_state=42
        ).fit(X_train_scaled_5f, y_train)

        # Evaluate on ALL models
        for test_model in CORE_MODELS:
            y_test = all_data[test_model]['y']
            X_test_5f = all_data[test_model]['X']
            X_test_4f = X_test_5f[:, idx_4f]
            
            X_test_scaled_4f = scaler_4f.transform(X_test_4f)
            X_test_scaled_5f = scaler_5f.transform(X_test_5f)

            # Score 4 Features
            r2_lin_4f = r2_score(y_test, lin_4f.predict(X_test_scaled_4f))
            r2_mlp_4f = r2_score(y_test, mlp_4f.predict(X_test_scaled_4f))
            r2_rf_4f  = r2_score(y_test, rf_4f.predict(X_test_scaled_4f))
            
            results_4f[train_model]['Linear'].append(r2_lin_4f)
            results_4f[train_model]['MLP'].append(r2_mlp_4f)
            results_4f[train_model]['Random Forest'].append(r2_rf_4f)

            # Score 5 Features
            r2_lin_5f = r2_score(y_test, lin_5f.predict(X_test_scaled_5f))
            r2_mlp_5f = r2_score(y_test, mlp_5f.predict(X_test_scaled_5f))
            r2_rf_5f  = r2_score(y_test, rf_5f.predict(X_test_scaled_5f))
            
            results_5f[train_model]['Linear'].append(r2_lin_5f)
            results_5f[train_model]['MLP'].append(r2_mlp_5f)
            results_5f[train_model]['Random Forest'].append(r2_rf_5f)
            
            print(f"  -> Test on {test_model:4s} | [4F] Lin:{r2_lin_4f:+.3f} MLP:{r2_mlp_4f:+.3f} RF:{r2_rf_4f:+.3f} | [5F] Lin:{r2_lin_5f:+.3f} MLP:{r2_mlp_5f:+.3f} RF:{r2_rf_5f:+.3f}")

    # ── Helper for adding labels to bar charts ────────────────
    def add_labels(ax_obj, bars, values, is_delta=False):
        for bar, v in zip(bars, values):
            va = 'bottom' if v >= 0 else 'top'
            oy = 0.02 if v >= 0 else -0.04
            if is_delta:
                # For deltas, keeping a +/- sign is helpful
                label_text = f'{v:+.3f}'
                oy = 0.005 if v >= 0 else -0.005
            else:
                label_text = f'{v:.2f}'

            ax_obj.text(bar.get_x() + bar.get_width() / 2, v + oy,
                    label_text, ha='center', va=va, fontsize=8, fontweight='bold',
                    color='black', rotation='vertical')

    # ── 4. Plotting the Side-by-Side Comparison Charts ────────
    print("\nGenerating final comparative charts (4 vs 5 Features)...")
    
    x = np.arange(len(CORE_MODELS))
    
    for train_model in CORE_MODELS:
        fig, ax = plt.subplots(figsize=(13, 7))
        width = 0.13  
        
        s_lin_4 = results_4f[train_model]['Linear']
        s_mlp_4 = results_4f[train_model]['MLP']
        s_rf_4  = results_4f[train_model]['Random Forest']
        
        s_lin_5 = results_5f[train_model]['Linear']
        s_mlp_5 = results_5f[train_model]['MLP']
        s_rf_5  = results_5f[train_model]['Random Forest']

        bars_l4 = ax.bar(x - 2.5*width, s_lin_4, width, label='Linear (4 Feat)', color=C_LIN, edgecolor='white')
        bars_m4 = ax.bar(x - 1.5*width, s_mlp_4, width, label='MLP (4 Feat)', color=C_MLP, edgecolor='white')
        bars_r4 = ax.bar(x - 0.5*width, s_rf_4,  width, label='Random Forest (4 Feat)', color=C_RF, edgecolor='white')

        bars_l5 = ax.bar(x + 0.5*width, s_lin_5, width, label='Linear (5 Feat)', color=C_LIN, hatch='//', edgecolor='white')
        bars_m5 = ax.bar(x + 1.5*width, s_mlp_5, width, label='MLP (5 Feat)', color=C_MLP, hatch='//', edgecolor='white')
        bars_r5 = ax.bar(x + 2.5*width, s_rf_5,  width, label='Random Forest (5 Feat)', color=C_RF, hatch='//', edgecolor='white')

        add_labels(ax, bars_l4, s_lin_4)
        add_labels(ax, bars_m4, s_mlp_4)
        add_labels(ax, bars_r4, s_rf_4)
        add_labels(ax, bars_l5, s_lin_5)
        add_labels(ax, bars_m5, s_mlp_5)
        add_labels(ax, bars_r5, s_rf_5)

        ax.axhline(0, color='black', lw=0.8)
        
        min_val = min(min(s_lin_4), min(s_mlp_4), min(s_rf_4), min(s_lin_5), min(s_mlp_5), min(s_rf_5))
        ax.set_ylim(min(0, min_val - 0.20), 1.15)
        
        ax.set_ylabel("R² Score (rank-normalized target)", fontsize=11)
        ax.set_title(f"Feature Set Comparison: 4 Features vs 5 Features\n(Trained on {train_model}, Tested across Test models)", fontweight='bold', fontsize=13)
        
        ax.set_xticks(x)
        xticklabels = [f"{m} (Test Set)" if m == train_model else f"{m} (Gen)" for m in CORE_MODELS]
        ax.set_xticklabels(xticklabels)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9, ncol=2)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        plt.tight_layout()
        out_file = f"comparison_4v5_trained_on_{train_model}.png"
        plt.savefig(out_file, dpi=300)
        print(f"  Saved chart: '{out_file}'")

    # ── 5. Plotting the Delta Charts (5-Feature minus 4-Feature) ──
    print("\nGenerating Delta charts (5F Score - 4F Score)...")
    
    for train_model in CORE_MODELS:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.25  
        
        # Calculate Delta: 4F - 5F (Impact of removing logit_norm)
        delta_lin = [s4 - s5 for s5, s4 in zip(results_5f[train_model]['Linear'], results_4f[train_model]['Linear'])]
        delta_mlp = [s4 - s5 for s5, s4 in zip(results_5f[train_model]['MLP'], results_4f[train_model]['MLP'])]
        delta_rf  = [s4 - s5 for s5, s4 in zip(results_5f[train_model]['Random Forest'], results_4f[train_model]['Random Forest'])]
        
        bars_dl = ax.bar(x - width, delta_lin, width, label='Linear Delta', color=C_LIN, edgecolor='white')
        bars_dm = ax.bar(x, delta_mlp, width, label='MLP Delta', color=C_MLP, edgecolor='white')
        bars_dr = ax.bar(x + width, delta_rf, width, label='Random Forest Delta', color=C_RF, edgecolor='white')

        add_labels(ax, bars_dl, delta_lin, is_delta=True)
        add_labels(ax, bars_dm, delta_mlp, is_delta=True)
        add_labels(ax, bars_dr, delta_rf, is_delta=True)

        ax.axhline(0, color='black', lw=1.2)
        
        # Create a balanced Y-axis so 0 is in the middle if there are negative and positive deltas
        max_delta = max(max(delta_lin), max(delta_mlp), max(delta_rf))
        min_delta = min(min(delta_lin), min(delta_mlp), min(delta_rf))
        y_limit = max(abs(max_delta), abs(min_delta)) + 0.05
        ax.set_ylim(-y_limit, y_limit)
        
        ax.set_ylabel("Δ R² Score (4 Features - 5 Features)", fontsize=11)
        ax.set_title(f"Performance Delta: Impact of Removing `logit_norm`\n(Trained on {train_model})", fontweight='bold', fontsize=13)
        ax.set_xticks(x)
        xticklabels = [f"{m} (Test Set)" if m == train_model else f"{m} (Gen)" for m in CORE_MODELS]
        ax.set_xticklabels(xticklabels)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        plt.tight_layout()
        out_file = f"delta_4v5_trained_on_{train_model}.png"
        plt.savefig(out_file, dpi=300)
        print(f"  Saved delta chart: '{out_file}'")

    print("\nDone!")