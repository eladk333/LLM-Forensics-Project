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
BASE_DIR           = '/home/nlp/katzela4/LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

# The full suite of models for analysis
CORE_MODELS = [
    '7M', '30M', '124M', '500M', 'GPT-J', 
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

# Folder mapping for custom models. GPT-J is in the current features_scripts directory.
MODEL_FOLDER_MAP = {
    '7M':            'MinGPT_Checkpoints_7M',
    '30M':           'MinGPT_Checkpoints_30M',
    '124M':          'MinGPT_Checkpoints_124M',
    '500M':          'MinGPT_Checkpoints_500M',
    'GPT-J':         '', 
    '7M_owt_24M':    '7M_owt_24M',
    '30M_owt_24M':   '30M_owt_24M',
    '30M_owt_100M':  '30M_owt_100M',
    '30M_owt_420M':  '30M_owt_420M',
    '124M_owt_420M': '124M_owt_420M',
}

# Ground Truth Frequency Mapping
FREQ_FILE_MAP = {
    '7M':            os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':           os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M':          os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M':          os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    'GPT-J':         os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
    '7M_owt_24M':    os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_24M':   os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_100M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_100M_cache_token_frequencies.csv'),
    '30M_owt_420M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    '124M_owt_420M': os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
}

# The full 5-feature set
FEATURES = ['embedding_norm', 'weight_variance', 'logit_norm', 'weight_mean', 'l1_norm']

# Chart Colors
C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#7b9dd4'

# ──────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ──────────────────────────────────────────────────────────────

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Converts raw values to [0, 1] range based on their rank."""
    if arr.ndim == 1:
        return ((rankdata(arr) - 1) / max(len(arr) - 1, 1)).astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j])
        out[:, j] = (r - 1) / max(len(r) - 1, 1)
    return out

def load_model_data(size: str, freq_cache: dict):
    folder = MODEL_FOLDER_MAP[size]
    
    if size == 'GPT-J':
        feat_file = 'gptj_model_features.csv'
    else:
        feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
        
    freq_path = FREQ_FILE_MAP[size]

    if not os.path.exists(feat_file):
        print(f"  ⚠ Feature file missing for {size}: {feat_file}")
        return None
    if freq_path not in freq_cache:
        print(f"  ⚠ Frequency file missing for {size}: {freq_path}")
        return None

    header  = pd.read_csv(feat_file, nrows=0).columns.tolist()
    valid   = [f for f in FEATURES if f in header]
    usecols = ['token_id'] + valid
    df_feat = pd.read_csv(feat_file, usecols=usecols, low_memory=False)
    
    df_feat['token_id'] = pd.to_numeric(df_feat['token_id'], errors='coerce').astype('Int64')
    df_feat.dropna(subset=['token_id'], inplace=True)
    df_feat['token_id'] = df_feat['token_id'].astype(np.int64)

    df_freq = freq_cache[freq_path].copy()
    df_freq['token_id'] = pd.to_numeric(df_freq['token_id'], errors='coerce').astype('Int64')
    df_freq.dropna(subset=['token_id'], inplace=True)
    df_freq['token_id'] = df_freq['token_id'].astype(np.int64)

    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    
    y_raw = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

    df = df[valid].copy()
    df['__y__'] = y_raw
    df.dropna(inplace=True)
    y_raw = df.pop('__y__')
    
    X_np = df[valid].values.astype(np.float32)
    y_np = y_raw.values.astype(np.float32)
    
    del df, df_feat, df_freq, y_raw
    gc.collect()

    return {'X': X_np, 'y': y_np, 'valid_features': valid}

# ──────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading frequency files...")
    needed_paths = set(FREQ_FILE_MAP[m] for m in CORE_MODELS)
    freq_cache   = {}
    for path in needed_paths:
        if not os.path.exists(path):
            print(f"  ⚠ File not found: {path}")
            continue
        freq_header = pd.read_csv(path, nrows=0).columns.tolist()
        freq_cols   = ['token_id', 'log_count'] if 'log_count' in freq_header else ['token_id', 'count']
        freq_cache[path] = pd.read_csv(path, usecols=freq_cols)

    print("\nLoading and normalizing all models...")
    all_data = {}
    for size in CORE_MODELS:
        raw_data = load_model_data(size, freq_cache)
        if raw_data:
            # Reorder columns to match FEATURES precisely (assumes all 5 are present)
            idx = [raw_data['valid_features'].index(f) for f in FEATURES]
            all_data[size] = {
                'X_5': rank_normalize(raw_data['X'][:, idx]),
                'y': rank_normalize(raw_data['y'])
            }
            # Create a 4-feature version by dropping 'logit_norm'
            idx_4 = [FEATURES.index(f) for f in FEATURES if f != 'logit_norm']
            all_data[size]['X_4'] = all_data[size]['X_5'][:, idx_4]
            
            print(f"  ✅ {size} loaded and rank-normalized.")
            
    del freq_cache
    gc.collect()

    if len(all_data) < 2:
        print("Not enough models loaded. Exiting.")
        exit(1)

    print(f"\n{'='*85}\n--- Running 1-to-All Cross-Model Evaluation (5-Feat vs 4-Feat) ---\n{'='*85}")
    
    results_5 = {m: {alg: [] for alg in ['Linear', 'MLP', 'Random Forest']} for m in CORE_MODELS}
    results_4 = {m: {alg: [] for alg in ['Linear', 'MLP', 'Random Forest']} for m in CORE_MODELS}

    for train_model in CORE_MODELS:
        if train_model not in all_data: continue
        print(f"\n[Iteration] Training on: {train_model}")
        
        # --- TRAIN 5 FEATURES ---
        X_train_5 = all_data[train_model]['X_5']
        y_train   = all_data[train_model]['y']

        scaler_5 = StandardScaler()
        X_train_5_s = scaler_5.fit_transform(X_train_5)

        models_5 = {
            'Linear': LinearRegression().fit(X_train_5_s, y_train),
            'MLP':    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=1000, random_state=42).fit(X_train_5_s, y_train),
            'RF':     RandomForestRegressor(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=42).fit(X_train_5_s, y_train),
        }

        # --- TRAIN 4 FEATURES ---
        X_train_4 = all_data[train_model]['X_4']
        
        scaler_4 = StandardScaler()
        X_train_4_s = scaler_4.fit_transform(X_train_4)

        models_4 = {
            'Linear': LinearRegression().fit(X_train_4_s, y_train),
            'MLP':    MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.01, max_iter=1000, random_state=42).fit(X_train_4_s, y_train),
            'RF':     RandomForestRegressor(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=42).fit(X_train_4_s, y_train),
        }

        # --- EVALUATION LOOP ---
        for test_model in CORE_MODELS:
            if test_model not in all_data: continue
            
            y_test = all_data[test_model]['y']

            # Test 5 Features
            X_test_5_s = scaler_5.transform(all_data[test_model]['X_5'])
            r2_5_lin = r2_score(y_test, models_5['Linear'].predict(X_test_5_s))
            r2_5_mlp = r2_score(y_test, models_5['MLP'].predict(X_test_5_s))
            r2_5_rf  = r2_score(y_test, models_5['RF'].predict(X_test_5_s))
            
            results_5[train_model]['Linear'].append(r2_5_lin)
            results_5[train_model]['MLP'].append(r2_5_mlp)
            results_5[train_model]['Random Forest'].append(r2_5_rf)

            # Test 4 Features
            X_test_4_s = scaler_4.transform(all_data[test_model]['X_4'])
            r2_4_lin = r2_score(y_test, models_4['Linear'].predict(X_test_4_s))
            r2_4_mlp = r2_score(y_test, models_4['MLP'].predict(X_test_4_s))
            r2_4_rf  = r2_score(y_test, models_4['RF'].predict(X_test_4_s))
            
            results_4[train_model]['Linear'].append(r2_4_lin)
            results_4[train_model]['MLP'].append(r2_4_mlp)
            results_4[train_model]['Random Forest'].append(r2_4_rf)
            
            print(f"  -> Test: {test_model:14s} | 4-Feat (Lin/MLP/RF): {r2_4_lin:+.3f}/{r2_4_mlp:+.3f}/{r2_4_rf:+.3f} | 5-Feat (Lin/MLP/RF): {r2_5_lin:+.3f}/{r2_5_mlp:+.3f}/{r2_5_rf:+.3f}")

    # ──────────────────────────────────────────────────────────────
    # PLOTTING HELPERS
    # ──────────────────────────────────────────────────────────────
    
    def add_vertical_labels(ax, bars, values, is_delta=False):
        for bar, v in zip(bars, values):
            label = f"{v:+.3f}" if is_delta else f"{v:.2f}"
            offset = 0.03 if v >= 0 else -0.03
            ax.text(bar.get_x() + bar.get_width()/2, v + offset, label, 
                    ha='center', va='bottom' if v >= 0 else 'top', 
                    rotation=90, fontsize=8, fontweight='bold')

    x = np.arange(len(CORE_MODELS))
    
    for train_model in CORE_MODELS:
        if train_model not in all_data: continue
        
        xtick_labels = [f"{m} {'(Test)' if m == train_model else '(Gen)'}" for m in CORE_MODELS]

        # -------------------------------------------------------------
        # PLOT 1: Comparison Chart (4-Feat vs 5-Feat)
        # -------------------------------------------------------------
        # Increased figure width slightly to accommodate more models
        fig, ax = plt.subplots(figsize=(16, 6)) 
        w = 0.12 # bar width for 6 bars
        
        # 4 Features (Solid)
        b1 = ax.bar(x - 2.5*w, results_4[train_model]['Linear'], w, label='Linear (4 Feat)', color=C_LIN, edgecolor='white')
        b2 = ax.bar(x - 1.5*w, results_4[train_model]['MLP'], w, label='MLP (4 Feat)', color=C_MLP, edgecolor='white')
        b3 = ax.bar(x - 0.5*w, results_4[train_model]['Random Forest'], w, label='Random Forest (4 Feat)', color=C_RF, edgecolor='white')

        # 5 Features (Hatched)
        b4 = ax.bar(x + 0.5*w, results_5[train_model]['Linear'], w, label='Linear (5 Feat)', color=C_LIN, hatch='//', edgecolor='white', alpha=0.85)
        b5 = ax.bar(x + 1.5*w, results_5[train_model]['MLP'], w, label='MLP (5 Feat)', color=C_MLP, hatch='//', edgecolor='white', alpha=0.85)
        b6 = ax.bar(x + 2.5*w, results_5[train_model]['Random Forest'], w, label='Random Forest (5 Feat)', color=C_RF, hatch='//', edgecolor='white', alpha=0.85)

        add_vertical_labels(ax, b1, results_4[train_model]['Linear'])
        add_vertical_labels(ax, b2, results_4[train_model]['MLP'])
        add_vertical_labels(ax, b3, results_4[train_model]['Random Forest'])
        add_vertical_labels(ax, b4, results_5[train_model]['Linear'])
        add_vertical_labels(ax, b5, results_5[train_model]['MLP'])
        add_vertical_labels(ax, b6, results_5[train_model]['Random Forest'])

        ax.axhline(0, color='black', lw=1)
        ax.set_title(f"Feature Set Comparison: 4 Features vs 5 Features\n(Trained on {train_model}, Tested across Test models)", fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
        ax.set_ylabel("R² Score (rank-normalized target)")
        
        all_r2_scores = list(results_4[train_model].values()) + list(results_5[train_model].values())
        ax.set_ylim(bottom=min(0, np.min(all_r2_scores)) - 0.15,
                    top=np.max(all_r2_scores) + 0.15)
        
        #ax.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.tight_layout()
        out_file = f"eval_comparison_{train_model}.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"  Saved comparison chart: '{out_file}'")

        # -------------------------------------------------------------
        # PLOT 2: Delta Chart (4-Feat minus 5-Feat)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(14, 6))
        w_d = 0.25
        
        delta_lin = np.array(results_4[train_model]['Linear']) - np.array(results_5[train_model]['Linear'])
        delta_mlp = np.array(results_4[train_model]['MLP']) - np.array(results_5[train_model]['MLP'])
        delta_rf  = np.array(results_4[train_model]['Random Forest']) - np.array(results_5[train_model]['Random Forest'])

        bd1 = ax.bar(x - w_d, delta_lin, w_d, label='Linear Delta', color=C_LIN, edgecolor='white')
        bd2 = ax.bar(x,       delta_mlp, w_d, label='MLP Delta', color=C_MLP, edgecolor='white')
        bd3 = ax.bar(x + w_d, delta_rf,  w_d, label='Random Forest Delta', color=C_RF, edgecolor='white')

        add_vertical_labels(ax, bd1, delta_lin, is_delta=True)
        add_vertical_labels(ax, bd2, delta_mlp, is_delta=True)
        add_vertical_labels(ax, bd3, delta_rf, is_delta=True)

        ax.axhline(0, color='black', lw=1)
        ax.set_title(f"Performance Delta: Impact of Removing `logit_norm`\n(Trained on {train_model})", fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
        ax.set_ylabel("Δ R² Score (4 Features - 5 Features)")
        
        all_deltas = np.concatenate([delta_lin, delta_mlp, delta_rf])
        ax.set_ylim(bottom=min(-0.2, np.min(all_deltas) - 0.15), 
                    top=max(0.2, np.max(all_deltas) + 0.15))

        #ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

        plt.tight_layout()
        out_file_delta = f"eval_delta_{train_model}.png"
        plt.savefig(out_file_delta, dpi=300)
        plt.close()
        print(f"  Saved delta chart: '{out_file_delta}'")

    print("\nAll tasks completed successfully.")