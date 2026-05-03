"""
external_generalization_combined.py
────────────────────────────────
Train Linear / MLP / RF on each of the 9 existing models INDIVIDUALLY (rank-normalized),
then train on ALL 9 models combined, and test on GPT-J (or OLMo) only.
Plots a grouped bar chart of the generalization R² for each base model + combined.
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR           = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

TRAIN_MODEL_SIZES = [
    '7M', '30M', '124M', '500M',
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

# Switch between 'gptj_6B' and 'olmo_7B' here
EXTERNAL_MODEL = 'gptj_6B'

MODEL_FOLDER_MAP = {
    '7M':            'MinGPT_Checkpoints_7M',
    '30M':           'MinGPT_Checkpoints_30M',
    '124M':          'MinGPT_Checkpoints_124M',
    '500M':          'MinGPT_Checkpoints_500M',
    '7M_owt_24M':    '7M_owt_24M',
    '30M_owt_24M':   '30M_owt_24M',
    '30M_owt_100M':  '30M_owt_100M',
    '30M_owt_420M':  '30M_owt_420M',
    '124M_owt_420M': '124M_owt_420M',
    'gptj_6B':       'gptj_6B',
    'olmo_7B':       'olmo_7B',
}

FREQ_FILE_MAP = {
    '7M':            os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':           os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M':          os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M':          os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    '7M_owt_24M':    os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_24M':   os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_100M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_100M_cache_token_frequencies.csv'),
    '30M_owt_420M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    '124M_owt_420M': os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    'gptj_6B':       os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
    'olmo_7B':       os.path.join(BASE_DIR, 'data', 'datasets', 'dolma_token_frequencies.csv'),
}

FEATURES_LIST = ['embedding_norm']

C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#7b9dd4'


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-column fractional ranks → [0, 1] as float32."""
    if arr.ndim == 1:
        return ((rankdata(arr) - 1) / max(len(arr) - 1, 1)).astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j])
        out[:, j] = (r - 1) / max(len(r) - 1, 1)
    return out


def load_model(size: str, freq_cache: dict):
    """
    Load one model's features + frequencies, merge, extract float32 arrays,
    and immediately drop the DataFrame to keep RAM low.
    Returns {'X', 'y', 'valid_features'} or None on failure.
    """
    folder    = MODEL_FOLDER_MAP[size]
    feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
    freq_path = FREQ_FILE_MAP[size]

    if not os.path.exists(feat_file):
        print(f"  ⚠ Feature file missing for {size}")
        return None
    if freq_path not in freq_cache:
        print(f"  ⚠ Frequency file missing for {size}")
        return None

    df_feat = pd.read_csv(feat_file, low_memory=False)
    valid   = [f for f in FEATURES_LIST if f in df_feat.columns]

    # Coerce token_id to int64 on both sides
    for df_tmp in [df_feat]:
        df_tmp['token_id'] = pd.to_numeric(
            df_tmp['token_id'], errors='coerce').astype('Int64')
        df_tmp.dropna(subset=['token_id'], inplace=True)
        df_tmp['token_id'] = df_tmp['token_id'].astype(np.int64)

    df_freq = freq_cache[freq_path].copy()
    df_freq['token_id'] = pd.to_numeric(
        df_freq['token_id'], errors='coerce').astype('Int64')
    df_freq.dropna(subset=['token_id'], inplace=True)
    df_freq['token_id'] = df_freq['token_id'].astype(np.int64)

    df    = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    y_raw = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

    # Drop rows where any feature or target is NaN
    before = len(df)
    df = df[valid + (['log_count'] if 'log_count' in df.columns else [])].copy()
    df['__y__'] = y_raw
    df.dropna(inplace=True)
    y_raw = df.pop('__y__')
    if len(df) < before:
        print(f"    (dropped {before - len(df):,} rows with NaN values)")

    # Extract only what we need, then free everything else
    X_np = df[valid].values.astype(np.float32)
    y_np = y_raw.values.astype(np.float32)
    n    = len(df)
    del df, df_feat, df_freq, y_raw
    gc.collect()

    mb = (X_np.nbytes + y_np.nbytes) / 1024**2
    print(f"  Loaded {size}: {n:,} tokens, {len(valid)} features  ({mb:.1f} MB)")
    return {'X': X_np, 'y': y_np, 'valid_features': valid}


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── 1. Cache frequency files ──────────────────────────────
    print("Loading frequency files...")
    needed_paths = set(FREQ_FILE_MAP[m] for m in TRAIN_MODEL_SIZES + [EXTERNAL_MODEL])
    freq_cache   = {}
    for path in needed_paths:
        if not os.path.exists(path):
            print(f"  ❌ Not found: {path}")
            continue
        freq_cache[path] = pd.read_csv(path)
        print(f"  ✅ {os.path.basename(path)}")

    # ── 2. Load training models ───────────────────────────────
    print("\nLoading training models...")
    train_data = {}
    for size in TRAIN_MODEL_SIZES:
        result = load_model(size, freq_cache)
        if result:
            train_data[size] = result

    # ── 3. Load external (test) model ────────────────────────
    print(f"\nLoading external model ({EXTERNAL_MODEL})...")
    ext_data = load_model(EXTERNAL_MODEL, freq_cache)

    # Frequency files no longer needed
    del freq_cache
    gc.collect()
    print("  (frequency cache freed)")

    if not train_data:
        print("No training models loaded. Exiting.")
        exit(1)
    if ext_data is None:
        print(f"External model {EXTERNAL_MODEL} could not be loaded. Exiting.")
        exit(1)

    # ── 4. Find common features ───────────────────────────────
    common_feats = set(FEATURES_LIST)
    for m, d in train_data.items():
        common_feats &= set(d['valid_features'])
    common_feats &= set(ext_data['valid_features'])
    common_feats  = sorted(common_feats)

    print(f"\nCommon features ({len(common_feats)}): {common_feats}")
    if not common_feats:
        print("No common features — cannot train. Exiting.")
        exit(1)

    # ── 5. Normalize external test set ───────────────────────
    idx_ext = [ext_data['valid_features'].index(f) for f in common_feats]
    X_test  = rank_normalize(ext_data['X'][:, idx_ext])
    y_test  = rank_normalize(ext_data['y'])
    print(f"\nTest set ({EXTERNAL_MODEL}): {X_test.shape[0]:,} rows")

    # ── 6. Fit & evaluate INDIVIDUAL models ──────────────────
    print("\nTraining and testing models individually...")
    results = []
    
    # Store these to build the combined dataset later
    X_train_parts = []
    y_train_parts = []

    for size, d in train_data.items():
        print(f"\n--- Training on {size} ---")
        idx     = [d['valid_features'].index(f) for f in common_feats]
        X_train = rank_normalize(d['X'][:, idx])
        y_train = rank_normalize(d['y'])
        
        X_train_parts.append(X_train)
        y_train_parts.append(y_train)

        # Linear
        lin = make_pipeline(StandardScaler(), LinearRegression())
        lin.fit(X_train, y_train)
        lin_r2 = r2_score(y_test, lin.predict(X_test))
        print(f"  Linear       R² = {lin_r2:.4f}")

        # MLP
        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), activation='relu',
                alpha=0.01, max_iter=2000, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=25,
                learning_rate='adaptive', random_state=42,
            )
        )
        mlp.fit(X_train, y_train)
        mlp_r2 = r2_score(y_test, mlp.predict(X_test))
        print(f"  MLP          R² = {mlp_r2:.4f}")

        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100, max_features='sqrt',
            min_samples_leaf=5, n_jobs=4, random_state=42,
        )
        rf.fit(X_train, y_train)
        rf_r2 = r2_score(y_test, rf.predict(X_test))
        print(f"  RandomForest R² = {rf_r2:.4f}")

        # Store results for charting
        results.append({
            'Model': size,
            'Linear': lin_r2,
            'MLP': mlp_r2,
            'RF': rf_r2
        })

    # ── 7. Fit & evaluate ALL COMBINED models ────────────────
    print("\n--- Training on ALL MODELS COMBINED ---")
    X_train_all = np.concatenate(X_train_parts)
    y_train_all = np.concatenate(y_train_parts)
    
    del X_train_parts, y_train_parts
    gc.collect()

    print(f"  Combined Train set: {X_train_all.shape[0]:,} rows")

    # Linear
    lin_all = make_pipeline(StandardScaler(), LinearRegression())
    lin_all.fit(X_train_all, y_train_all)
    lin_all_r2 = r2_score(y_test, lin_all.predict(X_test))
    print(f"  Linear       R² = {lin_all_r2:.4f}")

    # MLP
    mlp_all = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu',
            alpha=0.01, max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            learning_rate='adaptive', random_state=42,
        )
    )
    mlp_all.fit(X_train_all, y_train_all)
    mlp_all_r2 = r2_score(y_test, mlp_all.predict(X_test))
    print(f"  MLP          R² = {mlp_all_r2:.4f}")

    # Random Forest
    rf_all = RandomForestRegressor(
        n_estimators=100, max_features='sqrt',
        min_samples_leaf=5, n_jobs=4, random_state=42,
    )
    rf_all.fit(X_train_all, y_train_all)
    rf_all_r2 = r2_score(y_test, rf_all.predict(X_test))
    print(f"  RandomForest R² = {rf_all_r2:.4f}")

    # Append combined to results
    results.append({
        'Model': 'All Combined',
        'Linear': lin_all_r2,
        'MLP': mlp_all_r2,
        'RF': rf_all_r2
    })

    # ── 8. Chart (Grouped Bar Chart) ──────────────────────────
    df_res = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(df_res['Model']))
    width = 0.25

    # Plot grouped bars
    bars_lin = ax.bar(x - width, df_res['Linear'], width, label='Linear', color=C_LIN, edgecolor='white')
    bars_mlp = ax.bar(x,         df_res['MLP'],    width, label='MLP',    color=C_MLP, edgecolor='white')
    bars_rf  = ax.bar(x + width, df_res['RF'],     width, label='Random Forest', color=C_RF, edgecolor='white')

    ax.axhline(0, color='black', lw=0.8)
    
    # Helper function to add labels above/below bars
    def add_bar_labels(bars, values):
        for bar, v in zip(bars, values):
            va = 'bottom' if v >= 0 else 'top'
            oy = 0.02 if v >= 0 else -0.04
            ax.text(bar.get_x() + bar.get_width() / 2, v + oy,
                    f'{v:.3f}', ha='center', va=va, fontsize=8, color='black')

    # Apply text labels
    add_bar_labels(bars_lin, df_res['Linear'])
    add_bar_labels(bars_mlp, df_res['MLP'])
    add_bar_labels(bars_rf, df_res['RF'])

    # Calculate y-limits dynamically (padding slightly extra for the text labels)
    min_val = min(df_res['Linear'].min(), df_res['MLP'].min(), df_res['RF'].min())
    max_val = max(df_res['Linear'].max(), df_res['MLP'].max(), df_res['RF'].max())
    ax.set_ylim(min(0, min_val - 0.20), min(1.10, max_val + 0.20))

    ax.set_xticks(x)
    ax.set_xticklabels(df_res['Model'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("R² Score (rank-normalized target)", fontsize=11)
    
    # Bold 'All Combined' for visual separation
    ticks = ax.get_xticklabels()
    ticks[-1].set_fontweight('bold')
    
    ax.set_title(
        f"Generalization to {EXTERNAL_MODEL} (Individual & Combined Training)\n"
        f"Test: {EXTERNAL_MODEL}  |  ({len(common_feats)} features)",
        fontweight='bold', fontsize=12
    )
    
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_file = f"external_generalization_combined_{EXTERNAL_MODEL}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nSaved '{out_file}'")
    plt.close(fig)

    print("\nDone.")