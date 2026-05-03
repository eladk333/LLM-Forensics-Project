"""
train_500M_feature_importance.py
────────────────────────────────
Trains Linear / MLP / RF purely on the 500M model (rank-normalized),
and tests generalization on GPT-J.
Generates performance metrics and Feature Importance charts for 
both Linear Regression (Coefficients) and Random Forest (Permutation).
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
from sklearn.inspection import permutation_importance

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR           = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

# Restricted to only 500M as requested
TRAIN_MODEL_SIZES = ['500M']
EXTERNAL_MODEL    = 'gptj_6B'

MODEL_FOLDER_MAP = {
    '500M':          'MinGPT_Checkpoints_500M',
    'gptj_6B':       'gptj_6B',
}

FREQ_FILE_MAP = {
    '500M':          os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    'gptj_6B':       os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
}

FEATURES_LIST = ['embedding_norm', 'logit_norm', 'weight_variance', 'weight_mean', 'l1_norm']

# Nice formatting names for the charts
FEATURE_NAMES = {
    'embedding_norm': 'Embedding Norm',
    'logit_norm': 'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean': 'Weight Mean',
    'l1_norm': 'L1 Norm'
}

C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#4f78b4' # Matched closer to the deep blue in your example image


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

    before = len(df)
    df = df[valid + (['log_count'] if 'log_count' in df.columns else [])].copy()
    df['__y__'] = y_raw
    df.dropna(inplace=True)
    y_raw = df.pop('__y__')
    
    if len(df) < before:
        print(f"    (dropped {before - len(df):,} rows with NaN values)")

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

    # ── 2. Load 500M Training Model ───────────────────────────
    print("\nLoading training model (500M)...")
    train_data = load_model('500M', freq_cache)

    # ── 3. Load external (test) model ────────────────────────
    print(f"\nLoading external model ({EXTERNAL_MODEL})...")
    ext_data = load_model(EXTERNAL_MODEL, freq_cache)

    del freq_cache
    gc.collect()

    if train_data is None or ext_data is None:
        print("Required models could not be loaded. Exiting.")
        exit(1)

    # ── 4. Find common features ───────────────────────────────
    common_feats = set(FEATURES_LIST)
    common_feats &= set(train_data['valid_features'])
    common_feats &= set(ext_data['valid_features'])
    common_feats  = sorted(common_feats)
    
    # Map to pretty names for the charts
    pretty_feats = [FEATURE_NAMES.get(f, f) for f in common_feats]

    print(f"\nCommon features ({len(common_feats)}): {common_feats}")

    # ── 5. Normalize train and test sets ──────────────────────
    idx_train = [train_data['valid_features'].index(f) for f in common_feats]
    X_train   = rank_normalize(train_data['X'][:, idx_train])
    y_train   = rank_normalize(train_data['y'])

    idx_ext = [ext_data['valid_features'].index(f) for f in common_feats]
    X_test  = rank_normalize(ext_data['X'][:, idx_ext])
    y_test  = rank_normalize(ext_data['y'])

    # ── 6. Fit & Evaluate Models ──────────────────────────────
    print("\n--- Training on 500M ---")

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


    # ── 7. Plot Linear Regression Coefficients ────────────────
    print("\nGenerating Linear Regression Importance Chart...")
    
    # Extract coefficients from the pipeline
    lin_model = lin.named_steps['linearregression']
    coefs = lin_model.coef_
    
    # Sort by absolute magnitude
    sort_idx_lin = np.argsort(np.abs(coefs))
    sorted_coefs = coefs[sort_idx_lin]
    sorted_feats_lin = np.array(pretty_feats)[sort_idx_lin]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_feats_lin, sorted_coefs, color=C_LIN, edgecolor='white')
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel("Coefficient Value (Standardized Input)")
    ax.set_title("Linear Regression Coefficients (500M → GPT-J)", fontweight='bold')
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ha = 'left' if width > 0 else 'right'
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', ha=ha, fontsize=9)

    ax.grid(axis='x', alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig('feature_importance_linear.png', dpi=300)
    plt.close(fig)


    # ── 8. Plot Random Forest Permutation Importance ──────────
    print("Generating Random Forest Permutation Importance Chart...")
    
    # Calculate permutation importance on the test set
    perm_result = permutation_importance(
        rf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=4
    )
    
    # Sort by mean decrease in R2
    sort_idx_rf = perm_result.importances_mean.argsort()
    sorted_importances = perm_result.importances_mean[sort_idx_rf]
    sorted_std = perm_result.importances_std[sort_idx_rf]
    sorted_feats_rf = np.array(pretty_feats)[sort_idx_rf]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot horizontal bars with error bars
    bars = ax.barh(
        sorted_feats_rf, 
        sorted_importances, 
        xerr=sorted_std, 
        color=C_RF, 
        alpha=0.8,
        capsize=4,
        error_kw={'elinewidth': 1.5, 'alpha': 0.7}
    )
    
    ax.axvline(0, color='black', lw=0.8, linestyle='--')
    ax.set_xlabel("Mean decrease in R² (permutation, held-out test)")
    ax.set_title("RF Permutation Importance (500M → GPT-J)", fontweight='bold')

    # Add data labels slightly offset from the error bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        err = sorted_std[i]
        ax.text(width + err + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', va='center', ha='left', fontsize=9)

    ax.grid(axis='x', alpha=0.25, linestyle='--')
    
    # Extend x-axis slightly so labels don't get cut off
    max_val = max(sorted_importances + sorted_std)
    ax.set_xlim(left=-0.02, right=max_val + 0.05)

    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300)
    plt.close(fig)

    print("\nDone! Saved 'feature_importance_linear.png' and 'feature_importance_rf.png'.")