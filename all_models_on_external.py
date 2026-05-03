"""
external_generalization_only.py
────────────────────────────────
Train Linear / MLP / RF on all 9 existing models (rank-normalized),
then test on GPT-J (or OLMo) only.
No LOO, no cross-family — just the external generalization step.
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
EXTERNAL_MODEL = 'olmo_7B'

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

FEATURES_LIST = ['embedding_norm', 'logit_norm', 'weight_variance', 'weight_mean', 'l1_norm']
FEATURE_LABELS = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
}

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

    # Peek at columns first, then load only what we need — saves RAM on wide CSVs
    header  = pd.read_csv(feat_file, nrows=0).columns.tolist()
    valid   = [f for f in FEATURES_LIST if f in header]
    usecols = ['token_id'] + valid
    df_feat = pd.read_csv(feat_file, usecols=usecols, low_memory=False)

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

    import sys, resource

    def ram():
        """Current RSS in MB (works without psutil)."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024**2
        except ImportError:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def checkpoint(label):
        sys.stdout.write(f"  [MEM] {label}: {ram():.0f} MB\n")
        sys.stdout.flush()

    checkpoint("startup")

    # ── 1. Cache frequency files ──────────────────────────────
    print("Loading frequency files...")
    needed_paths = set(FREQ_FILE_MAP[m] for m in TRAIN_MODEL_SIZES + [EXTERNAL_MODEL])
    freq_cache   = {}
    for path in needed_paths:
        if not os.path.exists(path):
            print(f"  ❌ Not found: {path}")
            continue
        # Only load the columns we actually use from frequency files
        freq_header = pd.read_csv(path, nrows=0).columns.tolist()
        freq_cols   = ['token_id']
        if 'log_count' in freq_header:
            freq_cols.append('log_count')
        elif 'count' in freq_header:
            freq_cols.append('count')
        freq_cache[path] = pd.read_csv(path, usecols=freq_cols)
        print(f"  ✅ {os.path.basename(path)}  ({len(freq_cache[path]):,} rows, cols: {freq_cols})")
        checkpoint(f"after loading {os.path.basename(path)}")

    # ── 2. Load training models ───────────────────────────────
    print("\nLoading training models...")
    train_data = {}
    for size in TRAIN_MODEL_SIZES:
        result = load_model(size, freq_cache)
        if result:
            train_data[size] = result
            checkpoint(f"after loading {size}")

    # ── 3. Load external (test) model ────────────────────────
    print(f"\nLoading external model ({EXTERNAL_MODEL})...")
    checkpoint("before ext model load")
    ext_data = load_model(EXTERNAL_MODEL, freq_cache)
    checkpoint("after ext model load")

    # Frequency files no longer needed
    del freq_cache
    gc.collect()
    print("  (frequency cache freed)")
    checkpoint("after freq cache freed")

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

    # ── 5. Normalize & cache test set BEFORE building training set ──
    # Do this first so we can free ext_data['X'] raw array from RAM,
    # then build the large training concat without competing with it.
    print("\nNormalizing test set...")
    idx_ext = [ext_data['valid_features'].index(f) for f in common_feats]
    checkpoint("before rank_normalize test set")
    X_test  = rank_normalize(ext_data['X'][:, idx_ext])
    y_test  = rank_normalize(ext_data['y'])
    checkpoint("after rank_normalize test set")
    del ext_data   # raw ext arrays no longer needed
    gc.collect()
    checkpoint("after del ext_data")
    print(f"  Test set ({EXTERNAL_MODEL}): {X_test.shape[0]:,} rows")

    # ── 6. Build training set (rank-normalize per model) ─────
    print("\nBuilding training set...")
    X_parts, y_parts = [], []
    for size, d in train_data.items():
        idx  = [d['valid_features'].index(f) for f in common_feats]
        Xn   = rank_normalize(d['X'][:, idx])
        yn   = rank_normalize(d['y'])
        X_parts.append(Xn)
        y_parts.append(yn)

    checkpoint("before concat training")
    X_train = np.concatenate(X_parts);  del X_parts
    y_train = np.concatenate(y_parts);  del y_parts
    checkpoint("after concat training")
    # Free raw training arrays — only need normalized X_train/y_train now
    for d in train_data.values():
        del d['X'], d['y']
    del train_data
    gc.collect()
    checkpoint("after del train_data raw arrays")
    print(f"  Training set: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")

    # ── 7. Fit & evaluate ────────────────────────────────────
    print("\nFitting models...")
    checkpoint("before linear fit")

    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_r2 = r2_score(y_test, lin.predict(X_test))
    print(f"  Linear      R² = {lin_r2:.4f}")
    checkpoint("after linear")

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu',
            alpha=0.01, max_iter=2000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=25,
            learning_rate='adaptive', random_state=42,
        )
    )
    checkpoint("before mlp fit")
    mlp.fit(X_train, y_train)
    mlp_r2 = r2_score(y_test, mlp.predict(X_test))
    print(f"  MLP         R² = {mlp_r2:.4f}")
    checkpoint("after mlp")

    checkpoint("before RF fit")
    rf = RandomForestRegressor(
        n_estimators=100, max_features='sqrt',
        min_samples_leaf=5, n_jobs=4, random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))
    print(f"  RandomForest R² = {rf_r2:.4f}")
    checkpoint("after RF")

    # ── 8. Chart ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ['Linear', 'MLP', 'Random Forest']
    values = [lin_r2, mlp_r2, rf_r2]
    colors = [C_LIN,  C_MLP,  C_RF]

    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, values):
        va = 'bottom' if v >= 0 else 'top'
        oy = 0.02     if v >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2, v + oy,
                f'{v:.3f}', ha='center', va=va, fontsize=11, fontweight='bold')

    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylim(min(0, min(values) - 0.15), 1.10)
    ax.set_ylabel("R² Score (rank-normalized target)", fontsize=11)
    ax.set_title(
        f"Generalization to {EXTERNAL_MODEL}\n"
        f"Train: 9 existing models  |  Test: {EXTERNAL_MODEL}\n"
        f"(rank-normalized features + target,  {len(common_feats)} features)",
        fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_file = f"external_generalization_{EXTERNAL_MODEL}.png"
    plt.savefig(out_file, dpi=300)
    print(f"\nSaved '{out_file}'")
    plt.close(fig)

    print("\nDone.")
    print(f"  Linear       R² = {lin_r2:.4f}")
    print(f"  MLP          R² = {mlp_r2:.4f}")
    print(f"  RandomForest R² = {rf_r2:.4f}")