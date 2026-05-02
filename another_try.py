"""
cross_model_evaluation_normalized_only.py
─────────────────────────────────────
Extends the LOO evaluation with:
  1. Rank-normalization of features AND target per model (Raw data removed)
  2. Per-dataset-family regressors (Wiki / OWT)
  3. Normalized LOO chart (Solid colors)
  4. Cross-model generalization chart (Wiki→OWT, OWT→Wiki)
  5. RF feature importance chart (normalized, aggregated across folds)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
BASE_DIR          = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

MODEL_SIZES = [
    '7M', '30M', '124M', '500M',
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

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
    # Add GPT-J / OLMo here once you have extracted features + freq CSVs:
    # 'gptj_6B':  'gptj_6B',
    # 'olmo_7B':  'olmo_7B',
}

FREQ_FILE_MAP = {
    '7M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M': os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M': os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    '7M_owt_24M':    os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_24M':   os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_100M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_100M_cache_token_frequencies.csv'),
    '30M_owt_420M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    '124M_owt_420M': os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    # 'gptj_6B':  os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
    # 'olmo_7B':  os.path.join(BASE_DIR, 'data', 'datasets', 'dolma_token_frequencies.csv'),
}

FEATURE_CONFIG = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# Dataset family membership
WIKI_MODELS = ['7M', '30M', '124M']
OWT_MODELS  = ['500M', '7M_owt_24M', '30M_owt_24M',
                '30M_owt_100M', '30M_owt_420M', '124M_owt_420M']

# Colours
C_LIN = '#78b79f'   # teal
C_MLP = '#eb9875'   # orange
C_RF  = '#7b9dd4'   # steel blue

# ──────────────────────────────────────────────────────────────
# STORAGE
# ──────────────────────────────────────────────────────────────
global_dataframes = {}          # raw data
results_norm = {}               # LOO results, normalized features
rf_imp_norm  = {}               # permutation importances (normalized)


# ──────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ──────────────────────────────────────────────────────────────

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Convert a 1-D or 2-D array to per-column fractional ranks in [0, 1].
    This removes scale differences between models of different sizes.
    """
    if arr.ndim == 1:
        return (rankdata(arr) - 1) / (len(arr) - 1)
    # 2-D: normalize each column independently
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j])
        out[:, j] = (r - 1) / (len(r) - 1)
    return out


def normalize_model_data(X: np.ndarray, y: np.ndarray):
    """Rank-normalize both features and target for a single model."""
    return rank_normalize(X), rank_normalize(y)


# ──────────────────────────────────────────────────────────────
# ML CORE
# ──────────────────────────────────────────────────────────────

def _fit_and_evaluate(X_train, y_train, X_test, y_test,
                      feature_names, imp_store: dict):
    """
    Fit Linear + MLP + RF, evaluate on test set.
    Appends per-fold permutation importances into imp_store.
    """
    if X_train.shape[1] == 0:
        return None

    # ── Linear ────────────────────────────────────────────────
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_r2 = r2_score(y_test, lin.predict(X_test))

    # ── MLP ───────────────────────────────────────────────────
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            alpha=0.01,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            learning_rate='adaptive',
            random_state=42,
        )
    )
    mlp.fit(X_train, y_train)
    mlp_r2 = r2_score(y_test, mlp.predict(X_test))

    # ── Random Forest ─────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    # Permutation importance (held-out test set)
    perm = permutation_importance(rf, X_test, y_test,
                                  n_repeats=10, random_state=42, n_jobs=-1)
    for i, fname in enumerate(feature_names):
        imp_store.setdefault(fname, []).append(perm.importances_mean[i])

    return {'lin_r2': lin_r2, 'mlp_r2': mlp_r2, 'rf_r2': rf_r2}


# ──────────────────────────────────────────────────────────────
# LOO TRAINING  (normalized only)
# ──────────────────────────────────────────────────────────────

def train_loo(model_subset=None):
    """
    Run LOO for every model in model_subset (defaults to all loaded models).
    Populates results_norm.
    """
    targets = model_subset if model_subset else MODEL_SIZES

    for loo_model in targets:
        if loo_model not in global_dataframes:
            print(f"  ⚠ {loo_model} missing — skipping.")
            continue

        train_models = [m for m in targets
                        if m != loo_model and m in global_dataframes]
        if not train_models:
            continue

        # Common features
        common_feats = set(FEATURES_LIST)
        for m in train_models + [loo_model]:
            common_feats &= set(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)

        if not common_feats:
            print(f"  ⚠ No common features for {loo_model}.")
            results_norm[loo_model] = None
            continue

        # ── Normalized ────────────────────────────────────────
        # Rank-normalize each model's X and y independently, THEN concatenate.
        # This way scale and distribution differences are removed per model.
        X_tr_n_parts, y_tr_n_parts = [], []
        for m in train_models:
            Xm = global_dataframes[m]['df'][common_feats].values
            ym = global_dataframes[m]['y'].values
            Xm_n, ym_n = normalize_model_data(Xm, ym)
            X_tr_n_parts.append(Xm_n)
            y_tr_n_parts.append(ym_n)

        X_tr_n = np.concatenate(X_tr_n_parts)
        y_tr_n = np.concatenate(y_tr_n_parts)

        # Normalize test model independently
        X_te_r = global_dataframes[loo_model]['df'][common_feats].values
        y_te_r = global_dataframes[loo_model]['y'].values
        X_te_n, y_te_n = normalize_model_data(X_te_r, y_te_r)

        print(f"  [LOO norm] {loo_model} ← {len(train_models)} models, "
              f"{len(common_feats)} feats")
        results_norm[loo_model] = _fit_and_evaluate(
            X_tr_n, y_tr_n, X_te_n, y_te_n, common_feats, rf_imp_norm)
        r = results_norm[loo_model]
        if r:
            print(f"    Lin={r['lin_r2']:.3f}  MLP={r['mlp_r2']:.3f}  "
                  f"RF={r['rf_r2']:.3f}")


# ──────────────────────────────────────────────────────────────
# CROSS-FAMILY EVALUATION
# Trains on one dataset family, tests on the other
# ──────────────────────────────────────────────────────────────

cross_family_results = {}   # { 'wiki→owt': {model: {...}}, 'owt→wiki': {...} }

def train_cross_family():
    """
    For each model in the target family, train on ALL models of the source
    family (normalized), then test on the target model (normalized).
    Mirrors the paper's OLMo↔GPT-J generalization experiment.
    """
    experiments = [
        ('wiki→owt', WIKI_MODELS, OWT_MODELS),
        ('owt→wiki', OWT_MODELS,  WIKI_MODELS),
    ]

    for exp_name, source_family, target_family in experiments:
        cross_family_results[exp_name] = {}
        source_loaded = [m for m in source_family if m in global_dataframes]
        target_loaded = [m for m in target_family if m in global_dataframes]

        if not source_loaded or not target_loaded:
            print(f"  ⚠ Not enough models for {exp_name}.")
            continue

        # Common features across all source + target models
        common_feats = set(FEATURES_LIST)
        for m in source_loaded + target_loaded:
            common_feats &= set(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)

        if not common_feats:
            print(f"  ⚠ No common features for {exp_name}.")
            continue

        # Build normalized training set from entire source family
        X_src_parts, y_src_parts = [], []
        for m in source_loaded:
            Xm = global_dataframes[m]['df'][common_feats].values
            ym = global_dataframes[m]['y'].values
            Xm_n, ym_n = normalize_model_data(Xm, ym)
            X_src_parts.append(Xm_n)
            y_src_parts.append(ym_n)
        X_src = np.concatenate(X_src_parts)
        y_src = np.concatenate(y_src_parts)

        # Train once on the full source family
        rf_src = RandomForestRegressor(
            n_estimators=300, max_features='sqrt',
            min_samples_leaf=5, n_jobs=-1, random_state=42)
        lin_src = make_pipeline(StandardScaler(), LinearRegression())
        mlp_src = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                         alpha=0.01, max_iter=2000, early_stopping=True,
                         validation_fraction=0.15, n_iter_no_change=25,
                         learning_rate='adaptive', random_state=42))

        rf_src.fit(X_src, y_src)
        lin_src.fit(X_src, y_src)
        mlp_src.fit(X_src, y_src)

        print(f"\n  [Cross-family {exp_name}] "
              f"trained on {len(source_loaded)} models")

        # Evaluate on each target model independently
        for loo_model in target_loaded:
            X_te = global_dataframes[loo_model]['df'][common_feats].values
            y_te = global_dataframes[loo_model]['y'].values
            X_te_n, y_te_n = normalize_model_data(X_te, y_te)

            lin_r2 = r2_score(y_te_n, lin_src.predict(X_te_n))
            mlp_r2 = r2_score(y_te_n, mlp_src.predict(X_te_n))
            rf_r2  = r2_score(y_te_n, rf_src.predict(X_te_n))

            cross_family_results[exp_name][loo_model] = {
                'lin_r2': lin_r2, 'mlp_r2': mlp_r2, 'rf_r2': rf_r2}
            print(f"    → {loo_model:20s}  "
                  f"Lin={lin_r2:.3f}  MLP={mlp_r2:.3f}  RF={rf_r2:.3f}")


# ──────────────────────────────────────────────────────────────
# CHART 1  –  Normalized LOO (Solid bars)
# ──────────────────────────────────────────────────────────────

def _bar_group(ax, labels, norm_dict, title):
    """Helper: draw one LOO panel as a grouped bar chart with solid colors."""
    lin_n = [norm_dict[m]['lin_r2'] if norm_dict.get(m) else 0 for m in labels]
    mlp_n = [norm_dict[m]['mlp_r2'] if norm_dict.get(m) else 0 for m in labels]
    rf_n  = [norm_dict[m]['rf_r2']  if norm_dict.get(m) else 0 for m in labels]

    x     = np.arange(len(labels))
    w     = 0.25
    offsets = [-w, 0, w]
    vals    = [lin_n, mlp_n, rf_n]
    colors  = [C_LIN, C_MLP, C_RF]
    bar_labels = ['Linear', 'MLP', 'Random Forest']

    bars_all = []
    for offset, val, color, lbl in zip(offsets, vals, colors, bar_labels):
        b = ax.bar(x + offset, val, w,
                   label=lbl, color=color,
                   edgecolor='white', linewidth=0.5)
        bars_all.append(b)

    # Value labels
    for bars in bars_all:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.05:          # skip near-zero labels
                offset_y = 0.02 if h >= 0 else -0.07
                va       = 'bottom'    if h >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + offset_y, f'{h:.2f}',
                        ha='center', va=va, fontsize=8, rotation=90)

    all_vals = [v for lst in vals for v in lst]
    min_y = min(0, min(all_vals) - 0.15) if all_vals else 0
    ax.set_ylim(min_y, 1.15)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x)
    tick_labels = []
    for m in labels:
        ds = 'OWT' if ('owt' in m or m == '500M') else 'Wiki'
        tick_labels.append(f"{m}\n({ds})")
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')


def generate_loo_comparison_chart():
    valid = [m for m in MODEL_SIZES if results_norm.get(m)]
    if not valid:
        print("No LOO results to plot.")
        return

    fig, ax = plt.subplots(figsize=(15, 6))
    _bar_group(ax, valid, results_norm,
               "LOO Generalization: Rank-Normalized Features")

    plt.tight_layout()
    fname = "loo_normalized_only.png"
    plt.savefig(fname, dpi=300)
    print(f"\nSaved '{fname}'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CHART 2  –  Cross-family generalization
# ──────────────────────────────────────────────────────────────

def generate_cross_family_chart():
    if not cross_family_results:
        print("No cross-family results to plot.")
        return

    exp_keys = list(cross_family_results.keys())
    n_panels = len(exp_keys)

    fig, axes = plt.subplots(1, n_panels, figsize=(11 * n_panels, 6),
                             sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, exp_name in zip(axes, exp_keys):
        res   = cross_family_results[exp_name]
        models = list(res.keys())
        if not models:
            continue

        x   = np.arange(len(models))
        w   = 0.25
        lin = [res[m]['lin_r2'] for m in models]
        mlp = [res[m]['mlp_r2'] for m in models]
        rf  = [res[m]['rf_r2']  for m in models]

        b1 = ax.bar(x - w,   lin, w, color=C_LIN, label='Linear')
        b2 = ax.bar(x,       mlp, w, color=C_MLP, label='MLP')
        b3 = ax.bar(x + w,   rf,  w, color=C_RF,  label='Random Forest')

        for bars in [b1, b2, b3]:
            for bar in bars:
                h  = bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                oy = 0.02     if h >= 0 else -0.05
                ax.text(bar.get_x() + bar.get_width()/2,
                        h + oy, f'{h:.3f}',
                        ha='center', va=va, fontsize=8)

        all_vals = lin + mlp + rf
        ax.set_ylim(min(0, min(all_vals) - 0.1), 1.12)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_ylabel("R² Score (on normalized target)", fontsize=10)
        src, tgt = exp_name.split('→')
        ax.set_title(
            f"Cross-Family: Train on {src.upper()} → Test on {tgt.upper()}\n"
            f"(rank-normalized features + target)",
            fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.suptitle(
        "Cross-Dataset-Family Generalization\n"
        "(Mirrors paper §5.3: train on one distribution, evaluate on another)",
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = "cross_family_generalization.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    print(f"Saved '{fname}'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CHART 3  –  RF Feature Importance (normalized)
# ──────────────────────────────────────────────────────────────

def generate_feature_importance_chart():
    if not rf_imp_norm:
        print("No importance data.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    imp_store = rf_imp_norm
    title = "RF Permutation Importance (Normalized Features)"

    features = sorted(imp_store.keys())
    means    = np.array([np.mean(imp_store[f]) for f in features])
    stds     = np.array([np.std(imp_store[f])  for f in features])
    labels   = [FEATURE_CONFIG.get(f, f) for f in features]

    order  = np.argsort(means)
    means  = means[order]
    stds   = stds[order]
    labels = [labels[i] for i in order]

    colors        = ['#7b9dd4'] * len(means)
    colors[-1]    = '#2c5f9e'   # top feature highlighted

    y_pos = np.arange(len(means))
    ax.barh(y_pos, means, xerr=stds,
            color=colors, capsize=4,
            edgecolor='white', linewidth=0.5,
            error_kw=dict(elinewidth=1.2, ecolor='#444'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean decrease in R² (permutation, held-out test)", fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.axvline(0, color='black', lw=0.7, ls='--')
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(max(m, 0) + 0.001, i, f'{m:.4f}',
                va='center', ha='left', fontsize=9, color='#333')

    plt.tight_layout()
    fname = "rf_feature_importance_normalized.png"
    plt.savefig(fname, dpi=300)
    print(f"Saved '{fname}'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load frequency files ──────────────────────────────────
    print("Caching frequency files...")
    df_freq_cache = {}
    for path in set(FREQ_FILE_MAP.values()):
        if not os.path.exists(path):
            print(f"  ❌ Not found: {path}")
            continue
        df_freq_cache[path] = pd.read_csv(path)
        print(f"  ✅ {os.path.basename(path)}")

    # ── Load model feature CSVs ───────────────────────────────
    for size in MODEL_SIZES:
        folder    = MODEL_FOLDER_MAP.get(size, size)
        feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')

        if not os.path.exists(feat_file):
            print(f"  ⚠ Feature file missing for {size} — skipping.")
            continue
        if FREQ_FILE_MAP[size] not in df_freq_cache:
            print(f"  ⚠ Frequency file missing for {size} — skipping.")
            continue

        df_feat = pd.read_csv(feat_file)
        valid   = [f for f in FEATURES_LIST if f in df_feat.columns]

        df_freq = df_freq_cache[FREQ_FILE_MAP[size]]
        df      = pd.merge(df_freq, df_feat, on='token_id', how='inner')
        y       = df['log_count'] if 'log_count' in df.columns \
                  else np.log1p(df['count'])

        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid}
        print(f"  Loaded {size}: {len(df):,} tokens, {len(valid)} features.")

    if not global_dataframes:
        print("No data loaded. Exiting.")
        exit(1)

    # ── Step 1: Full LOO (normalized only) ───────────────────
    print("\n── LOO evaluation (normalized only) ────────────────────")
    train_loo()

    # ── Step 2: Cross-family evaluation ──────────────────────
    print("\n── Cross-family evaluation ─────────────────────────────")
    train_cross_family()

    # ── Charts ───────────────────────────────────────────────
    print("\n── Generating charts ───────────────────────────────────")
    generate_loo_comparison_chart()
    generate_cross_family_chart()
    generate_feature_importance_chart()

    print("\nDone.")
    print("\nTo add GPT-J / OLMo:")
    print("  1. Run extract_features_gptj_olmo.py to get model_features.csv")
    print("  2. Get token frequency CSVs (WIMBD or streaming)")
    print("  3. Uncomment the gptj_6B / olmo_7B entries in MODEL_FOLDER_MAP")
    print("     and FREQ_FILE_MAP, then add them to OWT_MODELS list")
    print("  4. Re-run — cross_family_generalization.png will include them")