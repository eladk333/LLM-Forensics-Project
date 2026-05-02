"""
cross_model_evaluation_with_external.py
─────────────────────────────────────────
Extends cross_model_evaluation_normalized_only.py with:
  5. External-model generalization:
       Train on ALL existing Wiki + OWT models (normalized),
       then test separately on GPT-J and OLMo (also normalized).
     Produces: external_generalization.png
"""

import os
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

# ── Existing (training) models ────────────────────────────────
TRAIN_MODEL_SIZES = [
    '7M', '30M', '124M', '500M',
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

# ── External (test-only) models ───────────────────────────────
EXTERNAL_MODEL_SIZES = ['gptj_6B', 'olmo_7B']

ALL_MODEL_SIZES = TRAIN_MODEL_SIZES + EXTERNAL_MODEL_SIZES

MODEL_FOLDER_MAP = {
    # --- existing ---
    '7M':            'MinGPT_Checkpoints_7M',
    '30M':           'MinGPT_Checkpoints_30M',
    '124M':          'MinGPT_Checkpoints_124M',
    '500M':          'MinGPT_Checkpoints_500M',
    '7M_owt_24M':    '7M_owt_24M',
    '30M_owt_24M':   '30M_owt_24M',
    '30M_owt_100M':  '30M_owt_100M',
    '30M_owt_420M':  '30M_owt_420M',
    '124M_owt_420M': '124M_owt_420M',
    # --- external ---
    'gptj_6B':       'gptj_6B',
    'olmo_7B':       'olmo_7B',
}

FREQ_FILE_MAP = {
    # --- existing ---
    '7M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M': os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M': os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    '7M_owt_24M':    os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_24M':   os.path.join(BASE_DIR, 'data', 'datasets', 'owt_24M_cache_token_frequencies.csv'),
    '30M_owt_100M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_100M_cache_token_frequencies.csv'),
    '30M_owt_420M':  os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    '124M_owt_420M': os.path.join(BASE_DIR, 'data', 'datasets', 'owt_420M_cache_token_frequencies.csv'),
    # --- external: point these at whichever corpus GPT-J / OLMo were trained on ---
    'gptj_6B':  os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
    'olmo_7B':  os.path.join(BASE_DIR, 'data', 'datasets', 'dolma_token_frequencies.csv'),
}

FEATURE_CONFIG = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# Dataset-family labels for tick labels
FAMILY_LABEL = {
    '7M': 'Wiki', '30M': 'Wiki', '124M': 'Wiki',
    '500M': 'OWT',
    '7M_owt_24M': 'OWT', '30M_owt_24M': 'OWT',
    '30M_owt_100M': 'OWT', '30M_owt_420M': 'OWT', '124M_owt_420M': 'OWT',
    'gptj_6B': 'Pile', 'olmo_7B': 'Dolma',
}

# Colours
C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#7b9dd4'

# ──────────────────────────────────────────────────────────────
# GLOBAL STORAGE
# ──────────────────────────────────────────────────────────────
global_dataframes   = {}   # {model_name: {'df': pd.DataFrame, 'y': Series, 'valid_features': list}}
results_norm        = {}   # LOO results for existing models
rf_imp_norm         = {}   # permutation importances (LOO, normalized)
external_results    = {}   # {external_model: {'lin_r2', 'mlp_r2', 'rf_r2'}}


# ──────────────────────────────────────────────────────────────
# NORMALIZATION HELPERS
# ──────────────────────────────────────────────────────────────

def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Per-column fractional ranks → [0, 1], stored as float32 to save RAM."""
    if arr.ndim == 1:
        return ((rankdata(arr) - 1) / max(len(arr) - 1, 1)).astype(np.float32)
    out = np.empty(arr.shape, dtype=np.float32)
    for j in range(arr.shape[1]):
        r = rankdata(arr[:, j])
        out[:, j] = (r - 1) / max(len(r) - 1, 1)
    return out


def normalize_model_data(X: np.ndarray, y: np.ndarray):
    return rank_normalize(X), rank_normalize(y)


# ──────────────────────────────────────────────────────────────
# SHARED ML HELPER
# ──────────────────────────────────────────────────────────────

def _fit_and_evaluate(X_train, y_train, X_test, y_test,
                      feature_names, imp_store: dict):
    if X_train.shape[1] == 0:
        return None

    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_r2 = r2_score(y_test, lin.predict(X_test))

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

    rf = RandomForestRegressor(
        n_estimators=100,       # 300→100: 3× less RAM, negligible R² impact
        max_features='sqrt',
        min_samples_leaf=5,
        n_jobs=4,               # cap workers; -1 forks one per core → OOM
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_r2 = r2_score(y_test, rf.predict(X_test))

    perm = permutation_importance(rf, X_test, y_test,
                                  n_repeats=5, random_state=42, n_jobs=4)
    for i, fname in enumerate(feature_names):
        imp_store.setdefault(fname, []).append(perm.importances_mean[i])

    return {'lin_r2': lin_r2, 'mlp_r2': mlp_r2, 'rf_r2': rf_r2}


# ──────────────────────────────────────────────────────────────
# LOO  (existing models only)
# ──────────────────────────────────────────────────────────────

def train_loo(model_subset=None):
    targets = model_subset if model_subset else TRAIN_MODEL_SIZES

    for loo_model in targets:
        if loo_model not in global_dataframes:
            print(f"  ⚠ {loo_model} missing — skipping.")
            continue

        train_models = [m for m in targets
                        if m != loo_model and m in global_dataframes]
        if not train_models:
            continue

        common_feats = set(FEATURES_LIST)
        for m in train_models + [loo_model]:
            common_feats &= set(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)

        if not common_feats:
            results_norm[loo_model] = None
            continue

        X_tr_parts, y_tr_parts = [], []
        for m in train_models:
            Xm = global_dataframes[m]['df'][common_feats].values
            ym = global_dataframes[m]['y'].values
            Xn, yn = normalize_model_data(Xm, ym)
            X_tr_parts.append(Xn)
            y_tr_parts.append(yn)

        X_tr = np.concatenate(X_tr_parts)
        y_tr = np.concatenate(y_tr_parts)
        del X_tr_parts, y_tr_parts   # free per-model copies immediately

        X_te_n, y_te_n = normalize_model_data(
            global_dataframes[loo_model]['df'][common_feats].values,
            global_dataframes[loo_model]['y'].values
        )

        print(f"  [LOO] {loo_model} ← {len(train_models)} models, "
              f"{len(common_feats)} feats")
        results_norm[loo_model] = _fit_and_evaluate(
            X_tr, y_tr, X_te_n, y_te_n, common_feats, rf_imp_norm)

        r = results_norm[loo_model]
        if r:
            print(f"    Lin={r['lin_r2']:.3f}  MLP={r['mlp_r2']:.3f}  "
                  f"RF={r['rf_r2']:.3f}")


# ──────────────────────────────────────────────────────────────
# EXTERNAL GENERALIZATION
# Train on ALL existing models → test on GPT-J / OLMo separately
# ──────────────────────────────────────────────────────────────

def train_generalize_to_external():
    """
    1. Determine features common to all training models AND the external model.
       (Done per-external-model so we use as many features as possible.)
    2. Rank-normalize each training model independently, concatenate.
    3. Fit Linear, MLP, RF on the combined training set.
    4. Rank-normalize the external model, evaluate, store R².
    """
    train_loaded = [m for m in TRAIN_MODEL_SIZES if m in global_dataframes]
    if not train_loaded:
        print("  ⚠ No training models loaded.")
        return

    for ext_model in EXTERNAL_MODEL_SIZES:
        if ext_model not in global_dataframes:
            print(f"  ⚠ External model '{ext_model}' not loaded — skipping.")
            continue

        # Features available in every training model AND the external model
        common_feats = set(FEATURES_LIST)
        for m in train_loaded + [ext_model]:
            common_feats &= set(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)

        if not common_feats:
            print(f"  ⚠ No common features for {ext_model}.")
            external_results[ext_model] = None
            continue

        # ── Build training set ────────────────────────────────
        X_tr_parts, y_tr_parts = [], []
        for m in train_loaded:
            Xm = global_dataframes[m]['df'][common_feats].values
            ym = global_dataframes[m]['y'].values
            Xn, yn = normalize_model_data(Xm, ym)
            X_tr_parts.append(Xn)
            y_tr_parts.append(yn)

        X_tr = np.concatenate(X_tr_parts)
        y_tr = np.concatenate(y_tr_parts)
        del X_tr_parts, y_tr_parts   # free per-model copies

        # ── Normalize external model ──────────────────────────
        X_ext_n, y_ext_n = normalize_model_data(
            global_dataframes[ext_model]['df'][common_feats].values,
            global_dataframes[ext_model]['y'].values
        )

        print(f"\n  [External] Train on {len(train_loaded)} models "
              f"→ test on {ext_model}  ({len(common_feats)} feats)")

        # ── Fit ───────────────────────────────────────────────
        lin = make_pipeline(StandardScaler(), LinearRegression())
        lin.fit(X_tr, y_tr)
        lin_r2 = r2_score(y_ext_n, lin.predict(X_ext_n))

        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), activation='relu',
                alpha=0.01, max_iter=2000, early_stopping=True,
                validation_fraction=0.15, n_iter_no_change=25,
                learning_rate='adaptive', random_state=42,
            )
        )
        mlp.fit(X_tr, y_tr)
        mlp_r2 = r2_score(y_ext_n, mlp.predict(X_ext_n))

        rf = RandomForestRegressor(
            n_estimators=100, max_features='sqrt',
            min_samples_leaf=5, n_jobs=4, random_state=42,
        )
        rf.fit(X_tr, y_tr)
        rf_r2 = r2_score(y_ext_n, rf.predict(X_ext_n))

        external_results[ext_model] = {
            'lin_r2': lin_r2, 'mlp_r2': mlp_r2, 'rf_r2': rf_r2,
            'n_train_models': len(train_loaded),
            'n_features': len(common_feats),
            'features_used': common_feats,
        }
        print(f"    Lin={lin_r2:.3f}  MLP={mlp_r2:.3f}  RF={rf_r2:.3f}")


# ──────────────────────────────────────────────────────────────
# CROSS-FAMILY  (Wiki ↔ OWT, unchanged from original)
# ──────────────────────────────────────────────────────────────

WIKI_MODELS = ['7M', '30M', '124M']
OWT_MODELS  = ['500M', '7M_owt_24M', '30M_owt_24M',
                '30M_owt_100M', '30M_owt_420M', '124M_owt_420M']
cross_family_results = {}


def train_cross_family():
    experiments = [
        ('wiki→owt', WIKI_MODELS, OWT_MODELS),
        ('owt→wiki', OWT_MODELS,  WIKI_MODELS),
    ]
    for exp_name, source_family, target_family in experiments:
        cross_family_results[exp_name] = {}
        source_loaded = [m for m in source_family if m in global_dataframes]
        target_loaded = [m for m in target_family if m in global_dataframes]

        if not source_loaded or not target_loaded:
            continue

        common_feats = set(FEATURES_LIST)
        for m in source_loaded + target_loaded:
            common_feats &= set(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)
        if not common_feats:
            continue

        X_src_parts, y_src_parts = [], []
        for m in source_loaded:
            Xm = global_dataframes[m]['df'][common_feats].values
            ym = global_dataframes[m]['y'].values
            Xn, yn = normalize_model_data(Xm, ym)
            X_src_parts.append(Xn)
            y_src_parts.append(yn)
        X_src = np.concatenate(X_src_parts)
        y_src = np.concatenate(y_src_parts)
        del X_src_parts, y_src_parts   # free per-model copies

        rf_src  = RandomForestRegressor(n_estimators=100, max_features='sqrt',
                                        min_samples_leaf=5, n_jobs=4, random_state=42)
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

        print(f"\n  [Cross-family {exp_name}] trained on {len(source_loaded)} models")
        for loo_model in target_loaded:
            X_te_n, y_te_n = normalize_model_data(
                global_dataframes[loo_model]['df'][common_feats].values,
                global_dataframes[loo_model]['y'].values
            )
            lin_r2 = r2_score(y_te_n, lin_src.predict(X_te_n))
            mlp_r2 = r2_score(y_te_n, mlp_src.predict(X_te_n))
            rf_r2  = r2_score(y_te_n, rf_src.predict(X_te_n))
            cross_family_results[exp_name][loo_model] = {
                'lin_r2': lin_r2, 'mlp_r2': mlp_r2, 'rf_r2': rf_r2}
            print(f"    → {loo_model:20s}  "
                  f"Lin={lin_r2:.3f}  MLP={mlp_r2:.3f}  RF={rf_r2:.3f}")


# ──────────────────────────────────────────────────────────────
# CHART: LOO (normalized)
# ──────────────────────────────────────────────────────────────

def generate_loo_comparison_chart():
    valid = [m for m in TRAIN_MODEL_SIZES if results_norm.get(m)]
    if not valid:
        print("No LOO results to plot.")
        return

    x  = np.arange(len(valid))
    w  = 0.25
    lin = [results_norm[m]['lin_r2'] for m in valid]
    mlp = [results_norm[m]['mlp_r2'] for m in valid]
    rf  = [results_norm[m]['rf_r2']  for m in valid]

    fig, ax = plt.subplots(figsize=(15, 6))
    b1 = ax.bar(x - w,  lin, w, color=C_LIN, label='Linear',        edgecolor='white')
    b2 = ax.bar(x,      mlp, w, color=C_MLP, label='MLP',           edgecolor='white')
    b3 = ax.bar(x + w,  rf,  w, color=C_RF,  label='Random Forest', edgecolor='white')

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if abs(h) > 0.05:
                oy = 0.02 if h >= 0 else -0.07
                va = 'bottom' if h >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + oy, f'{h:.2f}',
                        ha='center', va=va, fontsize=8, rotation=90)

    all_vals = lin + mlp + rf
    ax.set_ylim(min(0, min(all_vals) - 0.15), 1.15)
    ax.axhline(0, color='black', lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{m}\n({FAMILY_LABEL.get(m, '?')})" for m in valid], fontsize=9)
    ax.set_ylabel("R² Score", fontsize=10)
    ax.set_title("LOO Generalization: Rank-Normalized Features",
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    plt.savefig("loo_normalized_only.png", dpi=300)
    print("Saved 'loo_normalized_only.png'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CHART: External generalization  ← NEW
# ──────────────────────────────────────────────────────────────

def generate_external_generalization_chart():
    """
    Grouped bar chart: one group per external model (GPT-J, OLMo),
    three bars per group (Linear / MLP / RF).
    Also overlays the mean LOO R² as a dashed reference line.
    """
    ext_loaded = [m for m in EXTERNAL_MODEL_SIZES
                  if external_results.get(m) is not None]
    if not ext_loaded:
        print("No external generalization results to plot.")
        return

    # Mean LOO performance as a reference baseline
    loo_means = {}
    for model_type in ('lin', 'mlp', 'rf'):
        vals = [results_norm[m][f'{model_type}_r2']
                for m in TRAIN_MODEL_SIZES
                if results_norm.get(m)]
        loo_means[model_type] = np.mean(vals) if vals else None

    x  = np.arange(len(ext_loaded))
    w  = 0.22
    lin_vals = [external_results[m]['lin_r2'] for m in ext_loaded]
    mlp_vals = [external_results[m]['mlp_r2'] for m in ext_loaded]
    rf_vals  = [external_results[m]['rf_r2']  for m in ext_loaded]

    fig, ax = plt.subplots(figsize=(max(7, len(ext_loaded) * 3.5), 6))

    b1 = ax.bar(x - w,  lin_vals, w, color=C_LIN, label='Linear',
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x,      mlp_vals, w, color=C_MLP, label='MLP',
                edgecolor='white', linewidth=0.5)
    b3 = ax.bar(x + w,  rf_vals,  w, color=C_RF,  label='Random Forest',
                edgecolor='white', linewidth=0.5)

    # Value labels on bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h  = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            oy = 0.02     if h >= 0 else -0.05
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + oy, f'{h:.3f}',
                    ha='center', va=va, fontsize=9, fontweight='bold')

    # LOO reference lines
    ref_styles = [
        ('lin', C_LIN, 'LOO avg – Linear'),
        ('mlp', C_MLP, 'LOO avg – MLP'),
        ('rf',  C_RF,  'LOO avg – RF'),
    ]
    for key, color, label in ref_styles:
        if loo_means.get(key) is not None:
            ax.axhline(loo_means[key], color=color, lw=1.5,
                       ls='--', alpha=0.7, label=label)

    all_vals = lin_vals + mlp_vals + rf_vals
    ax.set_ylim(min(0, min(all_vals) - 0.12), 1.10)
    ax.axhline(0, color='black', lw=0.8)

    tick_labels = [
        f"{m}\n({FAMILY_LABEL.get(m, '?')})\n"
        f"[{external_results[m]['n_features']} feats, "
        f"{external_results[m]['n_train_models']} train models]"
        for m in ext_loaded
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel("R² Score (rank-normalized target)", fontsize=10)
    ax.set_title(
        "External Generalization: Train on All Existing Models → Test on GPT-J / OLMo\n"
        "(rank-normalized features + target;  dashed = mean in-distribution LOO R²)",
        fontweight='bold', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    plt.savefig("external_generalization.png", dpi=300)
    print("Saved 'external_generalization.png'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CHART: Cross-family
# ──────────────────────────────────────────────────────────────

def generate_cross_family_chart():
    if not cross_family_results:
        return

    exp_keys = list(cross_family_results.keys())
    fig, axes = plt.subplots(1, len(exp_keys),
                             figsize=(11 * len(exp_keys), 6), sharey=False)
    if len(exp_keys) == 1:
        axes = [axes]

    for ax, exp_name in zip(axes, exp_keys):
        res    = cross_family_results[exp_name]
        models = list(res.keys())
        if not models:
            continue

        x   = np.arange(len(models))
        w   = 0.25
        lin = [res[m]['lin_r2'] for m in models]
        mlp = [res[m]['mlp_r2'] for m in models]
        rf  = [res[m]['rf_r2']  for m in models]

        b1 = ax.bar(x - w, lin, w, color=C_LIN, label='Linear')
        b2 = ax.bar(x,     mlp, w, color=C_MLP, label='MLP')
        b3 = ax.bar(x + w, rf,  w, color=C_RF,  label='Random Forest')

        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                oy = 0.02     if h >= 0 else -0.05
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + oy, f'{h:.3f}',
                        ha='center', va=va, fontsize=8)

        all_vals = lin + mlp + rf
        ax.set_ylim(min(0, min(all_vals) - 0.1), 1.12)
        ax.axhline(0, color='black', lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, fontsize=9)
        ax.set_ylabel("R² Score (normalized)", fontsize=10)
        src, tgt = exp_name.split('→')
        ax.set_title(
            f"Cross-Family: Train on {src.upper()} → Test on {tgt.upper()}\n"
            "(rank-normalized features + target)",
            fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.suptitle("Cross-Dataset-Family Generalization",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("cross_family_generalization.png", dpi=300, bbox_inches='tight')
    print("Saved 'cross_family_generalization.png'")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# CHART: RF Feature Importance
# ──────────────────────────────────────────────────────────────

def generate_feature_importance_chart():
    if not rf_imp_norm:
        return

    features = sorted(rf_imp_norm.keys())
    means    = np.array([np.mean(rf_imp_norm[f]) for f in features])
    stds     = np.array([np.std(rf_imp_norm[f])  for f in features])
    labels   = [FEATURE_CONFIG.get(f, f) for f in features]

    order  = np.argsort(means)
    means, stds = means[order], stds[order]
    labels = [labels[i] for i in order]

    colors     = ['#7b9dd4'] * len(means)
    colors[-1] = '#2c5f9e'

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(means)), means, xerr=stds,
            color=colors, capsize=4, edgecolor='white',
            error_kw=dict(elinewidth=1.2, ecolor='#444'))
    ax.set_yticks(np.arange(len(means)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean decrease in R² (permutation, held-out test)", fontsize=10)
    ax.set_title("RF Permutation Importance (Normalized Features)",
                 fontweight='bold', fontsize=12)
    ax.axvline(0, color='black', lw=0.7, ls='--')
    ax.grid(axis='x', alpha=0.25, linestyle='--')

    for i, (m, _) in enumerate(zip(means, stds)):
        ax.text(max(m, 0) + 0.001, i, f'{m:.4f}',
                va='center', ha='left', fontsize=9, color='#333')

    plt.tight_layout()
    plt.savefig("rf_feature_importance_normalized.png", dpi=300)
    print("Saved 'rf_feature_importance_normalized.png'")
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

    # ── Load all model CSVs ───────────────────────────────────
    for size in ALL_MODEL_SIZES:
        folder    = MODEL_FOLDER_MAP.get(size, size)
        feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
        freq_path = FREQ_FILE_MAP.get(size)

        if not os.path.exists(feat_file):
            print(f"  ⚠ Feature file missing for {size} — skipping.")
            continue
        if freq_path not in df_freq_cache:
            print(f"  ⚠ Frequency file missing for {size} — skipping.")
            continue

        df_feat = pd.read_csv(feat_file, low_memory=False)
        # Coerce token_id to int64 on both sides to avoid object/int64 merge error
        # (some feature CSVs store token_id as strings or mixed types)
        try:
            df_feat['token_id'] = pd.to_numeric(df_feat['token_id'], errors='coerce').astype('Int64')
            df_feat = df_feat.dropna(subset=['token_id'])
            df_feat['token_id'] = df_feat['token_id'].astype(np.int64)
        except Exception as e:
            print(f"  ⚠ Could not coerce token_id for {size}: {e} — skipping.")
            continue

        valid   = [f for f in FEATURES_LIST if f in df_feat.columns]

        df_freq = df_freq_cache[freq_path].copy()
        df_freq['token_id'] = pd.to_numeric(df_freq['token_id'], errors='coerce').astype('Int64')
        df_freq = df_freq.dropna(subset=['token_id'])
        df_freq['token_id'] = df_freq['token_id'].astype(np.int64)

        df      = pd.merge(df_freq, df_feat, on='token_id', how='inner')
        y       = (df['log_count'] if 'log_count' in df.columns
                   else np.log1p(df['count']))

        global_dataframes[size] = {
            'df': df, 'y': y, 'valid_features': valid,
            'is_external': size in EXTERNAL_MODEL_SIZES,
        }
        tag = '(external)' if size in EXTERNAL_MODEL_SIZES else ''
        print(f"  Loaded {size} {tag}: {len(df):,} tokens, {len(valid)} features.")

    if not global_dataframes:
        print("No data loaded. Exiting.")
        exit(1)

    # Frequency CSVs are no longer needed — free their RAM before training
    del df_freq_cache
    import gc; gc.collect()
    print(f"  (freed frequency cache; {len(global_dataframes)} models in memory)")

    # ── Step 1: LOO on existing models ───────────────────────
    print("\n── LOO evaluation (normalized, existing models) ────────")
    train_loo()

    # ── Step 2: Cross-family (Wiki ↔ OWT) ────────────────────
    print("\n── Cross-family evaluation ─────────────────────────────")
    train_cross_family()

    # ── Step 3: External generalization (→ GPT-J, OLMo) ─────
    print("\n── External generalization (GPT-J / OLMo) ──────────────")
    train_generalize_to_external()

    # ── Charts ───────────────────────────────────────────────
    print("\n── Generating charts ───────────────────────────────────")
    generate_loo_comparison_chart()
    generate_cross_family_chart()
    generate_external_generalization_chart()
    generate_feature_importance_chart()

    print("\nDone.")