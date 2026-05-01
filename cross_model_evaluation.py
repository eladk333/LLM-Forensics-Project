import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import os

# ──────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────
MODEL_SIZES = [
    '7M', '30M', '124M', '500M',
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

# Local paths
BASE_DIR = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

# Maps model size -> subfolder name
MODEL_FOLDER_MAP = {
    '7M':   'MinGPT_Checkpoints_7M',
    '30M':  'MinGPT_Checkpoints_30M',
    '124M': 'MinGPT_Checkpoints_124M',
    '500M': 'MinGPT_Checkpoints_500M',
    '7M_owt_24M':    '7M_owt_24M',
    '30M_owt_24M':   '30M_owt_24M',
    '30M_owt_100M':  '30M_owt_100M',
    '30M_owt_420M':  '30M_owt_420M',
    '124M_owt_420M': '124M_owt_420M',
}

# Map each model to its specific token frequency dataset
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
}

FEATURE_CONFIG = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# ──────────────────────────────────────────────
# STORAGE
# ──────────────────────────────────────────────
global_dataframes = {}
results = {}

# Accumulates (feature_name -> list of importance scores) across all LOO folds
# for the RF model, so we can aggregate and plot at the end.
rf_importance_accumulator = {}   # { feature_name: [imp_fold1, imp_fold2, ...] }

# ──────────────────────────────────────────────
# ML HELPERS
# ──────────────────────────────────────────────

def _fit_and_evaluate(X_train, y_train, X_test, y_test, feature_names):
    """
    Train Linear + MLP + Random Forest on (X_train, y_train)
    and evaluate on (X_test, y_test).

    Also accumulates per-feature permutation importances from the RF
    into the global rf_importance_accumulator.

    Returns a dict with predictions and R² scores for all three models.
    """
    if X_train.shape[1] == 0:
        return None

    # ── Linear ──────────────────────────────────────────────────────────────
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2   = r2_score(y_test, lin_pred)

    # ── MLP ─────────────────────────────────────────────────────────────────
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
    mlp_pred = mlp.predict(X_test)
    mlp_r2   = r2_score(y_test, mlp_pred)

    # ── Random Forest ────────────────────────────────────────────────────────
    # No StandardScaler needed for RF (tree-based, scale-invariant).
    # n_estimators=300 gives stable estimates without being too slow.
    # max_features='sqrt' mirrors the paper's RF setup and reduces variance.
    rf = RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        min_samples_leaf=5,   # mild regularisation to avoid overfitting small folds
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2   = r2_score(y_test, rf_pred)

    # -- Permutation importance on the held-out test set (mirrors paper §5) --
    # We use permutation importance rather than impurity-based importance so
    # that the scores reflect generalisation performance, not training fit.
    perm = permutation_importance(
        rf, X_test, y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    for i, fname in enumerate(feature_names):
        rf_importance_accumulator.setdefault(fname, [])
        # mean importance over the 10 permutation repeats for this fold
        rf_importance_accumulator[fname].append(perm.importances_mean[i])

    return {
        'X_test':    X_test,
        'y_test':    y_test,
        'lin_pred':  lin_pred,
        'mlp_pred':  mlp_pred,
        'rf_pred':   rf_pred,
        'lin_r2':    lin_r2,
        'mlp_r2':    mlp_r2,
        'rf_r2':     rf_r2,
    }


# ──────────────────────────────────────────────
# LOO TRAINING
# ──────────────────────────────────────────────

def train_loo():
    """
    Leave-One-Out Evaluation:
    For every model in MODEL_SIZES:
        - Train on the concatenated data of ALL OTHER models.
        - Test exclusively on the left-out model's data.
    """
    for loo_model in MODEL_SIZES:
        if loo_model not in global_dataframes:
            print(f"  ⚠ {loo_model} data missing — skipping as LOO target.")
            continue

        train_models = [m for m in MODEL_SIZES if m != loo_model and m in global_dataframes]

        if not train_models:
            print(f"  ⚠ Not enough models to train for {loo_model}.")
            continue

        # Find common features across all training models + the test model
        common_feats = set(FEATURES_LIST)
        for m in train_models + [loo_model]:
            common_feats = common_feats.intersection(global_dataframes[m]['valid_features'])
        common_feats = sorted(common_feats)   # sorted for reproducibility

        if not common_feats:
            print(f"  ⚠ No common features for LOO {loo_model}.")
            results[loo_model] = None
            continue

        # Compile Training Data (Concatenate all other models)
        X_tr = pd.concat(
            [global_dataframes[m]['df'][common_feats] for m in train_models],
            ignore_index=True
        ).values
        y_tr = pd.concat(
            [global_dataframes[m]['y'] for m in train_models],
            ignore_index=True
        ).values

        # Compile Test Data (The isolated model)
        X_te = global_dataframes[loo_model]['df'][common_feats].values
        y_te = global_dataframes[loo_model]['y'].values

        print(f"  [LOO] Testing on {loo_model} "
              f"(trained on {len(train_models)} other models, "
              f"{len(common_feats)} features)...")

        results[loo_model] = _fit_and_evaluate(X_tr, y_tr, X_te, y_te, common_feats)

        r = results[loo_model]
        if r:
            print(f"    Lin R²={r['lin_r2']:.4f}  "
                  f"MLP R²={r['mlp_r2']:.4f}  "
                  f"RF  R²={r['rf_r2']:.4f}")


# ──────────────────────────────────────────────
# CHART 1 – LOO bar chart (Lin / MLP / RF)
# ──────────────────────────────────────────────

def generate_loo_chart():
    """Generates a single comprehensive chart for all LOO results (3 bars per model)."""
    color_linear = '#78b79f'   # Teal/Green
    color_mlp    = '#eb9875'   # Orange/Peach
    color_rf     = '#7b9dd4'   # Steel Blue  ← new

    fig, ax = plt.subplots(figsize=(16, 7))

    test_labels  = []
    lin_r2_vals  = []
    mlp_r2_vals  = []
    rf_r2_vals   = []

    for loo_model in MODEL_SIZES:
        data = results.get(loo_model)
        if not data:
            continue

        ds_name = "OpenWebText" if 'owt' in loo_model or loo_model == '500M' else "Wiki"
        label   = f"Tested on:\n{loo_model}\n({ds_name})"

        test_labels.append(label)
        lin_r2_vals.append(data['lin_r2'])
        mlp_r2_vals.append(data['mlp_r2'])
        rf_r2_vals.append(data['rf_r2'])

    if not test_labels:
        print("No valid data to plot.")
        return

    x     = np.arange(len(test_labels))
    width = 0.25   # narrower now that we have 3 bars

    bars1 = ax.bar(x - width,       lin_r2_vals, width, label='Linear R²',        color=color_linear)
    bars2 = ax.bar(x,               mlp_r2_vals, width, label='MLP R²',            color=color_mlp)
    bars3 = ax.bar(x + width,       rf_r2_vals,  width, label='Random Forest R²',  color=color_rf)

    ax.set_xticks(x)
    ax.set_xticklabels(test_labels, fontsize=9.5, rotation=0)
    ax.set_ylabel("R-Squared (R²) Score", fontsize=12)

    # Dynamically set Y-limits
    all_vals = lin_r2_vals + mlp_r2_vals + rf_r2_vals
    min_y = min(0, min(all_vals) - 0.1)
    ax.set_ylim(min_y, 1.1)
    ax.axhline(0, color='black', lw=0.8)

    ax.set_title(
        "Leave-One-Out (LOO) Model Generalization\n"
        "(Models trained on all data EXCEPT target, then evaluated on target)",
        fontweight='bold', fontsize=14
    )
    ax.legend(title="Predictor Algorithm", fontsize=10)

    # Value labels on top/bottom of bars
    for bar in list(bars1) + list(bars2) + list(bars3):
        h      = bar.get_height()
        offset = 0.02 if h >= 0 else -0.05
        va     = 'bottom' if h >= 0 else 'top'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + offset,
            f'{h:.3f}',
            ha='center', va=va, fontsize=8
        )

    plt.tight_layout()
    out_filename = "cross_model_evaluation_LOO.png"
    plt.savefig(out_filename, dpi=300)
    print(f"\nSaved '{out_filename}' to disk.")
    plt.close(fig)


# ──────────────────────────────────────────────
# CHART 2 – RF Feature Importance (aggregated)
# ──────────────────────────────────────────────

def generate_feature_importance_chart():
    """
    Aggregates permutation importances collected across all LOO folds
    and plots them as a horizontal bar chart with error bars,
    mirroring Figure 12 of the paper (Merullo et al., ICLR 2025).
    """
    if not rf_importance_accumulator:
        print("No RF importance data to plot.")
        return

    # Use the pretty display names where available
    feature_display = {k: v for k, v in FEATURE_CONFIG.items()}

    features   = sorted(rf_importance_accumulator.keys())
    means      = np.array([np.mean(rf_importance_accumulator[f]) for f in features])
    stds       = np.array([np.std(rf_importance_accumulator[f])  for f in features])
    labels     = [feature_display.get(f, f) for f in features]

    # Sort by mean importance (ascending so the most important is at the top)
    order  = np.argsort(means)
    means  = means[order]
    stds   = stds[order]
    labels = [labels[i] for i in order]

    # ── colour: highlight the top feature (mirrors paper's Figure 12 style) ──
    bar_colors = ['#7b9dd4'] * len(means)
    bar_colors[-1] = '#2c5f9e'   # darkest blue for the most important feature

    fig, ax = plt.subplots(figsize=(9, 5))

    y_pos = np.arange(len(means))
    bars  = ax.barh(
        y_pos, means,
        xerr=stds,
        color=bar_colors,
        capsize=4,
        edgecolor='white',
        linewidth=0.6,
        error_kw=dict(elinewidth=1.2, ecolor='#444444'),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Mean Decrease in R² (permutation importance, held-out test set)", fontsize=10)
    ax.set_title(
        "Random Forest Feature Importance\n"
        "(Aggregated across all LOO folds via permutation importance)",
        fontweight='bold', fontsize=13
    )
    ax.axvline(0, color='black', lw=0.8, ls='--')

    # Annotate each bar with its mean value
    for bar, m in zip(bars, means):
        ax.text(
            max(m, 0) + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f'{m:.4f}',
            va='center', ha='left', fontsize=9, color='#333333'
        )

    # Add a small note referencing the paper's methodology
    ax.text(
        0.98, 0.02,
        "Permutation importance on held-out test set\n(cf. Merullo et al., ICLR 2025 §5)",
        transform=ax.transAxes,
        ha='right', va='bottom', fontsize=7.5, color='grey',
        style='italic'
    )

    plt.tight_layout()
    out_filename = "rf_feature_importance.png"
    plt.savefig(out_filename, dpi=300)
    print(f"Saved '{out_filename}' to disk.")
    plt.close(fig)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':

    print("Initializing environment and caching frequencies...")
    df_freq_cache = {}
    for freq_path in set(FREQ_FILE_MAP.values()):
        if not os.path.exists(freq_path):
            print(f"❌ Frequency file not found: {freq_path}")
            continue
        df_freq_cache[freq_path] = pd.read_csv(freq_path)
        print(f"  ✅ Cached: {os.path.basename(freq_path)}")

    # ── Load each model's feature CSV ──────────────────────────────────────
    for size in MODEL_SIZES:
        folder    = MODEL_FOLDER_MAP.get(size, size)
        feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')

        if not os.path.exists(feat_file):
            print(f"  ⚠ Feature file not found for {size}: {feat_file}  — skipping.")
            continue

        df_feat = pd.read_csv(feat_file)
        valid   = [f for f in FEATURES_LIST if f in df_feat.columns]

        if FREQ_FILE_MAP[size] not in df_freq_cache:
            continue

        df_freq = df_freq_cache[FREQ_FILE_MAP[size]]
        df      = pd.merge(df_freq, df_feat, on='token_id', how='inner')

        y = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid}
        print(f"  Loaded {size}: {len(df):,} tokens, {len(valid)} features.")

    if not global_dataframes:
        print("No model data loaded. Exiting.")
        exit(1)

    print("\n── Training phase (Leave-One-Out) ──────────────────────")
    train_loo()
    print("── Training complete ───────────────────────────────────")

    # ── Chart 1: LOO R² comparison (Linear / MLP / RF) ────────────────────
    generate_loo_chart()

    # ── Chart 2: RF feature importance (aggregated across folds) ──────────
    generate_feature_importance_chart()

    print("\nDone.")