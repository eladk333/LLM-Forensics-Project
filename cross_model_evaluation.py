import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# ──────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────
MODEL_SIZES = ['7M', '30M', '124M', '500M']

# Local paths
BASE_DIR = r'C:\Users\elad.k.int\LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

# Map each model to its specific token frequency dataset
FREQ_FILE_MAP = {
    '7M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M': os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M': os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
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

# ──────────────────────────────────────────────
# ML HELPERS
# ──────────────────────────────────────────────

def _fit_and_evaluate(X_train, y_train, X_test, y_test):
    """Train Linear + MLP on (X_train, y_train) and evaluate on (X_test, y_test)."""
    if X_train.shape[1] == 0:
        return None

    # ── Linear ──
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_pred_raw = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred_raw)

    # ── MLP ──
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='tanh',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
    )
    mlp.fit(X_train, y_train)
    mlp_pred_raw = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred_raw)

    if X_test.shape[1] == 1:
        idx = X_test.iloc[:, 0].argsort()
        return {
            'X_test':        X_test.iloc[idx],
            'y_test':        y_test.iloc[idx],
            'lin_pred':      lin_pred_raw[idx],
            'mlp_pred':      mlp_pred_raw[idx],
            'lin_pred_raw':  lin_pred_raw,
            'mlp_pred_raw':  mlp_pred_raw,
            'y_test_raw':    y_test,
            'lin_r2':        lin_r2,
            'mlp_r2':        mlp_r2,
        }
    return {
        'X_test':       X_test,
        'y_test':       y_test,
        'lin_pred':     lin_pred_raw,
        'mlp_pred':     mlp_pred_raw,
        'lin_pred_raw': lin_pred_raw,
        'mlp_pred_raw': mlp_pred_raw,
        'y_test_raw':   y_test,
        'lin_r2':       lin_r2,
        'mlp_r2':       mlp_r2,
    }

def train_all():
    """
    For every train_model in MODEL_SIZES:
        - train the combined model on train_model's data (80/20 self-split)
        - test on train_model itself  (the held-out 20%)
        - test on every other model   (full dataset as test set)
    """
    for train_size in MODEL_SIZES:
        if train_size not in global_dataframes:
            print(f"  ⚠ {train_size} data missing — skipping as train source.")
            continue

        results[train_size] = {}

        gd_train      = global_dataframes[train_size]
        df_train_full = gd_train['df']
        y_train_full  = gd_train['y']
        feats_train   = gd_train['valid_features']

        X_all = df_train_full[feats_train]
        X_tr, X_held, y_tr, y_held = train_test_split(
            X_all, y_train_full, test_size=0.2, random_state=42
        )

        # ── self-test (held-out 20%) ──────────────────────────────────────
        print(f"  Training on {train_size}, testing on {train_size} (held-out 20%)...")
        results[train_size][train_size] = _fit_and_evaluate(X_tr, y_tr, X_held, y_held)
        r = results[train_size][train_size]
        if r:
            print(f"    Lin R2={r['lin_r2']:.4f}  MLP R2={r['mlp_r2']:.4f}")

        # ── cross-model tests ──────────────────────────────────────────────
        for test_size in MODEL_SIZES:
            if test_size == train_size:
                continue
            if test_size not in global_dataframes:
                print(f"  ⚠ {test_size} data missing — skipping as test target.")
                continue

            gd_test     = global_dataframes[test_size]
            df_test_all = gd_test['df']
            y_test_all  = gd_test['y']
            feats_test  = gd_test['valid_features']

            common = [f for f in FEATURES_LIST if f in set(feats_train) and f in set(feats_test)]
            if not common:
                print(f"  ⚠ No common features between {train_size} and {test_size}.")
                results[train_size][test_size] = None
                continue

            X_tr_common   = df_train_full[common].loc[X_tr.index]
            X_test_common = df_test_all[common]          

            print(f"  Training on {train_size}, testing on {test_size} ({len(common)} common features)...")
            results[train_size][test_size] = _fit_and_evaluate(
                X_tr_common, y_tr, X_test_common, y_test_all
            )
            r = results[train_size][test_size]
            if r:
                print(f"    Lin R2={r['lin_r2']:.4f}  MLP R2={r['mlp_r2']:.4f}")

# ──────────────────────────────────────────────
# GRAPH GENERATION
# ──────────────────────────────────────────────

def generate_individual_charts():
    """Generates 4 separate diagram files, one for each trained model."""
    # Colors matching the slide's aesthetic
    color_linear = '#78b79f'  # Teal/Green
    color_mlp = '#eb9875'     # Orange/Peach

    for train_model in MODEL_SIZES:
        train_results = results.get(train_model, {})
        if not train_results:
            print(f"No data to plot for {train_model}.")
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        test_labels = []
        lin_r2_vals = []
        mlp_r2_vals = []

        for test_model in MODEL_SIZES:
            data = train_results.get(test_model)
            if data is None:
                continue

            is_self = (test_model == train_model)
            # Guarantee no mistakes labeling datasets
            ds_name = "OpenWebText" if test_model == '500M' else "Wiki"
            
            # Formatted exactly to match the presentation slides
            label = f"{test_model}\n({'(20% Test Split)' if is_self else '(100% Data)'})\n{ds_name} Dataset"
            test_labels.append(label)
            lin_r2_vals.append(data['lin_r2'])
            mlp_r2_vals.append(data['mlp_r2'])

        x = np.arange(len(test_labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, lin_r2_vals, width, label='Linear_R2', color=color_linear)
        bars2 = ax.bar(x + width/2, mlp_r2_vals, width, label='MLP_R2', color=color_mlp)

        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, fontsize=10)
        ax.set_ylabel("R-Squared (R²) Score")
        
        # Dynamically set Y-limits based on the worst negative R2 score to prevent text cutoffs
        min_y = min(0, min(lin_r2_vals + mlp_r2_vals) - 0.2)
        ax.set_ylim(min_y, 1.1)
        ax.axhline(0, color='black', lw=0.8)
        
        ax.set_title(f"Cross-Model Generalization\n(Trained entirely on {train_model})", fontweight='bold')
        ax.legend(title="Predictor Algorithm", fontsize=9)

        # Add text labels on top/bottom of the bars
        for bar in bars1 + bars2:
            h = bar.get_height()
            offset = 0.02 if h >= 0 else -0.05
            val_align = 'bottom' if h >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f'{h:.3f}', ha='center', va=val_align, fontsize=9)

        plt.tight_layout()
        out_filename = f"cross_model_evaluation_trained_on_{train_model}.png"
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

    # ── Load each model's feature CSV ──
    for size in MODEL_SIZES:
        feat_file = os.path.join(
            BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'model_features.csv'
        )
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

    print("\n── Training phase ──────────────────────────────")
    train_all()
    print("── Training complete ───────────────────────────\n")

    # Generate the 4 distinct charts matching your presentation
    generate_individual_charts()