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

# ---------------------------------------------------------
# 1. Paths & Mappings
# ---------------------------------------------------------
BASE_DIR = r'/home/nlp/katzela4/LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

MODEL_SIZES = [
    '7M', '30M', '124M', '500M', 
    '7M_owt_24M', '30M_owt_24M', '30M_owt_100M', '30M_owt_420M', '124M_owt_420M'
]

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

global_storage = {}
global_dataframes = {}

# ---------------------------------------------------------
# 2. Training Logic
# ---------------------------------------------------------
def train_and_store(X_data, y_data):
    if X_data.shape[1] == 0:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)

    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=2000, 
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=42))

    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred)

    if X_test.shape[1] == 1:
        sort_idx = X_test.iloc[:, 0].argsort()
        X_test_sorted     = X_test.iloc[sort_idx]
        y_test_sorted     = y_test.iloc[sort_idx]
        lin_pred_sorted   = lin_pred[sort_idx]
        mlp_pred_sorted   = mlp_pred[sort_idx]
    else:
        X_test_sorted, y_test_sorted     = X_test, y_test
        lin_pred_sorted, mlp_pred_sorted = lin_pred, mlp_pred

    return {
        'X_test': X_test_sorted, 'y_test': y_test_sorted,
        'lin_pred': lin_pred_sorted, 'mlp_pred': mlp_pred_sorted,
        'lin_pred_raw': lin_pred, 'mlp_pred_raw': mlp_pred,
        'y_test_raw': y_test,
        'lin_r2': lin_r2, 'mlp_r2': mlp_r2
    }

# ---------------------------------------------------------
# 3. Static Report Generation
# ---------------------------------------------------------
def generate_static_reports():
    print("\n── Generating Static Dashboards ─────────────────────────")
    for size in MODEL_SIZES:
        if size not in global_storage or 'combined' not in global_storage[size] or global_storage[size]['combined'] is None:
            continue

        data = global_storage[size]['combined']
        model_folder = MODEL_FOLDER_MAP.get(size, size)
        save_path = os.path.join(BASE_MODELS_FOLDER, model_folder, f"{size}_analysis_dashboard.png")

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Model Analysis Summary: {size}", fontsize=20, weight='bold')

        # 1. Pred vs Actual (Linear)
        y_true   = data['y_test_raw']
        lin_pred = data['lin_pred_raw']
        mlp_pred = data['mlp_pred_raw']

        def plot_diag(ax):
            lims = [min(y_true.min(), ax.get_xlim()[0]), max(y_true.max(), ax.get_xlim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)

        sns.scatterplot(x=y_true, y=lin_pred, ax=axs[0, 0], alpha=0.3, color='blue')
        plot_diag(axs[0, 0])
        axs[0, 0].set_title(f"Combined Features: Linear Regression (R2: {data['lin_r2']:.3f})")
        axs[0, 0].set_xlabel("Actual Log Frequency")
        axs[0, 0].set_ylabel("Predicted Log Frequency")

        # 2. Pred vs Actual (MLP)
        sns.scatterplot(x=y_true, y=mlp_pred, ax=axs[0, 1], alpha=0.3, color='green')
        plot_diag(axs[0, 1])
        axs[0, 1].set_title(f"Combined Features: MLP Neural Net (R2: {data['mlp_r2']:.3f})")
        axs[0, 1].set_xlabel("Actual Log Frequency")
        axs[0, 1].set_ylabel("Predicted Log Frequency")

        # 3 & 4. Feature Spreads (Top 2 available features)
        active_feats = global_storage[size].get('active_combined_features', [])
        for i, ax in enumerate([axs[1, 0], axs[1, 1]]):
            if i < len(active_feats):
                feat = active_feats[i]
                label = FEATURE_CONFIG.get(feat, feat)
                col_data = data['X_test'][feat]
                
                sns.scatterplot(x=col_data, y=data['y_test'], ax=ax, alpha=0.2, color='gray', label='Actual')
                sns.scatterplot(x=col_data, y=data['mlp_pred'], ax=ax, alpha=0.6, color='red', s=15, label='MLP Pred')
                
                ax.set_title(f"Frequency Projection: {label}")
                ax.set_xlabel(label)
                ax.set_ylabel("Log Frequency")
                ax.legend()
            else:
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300)
        print(f"  ✅ Saved dashboard: {save_path}")
        plt.close(fig)

# ---------------------------------------------------------
# 4. Main Execution Setup
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Initializing environment and caching frequencies...")
    df_freq_cache = {}
    for freq_path in set(FREQ_FILE_MAP.values()):
        if not os.path.exists(freq_path):
            continue
        df_freq_cache[freq_path] = pd.read_csv(freq_path)
        print(f"  ✅ Cached: {os.path.basename(freq_path)}")

    for size in MODEL_SIZES:
        print(f"\nProcessing Model: {size}")
        folder = MODEL_FOLDER_MAP.get(size, size)
        feature_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')

        if not os.path.exists(feature_file) or FREQ_FILE_MAP[size] not in df_freq_cache:
            print(f"   ⚠️ Skipping {size}: Missing data files.")
            continue

        df_features    = pd.read_csv(feature_file)
        valid_features = [f for f in FEATURES_LIST if f in df_features.columns]
        df_freq = df_freq_cache[FREQ_FILE_MAP[size]]
        df = pd.merge(df_freq, df_features, on='token_id', how='inner')
        y = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid_features}
        global_storage[size]    = {'features': valid_features}

        print(f"   Training combined model...")
        global_storage[size]['combined'] = train_and_store(df[valid_features], y)
        global_storage[size]['active_combined_features'] = valid_features

    if not global_storage:
        print("\nNo models loaded. Check that feature files exist.")
        exit()

    generate_static_reports()