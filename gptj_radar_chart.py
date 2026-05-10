import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR           = '/home/nlp/katzela4/LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

CORE_MODELS = ['7M', '30M', '124M', '500M', 'GPT-J']

# Folder mapping for custom models.
MODEL_FOLDER_MAP = {
    '7M':   'MinGPT_Checkpoints_7M',
    '30M':  'MinGPT_Checkpoints_30M',
    '124M': 'MinGPT_Checkpoints_124M',
    '500M': 'MinGPT_Checkpoints_500M',
    'GPT-J': '', 
}

# Ground Truth Frequency Mapping
FREQ_FILE_MAP = {
    '7M':    os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M':  os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
    'GPT-J': os.path.join(BASE_DIR, 'data', 'datasets', 'pile_token_frequencies.csv'),
}

# The single 5-feature set
FEATURES = ['embedding_norm', 'logit_norm', 'weight_variance', 'weight_mean', 'l1_norm']

# Clean names for the chart labels
FEATURE_LABELS = ['Embedding Norm', 'Logit Norm', 'Weight Variance', 'Weight Mean', 'L1 Norm']

# Colors for the 5 different model lines on the chart
COLORS = {
    '7M': '#1f77b4',   # Blue
    '30M': '#ff7f0e',  # Orange
    '124M': '#2ca02c', # Green
    '500M': '#d62728', # Red
    'GPT-J': '#9467bd' # Purple
}

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
    folder = MODEL_FOLDER_MAP[size]
    
    if size == 'GPT-J':
        feat_file = 'gptj_model_features.csv'
    else:
        feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
        
    freq_path = FREQ_FILE_MAP[size]

    if not os.path.exists(feat_file):
        return None
    if freq_path not in freq_cache:
        return None

    # Load features
    header  = pd.read_csv(feat_file, nrows=0).columns.tolist()
    valid   = [f for f in FEATURES if f in header]
    usecols = ['token_id'] + valid
    df_feat = pd.read_csv(feat_file, usecols=usecols, low_memory=False)
    
    df_feat['token_id'] = pd.to_numeric(df_feat['token_id'], errors='coerce').astype('Int64')
    df_feat.dropna(subset=['token_id'], inplace=True)
    df_feat['token_id'] = df_feat['token_id'].astype(np.int64)

    # Load frequencies
    df_freq = freq_cache[freq_path].copy()
    df_freq['token_id'] = pd.to_numeric(df_freq['token_id'], errors='coerce').astype('Int64')
    df_freq.dropna(subset=['token_id'], inplace=True)
    df_freq['token_id'] = df_freq['token_id'].astype(np.int64)

    # Merge
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
    print("Loading datasets and extracting features...")
    needed_paths = set(FREQ_FILE_MAP[m] for m in CORE_MODELS)
    freq_cache   = {path: pd.read_csv(path, usecols=['token_id', 'log_count'] if 'log_count' in pd.read_csv(path, nrows=0).columns else ['token_id', 'count']) for path in needed_paths if os.path.exists(path)}

    all_data = {}
    for size in CORE_MODELS:
        raw_data = load_model_data(size, freq_cache)
        if raw_data:
            idx = [raw_data['valid_features'].index(f) for f in FEATURES]
            all_data[size] = {
                'X': rank_normalize(raw_data['X'][:, idx]),
                'y': rank_normalize(raw_data['y'])
            }
            print(f"  ✅ {size} loaded.")

    del freq_cache
    gc.collect()

    print("\nTraining models to extract Feature Importances...")
    
    # Dictionaries to hold the extracted weights/importances
    importance_lr = {}
    importance_rf = {}

    for train_model in CORE_MODELS:
        if train_model not in all_data: continue
        
        X_train = all_data[train_model]['X']
        y_train = all_data[train_model]['y']

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # 1. Linear Regression
        lr = LinearRegression().fit(X_train_s, y_train)
        # We take the absolute value because coefficients can be negative, 
        # and we want to plot the *magnitude* of the feature's impact on a spider chart starting from 0.
        importance_lr[train_model] = np.abs(lr.coef_)

        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_features='sqrt', n_jobs=-1, random_state=42).fit(X_train_s, y_train)
        # Random Forest naturally outputs importances between 0.0 and 1.0
        importance_rf[train_model] = rf.feature_importances_

    # ──────────────────────────────────────────────────────────────
    # SPIDER CHART PLOTTING
    # ──────────────────────────────────────────────────────────────
    print("\nGenerating Spider Charts...")

    # Calculate angles for the radar chart axes
    num_vars = len(FEATURES)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the loop to connect the last point back to the first
    angles += angles[:1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(polar=True))

    def plot_radar(ax, data_dict, title):
        # Set up the axes
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(FEATURE_LABELS, fontsize=10, fontweight='bold')
        
        # Plot each model's line
        for model_name in CORE_MODELS:
            if model_name in data_dict:
                # Get values and close the loop
                values = data_dict[model_name].tolist()
                values += values[:1]
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=COLORS[model_name])
                ax.fill(angles, values, color=COLORS[model_name], alpha=0.1)
        
        ax.set_title(title, size=16, fontweight='bold', y=1.1)

    # Plot Linear Regression (Absolute Coefficients)
    plot_radar(ax1, importance_lr, "Linear Regression\n(Absolute Coefficients)")

    # Plot Random Forest (Gini Feature Importance)
    plot_radar(ax2, importance_rf, "Random Forest\n(Feature Importances)")

    # Add a single master legend outside the plots
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=12, title="Model Size", title_fontsize='13')

    plt.tight_layout()
    plt.savefig("feature_importance_spider_chart.png", dpi=300, bbox_inches='tight')
    print("  ✅ Saved chart: 'feature_importance_spider_chart.png'")
    print("\nDone.")