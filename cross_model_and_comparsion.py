import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import rankdata

# scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ──────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR           = r''
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

TRAIN_MODEL = '7M'
TEST_MODELS = ['30M', '124M', '500M']

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

# Colors for charting
C_SK_LIN = '#78b79f'
C_PT_LIN = '#4a8b71'  
C_SK_MLP = '#eb9875'
C_PT_MLP = '#c46641'  
C_SK_RF  = '#7b9dd4'

# ──────────────────────────────────────────────────────────────
# PYTORCH MODELS & HELPERS
# ──────────────────────────────────────────────────────────────

class TorchLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

class TorchMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def train_torch_model(model, X_train, y_train, epochs=150, lr=0.01, batch_size=1024, weight_decay=0.0):
    """Generic training loop for PyTorch models."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Convert numpy arrays to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) # Reshape to match output
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    return model

def predict_torch(model, X_test):
    """Generate predictions from a trained PyTorch model."""
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).numpy().flatten()
    return preds

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
    folder    = MODEL_FOLDER_MAP[size]
    feat_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
    freq_path = FREQ_FILE_MAP[size]

    if not os.path.exists(feat_file):
        print(f"  ⚠ Feature file missing for {size}")
        return None
    if freq_path not in freq_cache:
        print(f"  ⚠ Frequency file missing for {size}")
        return None

    header  = pd.read_csv(feat_file, nrows=0).columns.tolist()
    valid   = [f for f in FEATURES_LIST if f in header]
    usecols = ['token_id'] + valid
    df_feat = pd.read_csv(feat_file, usecols=usecols, low_memory=False)

    for df_tmp in [df_feat]:
        df_tmp['token_id'] = pd.to_numeric(df_tmp['token_id'], errors='coerce').astype('Int64')
        df_tmp.dropna(subset=['token_id'], inplace=True)
        df_tmp['token_id'] = df_tmp['token_id'].astype(np.int64)

    df_freq = freq_cache[freq_path].copy()
    df_freq['token_id'] = pd.to_numeric(df_freq['token_id'], errors='coerce').astype('Int64')
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

    return {'X': X_np, 'y': y_np, 'valid_features': valid}

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, resource

    def ram():
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
    needed_paths = set(FREQ_FILE_MAP[m] for m in [TRAIN_MODEL] + TEST_MODELS)
    freq_cache   = {}
    for path in needed_paths:
        if not os.path.exists(path):
            continue
        freq_header = pd.read_csv(path, nrows=0).columns.tolist()
        freq_cols   = ['token_id']
        if 'log_count' in freq_header:
            freq_cols.append('log_count')
        elif 'count' in freq_header:
            freq_cols.append('count')
        freq_cache[path] = pd.read_csv(path, usecols=freq_cols)
        print(f"  ✅ {os.path.basename(path)}")

    # ── 2. Load models ────────────────────────────────────────
    print(f"\nLoading training model ({TRAIN_MODEL})...")
    train_data = load_model_data(TRAIN_MODEL, freq_cache)

    print("\nLoading generalization models...")
    test_data_dict = {}
    for size in TEST_MODELS:
        res = load_model_data(size, freq_cache)
        if res:
            test_data_dict[size] = res

    del freq_cache
    gc.collect()

    if not train_data:
        exit(1)

    # ── 3. Isolate the 2 requested features ───────────────────
    target_features = ['embedding_norm', 'weight_variance']
    common_feats = set(train_data['valid_features'])
    for size, d in test_data_dict.items():
        common_feats &= set(d['valid_features'])
    
    # Ensure our 2 features exist in the data
    valid_targets = [f for f in target_features if f in common_feats]
    if len(valid_targets) != 2:
        print(f"Error: Required features not found across all models. Found: {valid_targets}")
        exit(1)
        
    print(f"\nUsing features: {valid_targets}")

    # ── 4. Rank Normalize ─────────────────────────────────────
    print(f"\nNormalizing datasets...")
    idx_train = [train_data['valid_features'].index(f) for f in valid_targets]
    X_train_norm = rank_normalize(train_data['X'][:, idx_train])
    y_train_norm = rank_normalize(train_data['y'])
    del train_data
    gc.collect()

    X_train_split, X_test_7m_split, y_train, y_test_7m = train_test_split(
        X_train_norm, y_train_norm, test_size=0.2, random_state=42
    )
    del X_train_norm, y_train_norm
    gc.collect()

    gen_sets = {}
    for size, d in test_data_dict.items():
        idx_ext = [d['valid_features'].index(f) for f in valid_targets]
        gen_sets[size] = {
            'X': rank_normalize(d['X'][:, idx_ext]),
            'y': rank_normalize(d['y'])
        }
    del test_data_dict
    gc.collect()

    # ── 5. Standard Scaling (Crucial for PyTorch) ─────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_test_7m_scaled = scaler.transform(X_test_7m_split)
    
    for size in gen_sets:
        gen_sets[size]['X'] = scaler.transform(gen_sets[size]['X'])

    # ── 6. Training Scikit-Learn Models ───────────────────────
    print(f"\n{'='*55}\n--- Training Scikit-Learn Models ---\n{'='*55}")
    checkpoint("Before Sklearn")
    
    sk_lin = LinearRegression()
    sk_lin.fit(X_train_scaled, y_train)

    sk_mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation='relu',
        alpha=0.01, max_iter=2000, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=25,
        learning_rate='adaptive', random_state=42
    )
    sk_mlp.fit(X_train_scaled, y_train)

    sk_rf = RandomForestRegressor(
        n_estimators=100, max_features='sqrt',
        min_samples_leaf=5, n_jobs=4, random_state=42
    )
    sk_rf.fit(X_train_scaled, y_train)

    # ── 7. Training PyTorch Models ────────────────────────────
    print(f"\n{'='*55}\n--- Training PyTorch Models ---\n{'='*55}")
    checkpoint("Before PyTorch")
    
    pt_lin_model = TorchLinear(input_dim=2)
    pt_lin_model = train_torch_model(pt_lin_model, X_train_scaled, y_train, epochs=150, lr=0.01)
    
    pt_mlp_model = TorchMLP(input_dim=2)
    pt_mlp_model = train_torch_model(pt_mlp_model, X_train_scaled, y_train, epochs=150, lr=0.01, weight_decay=0.01)
    checkpoint("After PyTorch")

    # ── 8. Evaluation ─────────────────────────────────────────
    eval_set_names = ['7M (Test Set)'] + [f"{size} (Gen)" for size in TEST_MODELS]
    eval_sets_data = [(X_test_7m_scaled, y_test_7m)]
    for size in TEST_MODELS:
        eval_sets_data.append((gen_sets[size]['X'], gen_sets[size]['y']))

    results = {
        'SK_Linear': [], 'PT_Linear': [],
        'SK_MLP':    [], 'PT_MLP':    [],
        'SK_RF':     []
    }

    print(f"\n{'Dataset':<15} | {'SK Lin':<8} | {'PT Lin':<8} | {'SK MLP':<8} | {'PT MLP':<8} | {'SK RF':<8}")
    print("-" * 75)

    for name, (X_eval, y_eval) in zip(eval_set_names, eval_sets_data):
        r2_sk_lin = r2_score(y_eval, sk_lin.predict(X_eval))
        r2_sk_mlp = r2_score(y_eval, sk_mlp.predict(X_eval))
        r2_sk_rf  = r2_score(y_eval, sk_rf.predict(X_eval))
        
        r2_pt_lin = r2_score(y_eval, predict_torch(pt_lin_model, X_eval))
        r2_pt_mlp = r2_score(y_eval, predict_torch(pt_mlp_model, X_eval))
        
        results['SK_Linear'].append(r2_sk_lin)
        results['PT_Linear'].append(r2_pt_lin)
        results['SK_MLP'].append(r2_sk_mlp)
        results['PT_MLP'].append(r2_pt_mlp)
        results['SK_RF'].append(r2_sk_rf)
        
        print(f"{name:<15} | {r2_sk_lin:<8.4f} | {r2_pt_lin:<8.4f} | {r2_sk_mlp:<8.4f} | {r2_pt_mlp:<8.4f} | {r2_sk_rf:<8.4f}")

    # ── 9. Plotting the Main Comparison Chart ────────────────
    print("\nGenerating comparative chart...")
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(eval_set_names))
    width = 0.15 

    # Plot SK (Solid) vs PT (Hatched) pairs side by side
    bars1 = ax.bar(x - width*2, results['SK_Linear'], width, color=C_SK_LIN, edgecolor='white', label='Sklearn Linear')
    bars2 = ax.bar(x - width,   results['PT_Linear'], width, color=C_PT_LIN, edgecolor='white', hatch='////', label='PyTorch Linear')
    bars3 = ax.bar(x,           results['SK_MLP'],    width, color=C_SK_MLP, edgecolor='white', label='Sklearn MLP')
    bars4 = ax.bar(x + width,   results['PT_MLP'],    width, color=C_PT_MLP, edgecolor='white', hatch='////', label='PyTorch MLP')
    bars5 = ax.bar(x + width*2, results['SK_RF'],     width, color=C_SK_RF,  edgecolor='white', label='Sklearn RF (Baseline)')

    def add_labels(bars):
        for bar in bars:
            v = bar.get_height()
            va = 'bottom' if v >= 0 else 'top'
            oy = 0.02 if v >= 0 else -0.04
            ax.text(bar.get_x() + bar.get_width() / 2, v + oy,
                    f'{v:.2f}', ha='center', va=va, fontsize=8, fontweight='bold',
                    color='black', rotation='vertical')

    for b in [bars1, bars2, bars3, bars4, bars5]:
        add_labels(b)

    ax.axhline(0, color='black', lw=0.8)
    all_vals = [v for md in results.values() for v in md]
    ax.set_ylim(min(0, min(all_vals) - 0.20), 1.15)
    ax.set_ylabel("R² Score", fontsize=11)
    
    ax.set_title(f"Cross-Model Generalization: Scikit-Learn vs. PyTorch\n(Trained on {TRAIN_MODEL}, Using 2 Features: Emb Norm & Wt Variance)", fontweight='bold', fontsize=13)

    ax.legend(loc='upper right', ncol=3, framealpha=0.9, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(eval_set_names)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_file1 = "generalization_pytorch_comparison.png"
    plt.savefig(out_file1, dpi=300)
    print(f"Saved '{out_file1}'")
    plt.close(fig)

    # ── 10. Plotting the Delta (Difference) Chart ────────────
    print("Generating delta chart...")
    fig_delta, ax_delta = plt.subplots(figsize=(10, 5))
    width_delta = 0.3

    # Delta: PyTorch - Sklearn (Positive means PyTorch is better)
    delta_lin = np.array(results['PT_Linear']) - np.array(results['SK_Linear'])
    delta_mlp = np.array(results['PT_MLP']) - np.array(results['SK_MLP'])

    bars_d_lin = ax_delta.bar(x - width_delta/2, delta_lin, width_delta, color='#2c3e50', edgecolor='white', label='Delta Linear (PT - SK)')
    bars_d_mlp = ax_delta.bar(x + width_delta/2, delta_mlp, width_delta, color='#8e44ad', edgecolor='white', label='Delta MLP (PT - SK)')
        
    def add_delta_labels(bars):
        for bar in bars:
            v = bar.get_height()
            va = 'bottom' if v >= 0 else 'top'
            oy = 0.005 if v >= 0 else -0.005
            ax_delta.text(bar.get_x() + bar.get_width() / 2, v + oy,
                          f'{v:+.3f}', ha='center', va=va, fontsize=9, fontweight='bold',
                          color='black', rotation='vertical')

    add_delta_labels(bars_d_lin)
    add_delta_labels(bars_d_mlp)

    ax_delta.axhline(0, color='black', lw=1.2)
    
    all_deltas = np.concatenate([delta_lin, delta_mlp])
    y_min, y_max = min(all_deltas), max(all_deltas)
    pad = max(abs(y_min), abs(y_max)) * 0.4 + 0.02
    ax_delta.set_ylim(min(0, y_min - pad), max(0, y_max + pad))

    ax_delta.set_ylabel("Δ R² Score", fontsize=11)
    ax_delta.set_title(f"Generalization Delta: PyTorch vs. Scikit-Learn\n(Positive = PyTorch performed better)", fontweight='bold', fontsize=12)
    
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(eval_set_names)
    ax_delta.legend(loc='best', framealpha=0.9, fontsize=10)
    ax_delta.grid(axis='y', alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_file2 = "generalization_pytorch_delta.png"
    plt.savefig(out_file2, dpi=300)
    print(f"Saved '{out_file2}'")
    plt.close(fig_delta)

    print("\nDone.")