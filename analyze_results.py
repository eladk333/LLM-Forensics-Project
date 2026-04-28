import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

# ---------------------------------------------------------
# 1. Paths & Mappings
# ---------------------------------------------------------
BASE_DIR = r'C:\Users\elad.k.int\LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

MODEL_SIZES = ['7M', '30M', '124M', '500M']

MODEL_FOLDER_MAP = {
    '7M':   'MinGPT_Checkpoints_7M',
    '30M':  'MinGPT_Checkpoints_30M',
    '124M': 'MinGPT_Checkpoints_124M',
    '500M': 'MinGPT_Checkpoints_500M',
}

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

# ---------------------------------------------------------
# 2. Extract Feature Contributions
# ---------------------------------------------------------
def get_feature_importances():
    model_importances = {}
    valid_features_used = None

    # Cache frequencies
    df_freq_cache = {}
    for freq_path in set(FREQ_FILE_MAP.values()):
        if os.path.exists(freq_path):
            df_freq_cache[freq_path] = pd.read_csv(freq_path)

    for size in MODEL_SIZES:
        folder = MODEL_FOLDER_MAP[size]
        feature_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')
        freq_path = FREQ_FILE_MAP[size]

        if not os.path.exists(feature_file) or freq_path not in df_freq_cache:
            print(f"Skipping {size} - Missing data files.")
            continue

        # Load and merge data
        df_features = pd.read_csv(feature_file)
        valid_features = [f for f in FEATURES_LIST if f in df_features.columns]
        
        if valid_features_used is None:
            valid_features_used = valid_features

        df_freq = df_freq_cache[freq_path]
        df = pd.merge(df_freq, df_features, on='token_id', how='inner')
        
        y = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])
        X = df[valid_features_used]

        # Standardize 
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Linear Regression
        lr = LinearRegression()
        lr.fit(X_scaled, y)

        importances = np.abs(lr.coef_)
        model_importances[size] = importances

    return model_importances, valid_features_used

# ---------------------------------------------------------
# 3. Separate Radar Plots Logic
# ---------------------------------------------------------
def plot_radar_charts_separate(model_importances, features):
    if not model_importances:
        print("No models to plot!")
        return

    labels = [FEATURE_CONFIG.get(f, f) for f in features]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # Find the maximum value across all models to normalize the visual scales
    max_val = max([np.max(imps) for imps in model_importances.values()])
    plot_limit = max_val * 1.15 # Add 15% padding to the outer edge

    # Create a 2x2 grid of subplots (Polar)
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(polar=True))
    axs = axs.flatten() # Flatten the 2x2 array for easy iteration
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (model_size, importances) in enumerate(model_importances.items()):
        ax = axs[idx]
        values = importances.tolist()
        values += values[:1] # Close the loop
        
        color = colors[idx % len(colors)]
        
        # Configure axis rotation/direction
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set standardized limits for fair visual comparison
        ax.set_ylim(0, plot_limit)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, weight='bold')

        # Draw plot
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.25)
        
        # Subplot Title
        ax.set_title(f"Model Size: {model_size}", size=14, weight='bold', color=color, pad=20)

    # If any models failed to load, hide their empty subplots
    for i in range(len(model_importances), len(axs)):
        fig.delaxes(axs[i])

    plt.suptitle("Feature Contribution by Model Size\n(Absolute Standardized Linear Regression Coefficients)", 
                 size=18, weight='bold', y=0.98)
    
    plt.tight_layout()
    # Adjust top padding so the suptitle doesn't overlap subplot titles
    plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.3) 
    plt.show()

if __name__ == "__main__":
    print("Extracting features and training linear models...")
    importances, feats = get_feature_importances()
    
    if feats is None:
        print("Could not load necessary feature data.")
    else:
        print("Generating Separate Radar Charts...")
        plot_radar_charts_separate(importances, feats)