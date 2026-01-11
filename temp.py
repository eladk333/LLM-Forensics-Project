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

# ==========================================
# 1. SETUP & DATA LOADING
# ==========================================
MODEL_SIZE = '7M'
BASE_FOLDER = 'G:/My Drive/llm'
FREQ_FILE = os.path.join(BASE_FOLDER, 'wiki_token_frequencies.csv')
FEATURE_FILE = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'norm_comparison_dataset.csv')

if not os.path.exists(FREQ_FILE) or not os.path.exists(FEATURE_FILE):
    print("❌ Error: Files not found. Please check paths.")
else:
    print("📂 Loading and merging data...")
    df_freq = pd.read_csv(FREQ_FILE)
    df_feat = pd.read_csv(FEATURE_FILE)
    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    y = df['log_count']
    print(f"✅ Data Ready. Rows: {len(df)}")

# ==========================================
# 2. THE PLOTTING FUNCTION
# ==========================================
def plot_actual_vs_predicted(y_true, y_pred, title, r2, ax=None):
    """
    Generates the specific 'Actual vs Predicted' plot with the red diagonal line.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Scatter Plot
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.3, color='blue', edgecolor='w', s=40)

    # 2. Perfect Fit Line (Diagonal)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Prediction')

    # 3. Styling
    ax.set_title(f"{title}\n($R^2$: {r2:.3f})", fontsize=14)
    ax.set_xlabel("Actual Log Frequency (Ground Truth)", fontsize=11)
    ax.set_ylabel("Predicted Log Frequency", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.5)

# ==========================================
# 3. ANALYSIS LOOP
# ==========================================
scenarios = [
    ('Embedding Norm', ['embedding_norm']),
    ('Logit Norm', ['logit_norm']),
    ('Combined Features', ['embedding_norm', 'logit_norm'])
]

print("\n" + "="*60)
print("🚀 GENERATING ACTUAL vs. PREDICTED PLOTS")
print("="*60)

for name, features in scenarios:
    print(f"\nProcessing: {name}...")
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train Linear Regression ---
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_preds = lin_model.predict(X_test)
    lin_r2 = r2_score(y_test, lin_preds)

    # --- Train MLP ---
    mlp_model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42)
    )
    mlp_model.fit(X_train, y_train)
    mlp_preds = mlp_model.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_preds)

    # --- PLOT SIDE-BY-SIDE ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Linear Regression
    plot_actual_vs_predicted(y_test, lin_preds, f"{name} - Linear Regression", lin_r2, ax=axes[0])

    # Plot 2: MLP
    plot_actual_vs_predicted(y_test, mlp_preds, f"{name} - Neural Network (MLP)", mlp_r2, ax=axes[1])

    plt.tight_layout()
    plt.show()