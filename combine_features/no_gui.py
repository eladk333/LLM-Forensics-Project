import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "no_gui")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_CSV = os.path.join(CURRENT_DIR, "forensic_features.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

if not os.path.exists(PROB_CSV):
    PROB_CSV = os.path.join(CURRENT_DIR, "probability_results_server.csv")

EMBED_DIM_MAP = {'Model 7M': 128, 'Model 30M': 384, 'Model 124M': 768}
MODEL_COLORS = {'Model 7M': 'red', 'Model 30M': 'green', 'Model 124M': 'blue'}

# ==========================================
# 2. MODELS & UTILS
# ==========================================
def inverse_exp_func(prob, a, b):
    return a * np.exp(b * prob)

def get_mlp_model():
    """Consistent MLP architecture for non-linear forensic mapping"""
    return make_pipeline(
        StandardScaler(), 
        MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=5000, 
                     learning_rate='adaptive', random_state=42)
    )

def add_stats_box(ax, r2, mae, extra_text=""):
    stats_text = f"Accuracy Metrics:\n----------------\nR² Score: {r2:.4f}\nMAE: ±{mae:.1f} steps"
    if extra_text:
        stats_text = f"{extra_text}\n\n{stats_text}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ==========================================
# 3. DATA HELPERS
# ==========================================
def load_and_merge_data():
    if not os.path.exists(FEATURE_CSV) or not os.path.exists(PROB_CSV):
        return None
    df_feats = pd.read_csv(FEATURE_CSV)
    df_probs = pd.read_csv(PROB_CSV)
    df_probs['Model'] = df_probs['Model'].apply(lambda x: x if 'Model' in x else f"Model {x}")
    df_feats['Model'] = df_feats['Model'].astype(str)
    return pd.merge(df_feats, df_probs, on=['Model', 'Step'], how='inner')

def scale_features(df_subset):
    df_scaled = df_subset.copy()
    for m_name, n_embd in EMBED_DIM_MAP.items():
        mask = df_scaled['Model'] == m_name
        if mask.any():
            df_scaled.loc[mask, 'embedding_norm'] = df_scaled.loc[mask, 'embedding_norm'] / np.sqrt(n_embd)
    return df_scaled

# ==========================================
# 4. PLOT GENERATION
# ==========================================
def generate_all_plots():
    df = load_and_merge_data()
    if df is None:
        print("❌ Error: CSV files missing!")
        return

    # --- 1-3: Internal Joint Prediction (CV - Linear) ---
    for model_name in ['Model 7M', 'Model 30M', 'Model 124M']:
        subset = df[df['Model'] == model_name].sort_values('Step')
        if subset.empty: continue
        X, y = subset[['embedding_norm', 'Avg_Probability']].values, subset['Step'].values
        model = make_pipeline(StandardScaler(), LinearRegression())
        y_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y, y_pred, color=MODEL_COLORS[model_name], alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
        ax.set_title(f"Internal Forensic Accuracy: {model_name}")
        ax.set_xlabel("Actual Steps")
        ax.set_ylabel("Predicted Steps")
        add_stats_box(ax, r2_score(y, y_pred), mean_absolute_error(y, y_pred), "Internal CV (Linear Hybrid)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"cv_internal_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    # --- Zero-Shot Setup ---
    df_scaled = scale_features(df)
    train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
    test = df_scaled[df_scaled['Model'] == 'Model 124M']
    y_test_actual = test['Step'].values

    # --- 4-9: Zero-Shot Variants (Standard vs MLP) ---
    # Using a loop here makes the code shorter but more powerful
    scenarios = [
        (['embedding_norm'], "Linear", "zs_only_emb_norm.png"),
        (['Avg_Probability'], "Exponential", "zs_only_prob.png"),
        (['embedding_norm', 'Avg_Probability'], "Linear", "zs_combined_features.png")
    ]

    for cols, mode, filename in scenarios:
        # A) Standard/Best Fit (Linear or Exponential)
        if mode == "Exponential":
            popt, _ = curve_fit(inverse_exp_func, train[cols[0]].values, train['Step'].values, maxfev=10000)
            y_pred = inverse_exp_func(test[cols[0]].values, *popt)
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[cols].values, train['Step'].values)
            y_pred = model.predict(test[cols].values)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test_actual, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title(f"Zero-Shot: {', '.join(cols)} ({mode})")
        add_stats_box(ax, r2_score(y_test_actual, y_pred), mean_absolute_error(y_test_actual, y_pred), f"Model: {mode}")
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()

        # B) MLP Fit
        mlp = get_mlp_model()
        mlp.fit(train[cols].values, train['Step'].values)
        y_pred_mlp = mlp.predict(test[cols].values)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test_actual, y_pred_mlp, color='darkgreen', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title(f"Zero-Shot: {', '.join(cols)} (MLP Regressor)")
        add_stats_box(ax, r2_score(y_test_actual, y_pred_mlp), mean_absolute_error(y_test_actual, y_pred_mlp), "Model: MLP (Neural Network)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"zs_mlp_{filename}"), dpi=300)
        plt.close()

if __name__ == "__main__":
    print(f"🚀 Running Analysis: 9 Plots Total (CV + Zero-Shot + MLP)...")
    generate_all_plots()
    print(f"✅ Success. Results available in 'no_gui' folder.")