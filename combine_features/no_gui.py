import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
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
# 2. MODELS
# ==========================================
def inverse_exp_func(prob, a, b):
    return a * np.exp(b * prob)

# ==========================================
# 3. HELPERS
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

def add_stats_box(ax, r2, mae, extra_text=""):
    """Adds a prominent box with R^2 and MAE metrics"""
    stats_text = f"Accuracy Metrics:\n----------------\nR² Score: {r2:.4f}\nMAE: ±{mae:.1f} steps"
    if extra_text:
        stats_text = f"{extra_text}\n\n{stats_text}"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ==========================================
# 4. PLOT GENERATION
# ==========================================
def generate_all_plots():
    df = load_and_merge_data()
    if df is None:
        print("❌ Error: CSV files missing!")
        return

    # --- 1-3: Internal Joint Prediction (CV) ---
    for model_name in ['Model 7M', 'Model 30M', 'Model 124M']:
        subset = df[df['Model'] == model_name].sort_values('Step')
        if subset.empty: continue
        
        X = subset[['embedding_norm', 'Avg_Probability']].values
        y = subset['Step'].values
        
        model = make_pipeline(StandardScaler(), LinearRegression())
        y_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
        
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y, y_pred, color=MODEL_COLORS[model_name], alpha=0.6, label='Forensic Predictions')
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, label='Ideal Identity Line')
        
        ax.set_title(f"Internal Forensic Accuracy: {model_name}")
        ax.set_xlabel("Actual Training Steps (Ground Truth)")
        ax.set_ylabel("Predicted Steps")
        
        add_stats_box(ax, r2, mae, f"Internal consistency check for {model_name}.\nUses Embedding Norm + Prob.")
        
        ax.legend(loc='lower right')
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(OUTPUT_DIR, f"cv_internal_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    # --- 4-6: Zero-Shot Generalization ---
    df_scaled = scale_features(df)
    train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
    test = df_scaled[df_scaled['Model'] == 'Model 124M']
    
    if not train.empty and not test.empty:
        y_test_actual = test['Step'].values

        # A) Embedding Norm Only
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(train[['embedding_norm']].values, train['Step'].values)
        y_pred_norm = model.predict(test[['embedding_norm']].values)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        r2_n, mae_n = r2_score(y_test_actual, y_pred_norm), mean_absolute_error(y_test_actual, y_pred_norm)
        ax.scatter(y_test_actual, y_pred_norm, color='purple', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title("Zero-Shot: Embedding Norm Only (Linear)")
        ax.set_xlabel("Actual Steps (124M)")
        ax.set_ylabel("Predicted Steps")
        add_stats_box(ax, r2_n, mae_n, "Knowledge transfer via Scaling Laws.\nFeature: Embedding Norm.")
        plt.savefig(os.path.join(OUTPUT_DIR, "zs_only_emb_norm.png"), dpi=300)
        plt.close()

        # B) Probability Only (Exponential)
        popt, _ = curve_fit(inverse_exp_func, train['Avg_Probability'].values, train['Step'].values, maxfev=10000)
        y_pred_prob = inverse_exp_func(test['Avg_Probability'].values, *popt)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        r2_p, mae_p = r2_score(y_test_actual, y_pred_prob), mean_absolute_error(y_test_actual, y_pred_prob)
        ax.scatter(y_test_actual, y_pred_prob, color='orange', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title("Zero-Shot: Next-Token Probability Only (Exponential)")
        ax.set_xlabel("Actual Steps (124M)")
        ax.set_ylabel("Predicted Steps")
        add_stats_box(ax, r2_p, mae_p, "Generalization via training dynamics.\nFeature: Next-Token Probability.")
        plt.savefig(os.path.join(OUTPUT_DIR, "zs_only_prob.png"), dpi=300)
        plt.close()

        # C) Combined Features (Linear Joint)
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(train[['embedding_norm', 'Avg_Probability']].values, train['Step'].values)
        y_pred_comb = model.predict(test[['embedding_norm', 'Avg_Probability']].values)
        
        fig, ax = plt.subplots(figsize=(9, 6))
        r2_c, mae_c = r2_score(y_test_actual, y_pred_comb), mean_absolute_error(y_test_actual, y_pred_comb)
        ax.scatter(y_test_actual, y_pred_comb, color='black', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title("Zero-Shot: Joint Prediction (Combined)")
        ax.set_xlabel("Actual Steps (124M)")
        ax.set_ylabel("Predicted Steps")
        add_stats_box(ax, r2_c, mae_c, "Predicting 124M age using multi-modal\nsignals from smaller models.")
        plt.savefig(os.path.join(OUTPUT_DIR, "zs_combined_features.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    print(f"🚀 Generating forensic plots with R² and MAE stats...")
    generate_all_plots()
    print(f"✅ Success. Results available in 'no_gui' folder.")