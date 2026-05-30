import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
# 4. MAIN ANALYSIS
# ==========================================
def generate_all_plots():
    df = load_and_merge_data()
    if df is None:
        print("❌ Error: CSV files missing!")
        return

    print(f"\n{'='*65}\n📊 PART 1: INTERNAL ARCHITECTURE ANALYSIS (CV)\n{'='*65}")
    print(f"{'Model':<15} | {'R² Score':<10} | {'MAE (Steps)':<12}")
    print("-" * 45)

    for model_name in ['Model 7M', 'Model 30M', 'Model 124M']:
        subset = df[df['Model'] == model_name].sort_values('Step')
        if subset.empty: continue
        X, y = subset[['embedding_norm', 'Avg_Probability']].values, subset['Step'].values
        model = make_pipeline(StandardScaler(), LinearRegression())
        y_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=42))
        
        r2, mae = r2_score(y, y_pred), mean_absolute_error(y, y_pred)
        print(f"{model_name:<15} | {r2:<10.4f} | {mae:<12.2f}")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y, y_pred, color=MODEL_COLORS[model_name], alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
        ax.set_title(f"Internal Forensic Accuracy: {model_name}")
        ax.set_xlabel("Actual Steps")
        ax.set_ylabel("Predicted Steps")
        add_stats_box(ax, r2, mae, "Internal CV (Linear Hybrid)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"cv_internal_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

    # --- Zero-Shot Setup ---
    df_scaled = scale_features(df)
    train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
    test = df_scaled[df_scaled['Model'] == 'Model 124M']
    y_test_actual = test['Step'].values

    print(f"\n{'='*65}\n🚀 PART 2: ZERO-SHOT EXTRAPOLATION (7M+30M -> 124M)\n{'='*65}")
    print(f"{'Scenario (Model Type)':<40} | {'R² Score':<10} | {'MAE (Steps)':<12}")
    print("-" * 70)

    scenarios = [
        (['embedding_norm'], "Linear", "zs_only_emb_norm.png"),
        (['Avg_Probability'], "Exponential", "zs_only_prob.png"),
        (['embedding_norm', 'Avg_Probability'], "Linear", "zs_combined_features.png")
    ]

    for cols, mode, filename in scenarios:
        # A) Standard
        if mode == "Exponential":
            popt, _ = curve_fit(inverse_exp_func, train[cols[0]].values, train['Step'].values, maxfev=10000)
            y_pred = inverse_exp_func(test[cols[0]].values, *popt)
            
            # Physical Curve Plot
            test_sorted = test.sort_values(cols[0])
            plt.figure(figsize=(9, 6))
            plt.scatter(test_sorted[cols[0]], test_sorted['Step'], color='#FFFF99', edgecolors='k', label="Actual 124M")
            plt.plot(test_sorted[cols[0]], inverse_exp_func(test_sorted[cols[0]], *popt), 'r--', label="Exp Curve")
            plt.title(f"Zero-Shot Physical Curve: {cols[0]}")
            plt.savefig(os.path.join(OUTPUT_DIR, "zs_physical_curve_exp.png"))
            plt.close()
        else:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[cols].values, train['Step'].values)
            y_pred = model.predict(test[cols].values)
        
        r2, mae = r2_score(y_test_actual, y_pred), mean_absolute_error(y_test_actual, y_pred)
        scenario_name = f"{', '.join(cols)} ({mode})"
        print(f"{scenario_name:<40} | {r2:<10.4f} | {mae:<12.2f}")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test_actual, y_pred, color='blue', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        ax.set_title(f"Zero-Shot: {scenario_name}")
        add_stats_box(ax, r2, mae, f"Model: {mode}")
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close()

        # B) MLP
        mlp = get_mlp_model()
        mlp.fit(train[cols].values, train['Step'].values)
        y_pred_mlp = mlp.predict(test[cols].values)
        r2_m, mae_m = r2_score(y_test_actual, y_pred_mlp), mean_absolute_error(y_test_actual, y_pred_mlp)
        scenario_mlp = f"{', '.join(cols)} (MLP)"
        print(f"{scenario_mlp:<40} | {r2_m:<10.4f} | {mae_m:<12.2f}")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test_actual, y_pred_mlp, color='darkgreen', alpha=0.6)
        ax.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
        add_stats_box(ax, r2_m, mae_m, "Model: MLP")
        plt.savefig(os.path.join(OUTPUT_DIR, f"zs_mlp_{filename}"), dpi=300)
        plt.close()

    # --- Advanced Hybrids ---
    hybrid_cols = ['embedding_norm', 'Avg_Probability']
    X_tr_h, y_tr_h = train[hybrid_cols].values, train['Step'].values
    X_te_h, y_te_h = test[hybrid_cols].values, test['Step'].values

    # 1. Polynomial
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()).fit(X_tr_h, y_tr_h)
    y_p = poly.predict(X_te_h)
    r2_p, mae_p = r2_score(y_te_h, y_p), mean_absolute_error(y_te_h, y_p)
    print(f"{'Combined (Polynomial D2)':<40} | {r2_p:<10.4f} | {mae_p:<12.2f}")

    # 2. Random Forest
    rf = make_pipeline(StandardScaler(), RandomForestRegressor(100, random_state=42)).fit(X_tr_h, y_tr_h)
    y_rf = rf.predict(X_te_h)
    r2_rf, mae_rf = r2_score(y_te_h, y_rf), mean_absolute_error(y_te_h, y_rf)
    print(f"{'Combined (Random Forest)':<40} | {r2_rf:<10.4f} | {mae_rf:<12.2f}")

    print(f"\n{'='*65}\n✅ Analysis complete. Results in 'no_gui' folder.\n{'='*65}")

if __name__ == "__main__":
    generate_all_plots()