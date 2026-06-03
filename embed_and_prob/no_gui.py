import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
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

    # List to store results for final ranking
    final_rankings = []

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
        ax.scatter(y, y_pred, color=MODEL_COLORS[model_name], alpha=0.6, label="CV Predictions (Folds=5)")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5, label="Perfect Fit (Y=X)")
        ax.set_title(f"Internal Forensic Accuracy: {model_name}\n(Using Embedding Norm & Next Token Prob)")
        ax.set_xlabel("Actual Training Steps")
        ax.set_ylabel("Predicted Training Steps")
        add_stats_box(ax, r2, mae, f"Internal CV (Linear Hybrid)\nData: {model_name} Only")
        ax.legend(loc='lower right') 
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

    # ADDED: Single-feature polynomial baseline for embedding_norm
    scenarios = [
        (['embedding_norm'], "Linear", "zs_only_emb_norm_linear.png"),
        (['embedding_norm'], "Polynomial", "zs_only_emb_norm_poly.png"),
        (['Avg_Probability'], "Exponential", "zs_only_prob.png")
    ]

    for cols, mode, filename in scenarios:
        if mode == "Exponential":
            popt, _ = curve_fit(inverse_exp_func, train[cols[0]].values, train['Step'].values, maxfev=10000)
            y_pred = inverse_exp_func(test[cols[0]].values, *popt)
            
            # Physical Curve Plot (Feature vs Step)
            test_sorted = test.sort_values(cols[0])
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(test_sorted[cols[0]], test_sorted['Step'], color='#FFFF99', edgecolors='k', s=60, label="Actual Unseen 124M Data")
            ax.plot(test_sorted[cols[0]], inverse_exp_func(test_sorted[cols[0]], *popt), 'r--', linewidth=2.5, label="Exponential Curve (Fitted on 7M+30M)")
            ax.set_title(f"Zero-Shot Physical Projection:\n{cols[0]} mapped to Training Steps")
            ax.set_xlabel(f"Scaled {cols[0]}")
            ax.set_ylabel("Training Steps")
            
            r2, mae = r2_score(y_test_actual, y_pred), mean_absolute_error(y_test_actual, y_pred)
            add_stats_box(ax, r2, mae, "Method: Exponential Curve Fit")
            ax.legend(loc='lower right')
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
            plt.close()
            
            scenario_name = f"{cols[0]} ({mode})"
            print(f"{scenario_name:<40} | {r2:<10.4f} | {mae:<12.2f}")
            final_rankings.append((scenario_name, r2, mae, filename))

        elif mode == "Linear" and len(cols) == 1:
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(train[cols].values, train['Step'].values)
            y_pred = model.predict(test[cols].values)
            
            test_sorted = test.sort_values(cols[0])
            y_pred_sorted = model.predict(test_sorted[cols].values)
            
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(test_sorted[cols[0]], test_sorted['Step'], color='lightblue', edgecolors='k', s=60, label="Actual Unseen 124M Data")
            ax.plot(test_sorted[cols[0]], y_pred_sorted, 'r--', linewidth=2.5, label="Linear Trend (Fitted on 7M+30M)")
            ax.set_title(f"Zero-Shot Feature Projection:\n{cols[0]} mapped to Training Steps (Linear)")
            ax.set_xlabel(f"Scaled {cols[0]}")
            ax.set_ylabel("Training Steps")
            
            r2, mae = r2_score(y_test_actual, y_pred), mean_absolute_error(y_test_actual, y_pred)
            add_stats_box(ax, r2, mae, "Method: Simple Linear Regression")
            ax.legend(loc='lower right')
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
            plt.close()
            
            scenario_name = f"{cols[0]} ({mode})"
            print(f"{scenario_name:<40} | {r2:<10.4f} | {mae:<12.2f}")
            final_rankings.append((scenario_name, r2, mae, filename))

        elif mode == "Polynomial" and len(cols) == 1:
            # ADDED: Single feature polynomial mapping (Degree 2)
            model = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
            model.fit(train[cols].values, train['Step'].values)
            y_pred = model.predict(test[cols].values)
            
            test_sorted = test.sort_values(cols[0])
            y_pred_sorted = model.predict(test_sorted[cols].values)
            
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(test_sorted[cols[0]], test_sorted['Step'], color='plum', edgecolors='k', s=60, label="Actual Unseen 124M Data")
            ax.plot(test_sorted[cols[0]], y_pred_sorted, 'g--', linewidth=2.5, label="Polynomial Trend (Deg 2) (Fitted on 7M+30M)")
            ax.set_title(f"Zero-Shot Feature Projection:\n{cols[0]} mapped to Training Steps (Polynomial)")
            ax.set_xlabel(f"Scaled {cols[0]}")
            ax.set_ylabel("Training Steps")
            
            r2, mae = r2_score(y_test_actual, y_pred), mean_absolute_error(y_test_actual, y_pred)
            add_stats_box(ax, r2, mae, "Method: Polynomial Regression (Deg 2)")
            ax.legend(loc='lower right')
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
            plt.close()
            
            scenario_name = f"{cols[0]} ({mode})"
            print(f"{scenario_name:<40} | {r2:<10.4f} | {mae:<12.2f}")
            final_rankings.append((scenario_name, r2, mae, filename))

    # --- Combined Features (Actual vs Predicted plots) ---
    hybrid_cols = ['embedding_norm', 'Avg_Probability']
    X_tr_h, y_tr_h = train[hybrid_cols].values, train['Step'].values
    X_te_h, y_te_h = test[hybrid_cols].values, test['Step'].values

    combined_models = [
        ("Combined (Linear Regression)", make_pipeline(StandardScaler(), LinearRegression()), "zs_combined_linear.png", "lightblue"),
        ("Combined (Ridge L2)", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "zs_combined_ridge.png", "teal"),
        ("Combined (Polynomial D2)", make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()), "zs_combined_polynomial.png", "purple"),
        ("Combined (Random Forest)", make_pipeline(StandardScaler(), RandomForestRegressor(100, random_state=42)), "zs_combined_random_forest.png", "orange"),
        ("Combined (MLP Neural Net)", get_mlp_model(), "zs_combined_mlp.png", "darkgreen")
    ]

    for model_name, pipeline, fname, color in combined_models:
        pipeline.fit(X_tr_h, y_tr_h)
        y_pred = pipeline.predict(X_te_h)
        r2, mae = r2_score(y_te_h, y_pred), mean_absolute_error(y_te_h, y_pred)
        print(f"{model_name:<40} | {r2:<10.4f} | {mae:<12.2f}")
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_te_h, y_pred, color=color, alpha=0.7, edgecolors='k', s=50, label=f"{model_name} Predictions")
        ax.plot([y_te_h.min(), y_te_h.max()], [y_te_h.min(), y_te_h.max()], 'r--', linewidth=2, label="Perfect Fit (Y=X)")
        ax.set_title(f"Zero-Shot Generalization Performance\nTraining: 7M+30M  |  Testing: 124M")
        ax.set_xlabel("Actual 124M Training Steps")
        ax.set_ylabel("Predicted Training Steps")
        add_stats_box(ax, r2, mae, f"Algorithm: {model_name}\nFeatures: Norm & Probability")
        ax.legend(loc='lower right')
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=300)
        plt.close()
        
        final_rankings.append((model_name, r2, mae, fname))

    # --- Print Summary and Recommendations ---
    print(f"\n{'='*65}\n🏆 ZERO-SHOT MODEL RANKING (BY LOWEST MAE)\n{'='*65}")
    final_rankings.sort(key=lambda x: x[2])
    
    print(f"{'Rank':<5} | {'Model':<30} | {'MAE':<10} | {'R²':<8} | {'Filename'}")
    print("-" * 80)
    for idx, (name, r2, mae, fname) in enumerate(final_rankings, 1):
        print(f"#{idx:<4} | {name:<30} | {mae:<10.1f} | {r2:<8.4f} | {fname}")
        
    print("\n💡 Recommendation for Presentation:")
    print(f"-> Use '{final_rankings[0][3]}' as your best performing model.")
    print(f"-> Use '{final_rankings[-1][3]}' as a baseline to show improvement.")
    print(f"\n✅ All analysis complete. Results saved in '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    generate_all_plots()