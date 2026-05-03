import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP & PATHS
# ==========================================
BASE_PATH = os.getcwd() 
if not BASE_PATH.endswith("gradient-pretiction-via-embedding_norm"):
    WORK_DIR = os.path.join(BASE_PATH, "gradient-pretiction-via-embedding_norm")
else:
    WORK_DIR = BASE_PATH

INPUT_CSV = os.path.join(WORK_DIR, "embedding_norms_empirical.csv")
PLOTS_DIR = os.path.join(WORK_DIR, "plots_empirical_ml")

if os.path.exists(PLOTS_DIR):
    import shutil
    shutil.rmtree(PLOTS_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 2. THE MASTER PLOTTING FUNCTION
# ==========================================
def plot_master(x_data, y_data, title, x_label, y_label, filename, 
                scatter_label=None, line_x=None, line_y=None, 
                line_label=None, metrics_text=None, multi_series=None):
    
    plt.figure(figsize=(10, 7))
    
    if multi_series:
        colors = ['#1f77b4', '#d62728', '#7f7f7f']
        for i, (x, y, lbl) in enumerate(multi_series):
            plt.scatter(x, y, alpha=0.5, color=colors[i], label=lbl, edgecolors='none', s=20)
    else:
        plt.scatter(x_data, y_data, alpha=0.6, color='#1f77b4', edgecolors='k', label=scatter_label, s=30)
        if line_x is not None and line_y is not None:
            plt.plot(line_x, line_y, color='#d62728', linestyle='--', linewidth=2.5, label=line_label)

    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    if metrics_text:
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
            
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()

# ==========================================
# 3. ANALYSIS ENGINE
# ==========================================
def run_full_analysis():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    models = ['7M', '30M', '124M']

    for model in models:
        print(f"\n" + "="*50)
        print(f"📊 ARCHITECTURE ANALYSIS: {model}")
        print("="*50)
        
        m_df = df[df['Model'] == model]
        g_df = m_df[m_df['Method'] == 'Global'].sort_values('Feature_Value')
        
        if g_df.empty:
            print(f"⚠️ No global data found for {model}. Skipping.")
            continue
        
        X_g = g_df[['Feature_Value']].values
        y_g = g_df['Step'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # --- MODEL 1: Standard Linear Regression (Legacy Reference) ---
        lin_reg = LinearRegression()
        cv_r2_lin = cross_val_score(lin_reg, X_g, y_g, cv=kf, scoring='r2').mean()
        lin_reg.fit(X_g, y_g)
        mae_lin = mean_absolute_error(y_g, lin_reg.predict(X_g))
        
        # --- MODEL 2: Polynomial Regression (Degree 2) ---
        poly_pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('reg', LinearRegression())
        ])
        cv_r2_poly = cross_val_score(poly_pipe, X_g, y_g, cv=kf, scoring='r2').mean()
        poly_pipe.fit(X_g, y_g)
        mae_poly = mean_absolute_error(y_g, poly_pipe.predict(X_g))

        # --- MODEL 3: Logarithmic Transformation ---
        log_pipe = Pipeline([
            ('log', FunctionTransformer(np.log, validate=True)),
            ('reg', LinearRegression())
        ])
        cv_r2_log = cross_val_score(log_pipe, X_g, y_g, cv=kf, scoring='r2').mean()
        log_pipe.fit(X_g, y_g)
        mae_log = mean_absolute_error(y_g, log_pipe.predict(X_g))

        # --- PRINT RESULTS TO TERMINAL ---
        print(f"{'Method':<20} | {'CV R2':<10} | {'MAE':<10}")
        print("-" * 45)
        print(f"{'Pure Linear':<20} | {cv_r2_lin:<10.4f} | {mae_lin:<10.2f}")
        print(f"{'Polynomial (D2)':<20} | {cv_r2_poly:<10.4f} | {mae_poly:<10.2f}")
        print(f"{'Logarithmic':<20} | {cv_r2_log:<10.4f} | {mae_log:<10.2f}")

        # --- INDIVIDUAL PLOT GENERATION ---
        X_smooth = np.linspace(X_g.min(), X_g.max(), 200).reshape(-1, 1)
        
        # Configuration for the 3 individual plots
        plot_configs = [
            {
                "filename_suffix": "1_Linear_Fit",
                "title": "Linear Baseline",
                "pred": lin_reg.predict(X_smooth),
                "color": "gray",
                "line_style": "--",
                "r2": cv_r2_lin,
                "mae": mae_lin
            },
            {
                "filename_suffix": "2_Polynomial_Fit",
                "title": "Polynomial Fit (Degree 2)",
                "pred": poly_pipe.predict(X_smooth),
                "color": "blue",
                "line_style": "-",
                "r2": cv_r2_poly,
                "mae": mae_poly
            },
            {
                "filename_suffix": "3_Logarithmic_Fit",
                "title": "Logarithmic Fit",
                "pred": log_pipe.predict(X_smooth),
                "color": "red",
                "line_style": "-",
                "r2": cv_r2_log,
                "mae": mae_log
            }
        ]

        # Generate a separate plot for each configuration
        for config in plot_configs:
            plt.figure(figsize=(10, 7))
            
            # Actual points in light yellow with a black edge for visibility
            plt.scatter(X_g, y_g, alpha=0.9, color='#FFFF99', edgecolors='k', 
                        label="Actual Checkpoints", s=60, zorder=3)
            
            # The prediction line
            plt.plot(X_smooth, config["pred"], color=config["color"], 
                     linestyle=config["line_style"], linewidth=2.5, 
                     label=config["title"], zorder=2)
            
            plt.title(f"{model} - {config['title']}", fontsize=15, fontweight='bold', pad=20)
            plt.xlabel("Frobenius Norm", fontsize=12)
            plt.ylabel("Training Step", fontsize=12)
            
            metrics_text = f"Architecture: {model}\nMethod: {config['title']}\nCV R²: {config['r2']:.4f}\nMAE: {config['mae']:.1f} steps"
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=11,
                    verticalalignment='top', bbox=props, family='monospace')
                
            plt.legend(loc='lower right', frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{model}_{config['filename_suffix']}.png"), dpi=300)
            plt.close()

        # =====================================================================
        # ARCHIVED EXPERIMENTS (BINS) - DO NOT RUN
        # =====================================================================
        '''
        # (The original bin code remains here as a comment for future reference)
        # Fixes included: One-Hot Encoding for Bin_ID, GroupShuffleSplit for steps.
        '''

    # --- FINAL PLOT: COMBINED GLOBAL BASELINE ---
    print("\n🎨 Generating Combined Architecture Comparison...")
    plt.figure(figsize=(10, 7))
    clrs = {'7M': 'blue', '30M': 'green', '124M': 'red'}
    for m in models:
        m_g = df[(df['Model'] == m) & (df['Method'] == 'Global')].sort_values('Step')
        if not m_g.empty:
            v = m_g['Feature_Value'].values
            norm_v = (v - v.min()) / (v.max() - v.min())
            plt.plot(m_g['Step'], norm_v, label=f"Model {m}", color=clrs[m], marker='o', alpha=0.7)
    
    plt.title("Combined Global Trends: Normalized Norm vs Steps", fontsize=15, fontweight='bold')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Normalized Embedding Norm (0 to 1)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(PLOTS_DIR, "ALL_MODELS_Combined_Baseline.png"), dpi=300)
    plt.close()

    print(f"\n✅ All analysis complete. Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    run_full_analysis()