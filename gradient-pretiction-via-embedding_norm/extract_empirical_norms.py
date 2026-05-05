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

# Architecture details for Theoretical Scaling
EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}

# ==========================================
# 2. ANALYSIS ENGINE
# ==========================================
def run_full_analysis():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    models = ['7M', '30M', '124M']

    # ---------------------------------------------------------
    # PART A: INTERNAL ARCHITECTURE ANALYSIS (CV within Model)
    # ---------------------------------------------------------
    for model in models:
        print(f"\n" + "="*60)
        print(f"📊 PART A: INTERNAL ANALYSIS - {model}")
        print("="*60)
        
        m_df = df[df['Model'] == model]
        g_df = m_df[m_df['Method'] == 'Global'].sort_values('Feature_Value')
        
        if g_df.empty:
            continue
        
        X_g = g_df[['Feature_Value']].values
        y_g = g_df['Step'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Pipelines
        lin_pipe = Pipeline([('reg', LinearRegression())])
        poly_pipe = Pipeline([('poly', PolynomialFeatures(degree=2)), ('reg', LinearRegression())])
        log_pipe = Pipeline([('log', FunctionTransformer(np.log, validate=True)), ('reg', LinearRegression())])

        # Fitting & Metrics
        cv_r2_lin = cross_val_score(lin_pipe, X_g, y_g, cv=kf, scoring='r2').mean()
        lin_pipe.fit(X_g, y_g)
        mae_lin = mean_absolute_error(y_g, lin_pipe.predict(X_g))
        
        cv_r2_poly = cross_val_score(poly_pipe, X_g, y_g, cv=kf, scoring='r2').mean()
        poly_pipe.fit(X_g, y_g)
        mae_poly = mean_absolute_error(y_g, poly_pipe.predict(X_g))

        cv_r2_log = cross_val_score(log_pipe, X_g, y_g, cv=kf, scoring='r2').mean()
        log_pipe.fit(X_g, y_g)
        mae_log = mean_absolute_error(y_g, log_pipe.predict(X_g))

        print(f"{'Method':<25} | {'CV R2':<10} | {'MAE':<10}")
        print("-" * 50)
        print(f"{'Pure Linear':<25} | {cv_r2_lin:<10.4f} | {mae_lin:<10.2f}")
        print(f"{'Polynomial (D2)':<25} | {cv_r2_poly:<10.4f} | {mae_poly:<10.2f}")
        print(f"{'Logarithmic':<25} | {cv_r2_log:<10.4f} | {mae_log:<10.2f}")

        # Plot Generation (Internal)
        X_smooth = np.linspace(X_g.min(), X_g.max(), 200).reshape(-1, 1)
        plot_configs = [
            {"suffix": "1_Linear_Fit", "title": "Linear Baseline", "pred": lin_pipe.predict(X_smooth), "color": "gray", "ls": "--", "r2": cv_r2_lin, "mae": mae_lin},
            {"suffix": "2_Polynomial_Fit", "title": "Polynomial Fit (Deg 2)", "pred": poly_pipe.predict(X_smooth), "color": "blue", "ls": "-", "r2": cv_r2_poly, "mae": mae_poly},
            {"suffix": "3_Logarithmic_Fit", "title": "Logarithmic Fit", "pred": log_pipe.predict(X_smooth), "color": "red", "ls": "-", "r2": cv_r2_log, "mae": mae_log}
        ]

        for config in plot_configs:
            plt.figure(figsize=(10, 7))
            plt.scatter(X_g, y_g, alpha=0.9, color='#FFFF99', edgecolors='k', label=f"Actual {model} Checkpoints", s=60, zorder=3)
            plt.plot(X_smooth, config["pred"], color=config["color"], linestyle=config["ls"], linewidth=2.5, label="Model Prediction", zorder=2)
            
            plt.title(f"Internal Validation: {model} - {config['title']}", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel("Frobenius Norm (Raw Feature)", fontsize=12)
            plt.ylabel("Training Step", fontsize=12)
            
            metrics_text = (f"Experiment: Internal Validation\n"
                            f"Architecture: {model}\n"
                            f"Algorithm: {config['title']}\n\n"
                            f"Cross-Validation R²: {config['r2']:.4f}\n"
                            f"Mean Absolute Error: {config['mae']:.1f} steps")
            
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props, family='monospace')
            plt.legend(loc='lower right', frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"PART_A_{model}_{config['suffix']}.png"), dpi=300)
            plt.close()


    # ---------------------------------------------------------
    # PART B: ZERO-SHOT EXTRAPOLATION (PURE & SHIFTED)
    # ---------------------------------------------------------
    train_df_raw = df[(df['Model'].isin(['7M', '30M'])) & (df['Method'] == 'Global')].sort_values('Feature_Value')
    test_df_raw = df[(df['Model'] == '124M') & (df['Method'] == 'Global')].sort_values('Step')

    if not train_df_raw.empty and not test_df_raw.empty:
        # Theoretical Scaling Function (Dividing by sqrt(d))
        def apply_theoretical_scaling(df_subset):
            df_scaled = df_subset.copy()
            for m_name, d in EMBED_DIM_MAP.items():
                mask = df_scaled['Model'] == m_name
                if mask.any():
                    df_scaled.loc[mask, 'Feature_Value'] = df_scaled.loc[mask, 'Feature_Value'] / np.sqrt(d)
            return df_scaled

        train_df_theo = apply_theoretical_scaling(train_df_raw)
        test_df_theo = apply_theoretical_scaling(test_df_raw)

        X_tr = train_df_theo[['Feature_Value']].values
        y_tr = train_df_theo['Step'].values
        X_te = test_df_theo[['Feature_Value']].values
        y_te = test_df_theo['Step'].values

        # Shared Pipelines trained ONLY on 7M + 30M
        zs_lin = Pipeline([('reg', LinearRegression())]).fit(X_tr, y_tr)
        zs_poly = Pipeline([('poly', PolynomialFeatures(degree=2)), ('reg', LinearRegression())]).fit(X_tr, y_tr)
        zs_log = Pipeline([('log', FunctionTransformer(np.log, validate=True)), ('reg', LinearRegression())]).fit(X_tr, y_tr)

        X_smooth_te = np.linspace(X_te.min(), X_te.max(), 200).reshape(-1, 1)

        print("\n" + "="*60)
        print("🚀 PART B: ZERO-SHOT EXTRAPOLATION")
        print("   Train: 7M + 30M | Test: 124M")
        print("   Scaling: Theoretical (sqrt(d))")
        print("="*60)
        
        # Pure Zero-Shot Predictions
        pred_lin_raw = zs_lin.predict(X_te)
        pred_poly_raw = zs_poly.predict(X_te)
        pred_log_raw = zs_log.predict(X_te)

        # --- REVERSE FORENSICS: EXPLICIT BIAS SHIFT FOR POLYNOMIAL ---
        # Strictly split into Calibration Set (Last 5) and Holdout Set (The rest)
        K = 5
        X_calib = X_te[-K:]
        y_calib = y_te[-K:]
        X_holdout = X_te[:-K]
        y_holdout = y_te[:-K]

        # Calculate the bias shift based ONLY on the calibration tail
        bias_shift = np.mean(y_calib - zs_poly.predict(X_calib))
        
        # Apply shift strictly for Holdout evaluation
        pred_poly_shifted_holdout = zs_poly.predict(X_holdout) + bias_shift
        # -------------------------------------------------------------

        # Metrics Calculation
        r2_lin_zs, mae_lin_zs = r2_score(y_te, pred_lin_raw), mean_absolute_error(y_te, pred_lin_raw)
        r2_poly_zs, mae_poly_zs = r2_score(y_te, pred_poly_raw), mean_absolute_error(y_te, pred_poly_raw)
        r2_log_zs, mae_log_zs = r2_score(y_te, pred_log_raw), mean_absolute_error(y_te, pred_log_raw)
        
        # Kosher Metrics for Shifted Model (Holdout Only)
        r2_poly_shift = r2_score(y_holdout, pred_poly_shifted_holdout)
        mae_poly_shift = mean_absolute_error(y_holdout, pred_poly_shifted_holdout)

        print(f"{'Method':<25} | {'Test R2':<10} | {'Test MAE':<10}")
        print("-" * 50)
        print(f"{'Pure ZS Linear':<25} | {r2_lin_zs:<10.4f} | {mae_lin_zs:<10.2f}")
        print(f"{'Pure ZS Poly':<25} | {r2_poly_zs:<10.4f} | {mae_poly_zs:<10.2f}")
        print(f"{'Tail Calib Poly (Holdout)':<25} | {r2_poly_shift:<10.4f} | {mae_poly_shift:<10.2f}")
        print(f"{'Pure ZS Log':<25} | {r2_log_zs:<10.4f} | {mae_log_zs:<10.2f}")

        # Plot 4 Separate Graphs including the shifted Polynomial
        zs_configs = [
            {"name": "Linear", "title": "Linear Method", "pred": zs_lin.predict(X_smooth_te), "r2": r2_lin_zs, "mae": mae_lin_zs, "color": "gray", "ls": ":", "bias_text": "NONE", "eval": "All 124M Steps"},
            {"name": "Polynomial", "title": "Polynomial Method", "pred": zs_poly.predict(X_smooth_te), "r2": r2_poly_zs, "mae": mae_poly_zs, "color": "blue", "ls": "-", "bias_text": "NONE", "eval": "All 124M Steps"},
            {"name": "Polynomial_Shifted", "title": "Polynomial (+ Tail Calibration)", "pred": zs_poly.predict(X_smooth_te) + bias_shift, "r2": r2_poly_shift, "mae": mae_poly_shift, "color": "purple", "ls": "-.", "bias_text": f"Shifted by {bias_shift:.1f} steps", "eval": "Holdout Set (Past Steps)"},
            {"name": "Logarithmic", "title": "Logarithmic Method", "pred": zs_log.predict(X_smooth_te), "r2": r2_log_zs, "mae": mae_log_zs, "color": "red", "ls": "--", "bias_text": "NONE", "eval": "All 124M Steps"}
        ]

        for conf in zs_configs:
            plt.figure(figsize=(10, 7))
            
            if conf["name"] == "Polynomial_Shifted":
                # Strict visualization for Calibrated Model
                plt.scatter(X_holdout, y_holdout, alpha=0.9, color='#FFFF99', edgecolors='k', label="Unseen Past Steps (Holdout)", s=60, zorder=3)
                plt.scatter(X_calib, y_calib, color='red', edgecolors='black', label=f"Tail Calibration Set (Last {K})", s=130, marker='*', zorder=4)
            else:
                # Standard visualization for Pure ZS
                plt.scatter(X_te, y_te, alpha=0.9, color='#FFFF99', edgecolors='k', label="Actual 124M Steps", s=60, zorder=3)
                
            plt.plot(X_smooth_te, conf["pred"], color=conf["color"], linestyle=conf["ls"], linewidth=2.5, label=f"ZS {conf['name']}", zorder=2)
            
            plt.title(f"Zero-Shot Extrapolation: {conf['title']}", fontsize=14, fontweight='bold', pad=20)
            plt.xlabel("Scaled Frobenius Norm (124M Data)", fontsize=12)
            plt.ylabel("Training Step", fontsize=12)
            
            metrics_text = (f"Train Set: 7M + 30M\n"
                            f"Test Set: {conf['eval']}\n"
                            f"Scaling: x / sqrt(d)\n"
                            f"Bias Shift: {conf['bias_text']}\n\n"
                            f"Test R²: {conf['r2']:.4f}\n"
                            f"Test MAE: {conf['mae']:.1f}")
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props, family='monospace')
            plt.legend(loc='lower right', frameon=True, shadow=True)
            plt.grid(True, linestyle='--', alpha=0.4, zorder=1)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"PART_B_Zero_Shot_{conf['name']}.png"), dpi=300)
            plt.close()

    # ---------------------------------------------------------
    # PART C: COMBINED GLOBAL BASELINE PLOT
    # ---------------------------------------------------------
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
    plt.savefig(os.path.join(PLOTS_DIR, "PART_C_ALL_MODELS_Combined_Baseline.png"), dpi=300)
    plt.close()

    print(f"\n✅ All analysis complete. Plots saved in: {PLOTS_DIR}")

if __name__ == "__main__":
    run_full_analysis()