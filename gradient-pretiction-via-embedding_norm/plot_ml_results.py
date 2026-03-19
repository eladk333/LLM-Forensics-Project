import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

# Create fresh directory for clean results
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
    
    # CASE A: Multiple Raw Series (Clouds)
    if multi_series:
        colors = ['#1f77b4', '#d62728', '#7f7f7f'] # Blue, Red, Gray
        for i, (x, y, lbl) in enumerate(multi_series):
            plt.scatter(x, y, alpha=0.5, color=colors[i], label=lbl, edgecolors='none', s=20)
    
    # CASE B: Standard Scatter + Optional Prediction Line
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
    df['Bin_ID'] = df['Bin_ID'].astype(str)
    models = ['7M', '30M', '124M']

    for model in models:
        print(f"🎨 Generating all plots for {model}...")
        m_df = df[df['Model'] == model]

        # --- PLOT 0: RAW DATA DYNAMICS (Steps vs Norm) ---
        b0 = m_df[(m_df['Method'] == 'Frequency') & (m_df['Bin_ID'] == '0')].sort_values('Step')
        b40 = m_df[(m_df['Method'] == 'Frequency') & (m_df['Bin_ID'] == '40')].sort_values('Step')
        r0 = m_df[(m_df['Method'] == 'Random') & (m_df['Bin_ID'] == '0')].sort_values('Step')

        plot_master(
            x_data=None, y_data=None,
            title=f"{model} - Raw Dynamics: Steps vs Embedding Norm",
            x_label="Training Steps", y_label="Embedding L2 Norm",
            filename=f"{model}_0_Raw_Dynamics.png",
            multi_series=[(b0['Step'], b0['Feature_Value'], "Top 1000 Tokens (Bin 0)"),
                          (b40['Step'], b40['Feature_Value'], "Rare Tokens (Bin 40)"),
                          (r0['Step'], r0['Feature_Value'], "Random Control")]
        )

        # --- PLOT 0.5: RAW GLOBAL BASELINE (Steps vs Norm - No ML) ---
        # ADDED: This shows the raw correlation without the regression target swap
        g_raw = m_df[m_df['Method'] == 'Global'].sort_values('Step')
        plot_master(
            x_data=g_raw['Step'], y_data=g_raw['Feature_Value'],
            title=f"{model} - Raw Global Baseline: Steps vs Frobenius Norm",
            x_label="Training Steps", y_label="Global Frobenius Norm",
            filename=f"{model}_0.5_Raw_Global_Baseline.png",
            scatter_label="Global Checkpoints"
        )

        # --- PLOT 1: GLOBAL BASELINE (Norm vs Step) ---
        g_df = m_df[m_df['Method'] == 'Global'].sort_values('Feature_Value')
        X_g = g_df[['Feature_Value']].values
        y_g = g_df['Step'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(LinearRegression(), X_g, y_g, cv=kf, scoring='r2').mean()
        reg_g = LinearRegression().fit(X_g, y_g)
        mae_g = mean_absolute_error(y_g, reg_g.predict(X_g))

        plot_master(
            x_data=g_df['Feature_Value'], y_data=y_g,
            title=f"{model} - Global Baseline: Norm vs Training Step",
            x_label="Frobenius Norm (Input Feature)", y_label="Training Step (Target)",
            filename=f"{model}_1_Global_ML_Regression.png",
            scatter_label="Actual Checkpoints",
            line_x=g_df['Feature_Value'], line_y=reg_g.predict(X_g),
            line_label="Linear Regression Fit",
            metrics_text=f"Arch: {model}\nMethod: Global Norm\nCV R²: {cv_r2:.4f}\nMAE: {mae_g:.2f} steps"
        )

        # --- PLOT 2: FREQUENCY BINS ML (True vs Pred) ---
        f_df = m_df[m_df['Method'] == 'Frequency']
        X_f = f_df[['Feature_Value', 'Bin_ID']].values
        y_f = f_df['Step'].values
        X_tr, X_te, y_tr, y_te = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
        reg_f = LinearRegression().fit(X_tr, y_tr)
        y_pred = reg_f.predict(X_te)

        plot_master(
            x_data=y_te, y_data=y_pred,
            title=f"{model} - Frequency Bins: ML Performance",
            x_label="True Step (Ground Truth)", y_label="Predicted Step (Model Output)",
            filename=f"{model}_2_Frequency_ML_Performance.png",
            scatter_label="Predictions (Test Set)",
            line_x=[y_te.min(), y_te.max()], line_y=[y_te.min(), y_te.max()],
            line_label="Perfect Prediction (X=Y)",
            metrics_text=f"Arch: {model}\nData: Frequency Bins\nR² (Test): {r2_score(y_te, y_pred):.4f}\nMAE: {mean_absolute_error(y_te, y_pred):.2f}"
        )

        # --- PLOT 3: RANDOM BINS ML (True vs Pred) ---
        # ADDED: This is essential to show that Random Bins perform differently
        r_df = m_df[m_df['Method'] == 'Random']
        X_r = r_df[['Feature_Value', 'Bin_ID']].values
        y_r = r_df['Step'].values
        X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
        reg_r = LinearRegression().fit(X_tr_r, y_tr_r)
        y_pred_r = reg_r.predict(X_te_r)

        plot_master(
            x_data=y_te_r, y_data=y_pred_r,
            title=f"{model} - Random Bins: ML Performance (Control)",
            x_label="True Step (Ground Truth)", y_label="Predicted Step (Model Output)",
            filename=f"{model}_3_Random_ML_Performance.png",
            scatter_label="Predictions (Test Set)",
            line_x=[y_te_r.min(), y_te_r.max()], line_y=[y_te_r.min(), y_te_r.max()],
            line_label="Perfect Prediction (X=Y)",
            metrics_text=f"Arch: {model}\nData: Random Bins\nR² (Test): {r2_score(y_te_r, y_pred_r):.4f}\nMAE: {mean_absolute_error(y_te_r, y_pred_r):.2f}"
        )

    # --- FINAL PLOT: COMBINED GLOBAL BASELINE ---
    print("🎨 Generating Combined Architecture Comparison...")
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

    print(f"\n✅ All scientific plots generated in {PLOTS_DIR}")

if __name__ == "__main__":
    run_full_analysis()