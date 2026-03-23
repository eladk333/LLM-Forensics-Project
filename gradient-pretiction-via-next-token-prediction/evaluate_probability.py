import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = os.getcwd() 
WORK_DIR = os.path.join(BASE_PATH, "gradient-pretiction-via-next-token-prediction")
CSV_PATH = os.path.join(WORK_DIR, "probability_results_batch_level.csv")
OUTPUT_IMG_DIR = os.path.join(WORK_DIR, "plots_complete_analysis")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ==========================================
# FORENSICS MODEL LOGIC (Exponential)
# ==========================================
def inverse_model(prob, a, b):
    # Equation: Step = a * e^(b * Prob)
    return a * np.exp(b * prob)

# ==========================================
# MAIN UNIFIED SCRIPT
# ==========================================
def run_full_analysis_and_plotting():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found at {CSV_PATH}")
        return

    print(f"📂 Loading data from {CSV_PATH}...")
    df_batches = pd.read_csv(CSV_PATH)
    
    # Fix Final Model Step
    if 1 in df_batches['Step'].values:
        max_step = df_batches['Step'].max()
        df_batches.loc[df_batches['Step'] == 1, 'Step'] = max_step + 500

    print(f"⚙️ Deriving 60-point Baseline dataset...")
    df_baseline = df_batches.groupby(['Model', 'Step'])['Batch_Probability'].mean().reset_index()
    
    models = ['Model 7M', 'Model 30M', 'Model 124M'] # Sorted for consistent plotting
    colors = {'Model 124M': 'blue', 'Model 30M': 'green', 'Model 7M': 'red'}
    
    stats_report = []

    print(f"📊 Running ML Evaluation & Generating All 12 Plots...")

    # ==========================================
    # PART 1: INDIVIDUAL MODEL ANALYSIS & PLOTS
    # ==========================================
    for model_name in models:
        c = colors.get(model_name, 'black')
        
        # --- Data Prep ---
        subset_base = df_baseline[df_baseline['Model'] == model_name]
        X_base = subset_base["Batch_Probability"].values
        Y_base = subset_base["Step"].values
        max_step_base = Y_base.max()

        subset_batch = df_batches[df_batches['Model'] == model_name]
        X_batch = subset_batch["Batch_Probability"].values
        Y_batch = subset_batch["Step"].values

        # --- Experiment 1: Baseline (Cross-Validation) ---
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores_cv, mae_scores_cv = [], []
        for train_idx, test_idx in kf.split(X_base):
            try:
                popt_cv, _ = curve_fit(inverse_model, X_base[train_idx], Y_base[train_idx], maxfev=5000)
                y_pred_cv = inverse_model(X_base[test_idx], *popt_cv)
                r2_scores_cv.append(r2_score(Y_base[test_idx], y_pred_cv))
                mae_scores_cv.append(mean_absolute_error(Y_base[test_idx], y_pred_cv))
            except: pass
        
        base_r2 = np.mean(r2_scores_cv) if r2_scores_cv else 0
        base_mae = np.mean(mae_scores_cv) if mae_scores_cv else 0
        popt_base_all, _ = curve_fit(inverse_model, X_base, Y_base, maxfev=5000)
        a_base, b_base = popt_base_all

        # --- Experiment 2: Batches (Train/Test Split) ---
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_batch, Y_batch, test_size=0.2, random_state=42)
        try:
            popt_batch, _ = curve_fit(inverse_model, X_train_b, y_train_b, maxfev=5000)
            y_pred_batch = inverse_model(X_test_b, *popt_batch)
            batch_r2 = r2_score(y_test_b, y_pred_batch)
            batch_mae = mean_absolute_error(y_test_b, y_pred_batch)
            a_batch, b_batch = popt_batch
        except:
            batch_r2, batch_mae, a_batch, b_batch = 0, 0, 0, 0

        # Store Stats for Terminal Table
        stats_report.append({
            "Model": model_name,
            "Base_R2": base_r2, "Base_MAE": base_mae, "Base_Eq": f"Step={a_base:.2f}*e^({b_base:.2f}*p)",
            "Batch_R2": batch_r2, "Batch_MAE": batch_mae, "Batch_Eq": f"Step={a_batch:.2f}*e^({b_batch:.2f}*p)"
        })

        # --- PLOT 1: Individual Raw Dynamics (X=Steps, Y=Probability) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(Y_batch, X_batch, color=c, alpha=0.03, s=15, label='Batch Probabilities')
        plt.plot(Y_base, X_base, color='black', marker='o', linestyle='-', markersize=4, label='Mean Probability')
        plt.title(f"{model_name} Raw Dynamics\nProbability of Next Token Over Time", fontsize=14)
        plt.xlabel("Gradient Updates (Training Steps)", fontsize=12, fontweight='bold')
        plt.ylabel("Next Token Probability", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        leg = plt.legend()
        for lh in leg.legend_handles: lh.set_alpha(1)
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"1_raw_dynamics_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

        # --- PLOT 2: Individual Baseline Forensics (X=Probability, Y=Steps) ---
        plt.figure(figsize=(10, 6))
        plt.scatter(X_base[:-1], Y_base[:-1], color=c, alpha=0.8, s=60, label='Checkpoints (Mean)')
        plt.scatter(X_base[-1], Y_base[-1], color='yellow', edgecolor='black', marker='*', s=400, label='Full Epoch', zorder=10)
        px_base = np.linspace(X_base.min(), X_base.max(), 100)
        plt.plot(px_base, inverse_model(px_base, *popt_base_all), color='black', linestyle='--', linewidth=2.5, label=f'Predictor (CV R²={base_r2:.3f})')
        plt.title(f"{model_name} Baseline Forensics\nMethod: 5-Fold Cross Validation\nStep = {a_base:.2f} * e^({b_base:.2f} * Prob)", fontsize=14)
        plt.xlabel("Input: Mean Next Token Probability", fontsize=12, fontweight='bold')
        plt.ylabel("Target: Gradient Updates (Steps)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"2_baseline_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

        # --- PLOT 3: Individual Batch Cloud Forensics (X=Probability, Y=Steps) ---
        plt.figure(figsize=(10, 6))
        
        # חלוקת ה-Test Set ל"לא הצעד האחרון" ו"הצעד האחרון" בשביל הצבעים
        mask_test_not_last = y_test_b < max_step_base
        plt.scatter(X_test_b[mask_test_not_last], y_test_b[mask_test_not_last], color=c, alpha=0.1, s=25, label='Test Set Batches')
        
        mask_test_last = y_test_b == max_step_base
        plt.scatter(X_test_b[mask_test_last], y_test_b[mask_test_last], color='darkred', alpha=0.3, s=30, label='Final Epoch Batches (Test)')
        
        # קו החיזוי
        px_batch = np.linspace(X_test_b.min(), X_test_b.max(), 100)
        plt.plot(px_batch, inverse_model(px_batch, *popt_batch), color='black', linestyle='--', linewidth=2.5, label=f'Predictor (Test R²={batch_r2:.3f})')
        
        plt.title(f"{model_name} Batch Forensics\nMethod: 80/20 Train-Test Split (Test Data Only)\nStep = {a_batch:.2f} * e^({b_batch:.2f} * Prob)", fontsize=14)
        plt.xlabel("Input: Single Batch Next Token Probability", fontsize=12, fontweight='bold')
        plt.ylabel("Target: Gradient Updates (Steps)", fontsize=12, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        leg = plt.legend(loc='upper left')
        for lh in leg.legend_handles: lh.set_alpha(1)
        
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"3_batch_cloud_{model_name.replace(' ', '_')}.png"), dpi=300)
        plt.close()

        
    # ==========================================
    # PART 2: COMBINED PLOTS (ALL MODELS)
    # ==========================================
    # COMBINED 1: Raw Dynamics (Including Batch Clouds)
    plt.figure(figsize=(12, 7))
    for model_name in models:
        c = colors.get(model_name, 'black')
        sub_batch = df_batches[df_batches['Model'] == model_name]
        sub_base = df_baseline[df_baseline['Model'] == model_name]
        if not sub_batch.empty:
            plt.scatter(sub_batch['Step'], sub_batch['Batch_Probability'], color=c, alpha=0.01, s=10)
        if not sub_base.empty:
            plt.plot(sub_base['Step'], sub_base['Batch_Probability'], color=c, marker='o', linestyle='-', label=model_name)
    plt.title("Combined Raw Dynamics\nNext Token Probability (Batches & Mean) vs. Steps", fontsize=16)
    plt.xlabel("Gradient Updates (Training Steps)", fontweight='bold')
    plt.ylabel("Next Token Probability", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    leg = plt.legend(loc='lower right')
    for lh in leg.legend_handles: lh.set_alpha(1)
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, "4_combined_raw_dynamics.png"), dpi=300)
    plt.close()

    # COMBINED 2: Baseline
    plt.figure(figsize=(12, 7))
    for model_name in models:
        c = colors.get(model_name, 'black')
        sub_base = df_baseline[df_baseline['Model'] == model_name]
        if not sub_base.empty:
            X, Y = sub_base['Batch_Probability'].values, sub_base['Step'].values
            plt.scatter(X, Y, color=c, alpha=0.7, s=50)
            popt, _ = curve_fit(inverse_model, X, Y, maxfev=5000)
            px = np.linspace(X.min(), X.max(), 100)
            plt.plot(px, inverse_model(px, *popt), color=c, linestyle='--', linewidth=2, label=model_name)
    plt.title("Combined Baseline Forensics (60 Points)\nMethod: 5-Fold Cross Validation", fontsize=16)
    plt.xlabel("Input: Mean Next Token Probability", fontweight='bold')
    plt.ylabel("Target: Gradient Updates (Steps)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, "5_combined_baseline.png"), dpi=300)
    plt.close()

    # COMBINED 3: Batch Cloud
    plt.figure(figsize=(12, 7))
    for model_name in models:
        c = colors.get(model_name, 'black')
        sub_batch = df_batches[df_batches['Model'] == model_name]
        if not sub_batch.empty:
            X, Y = sub_batch['Batch_Probability'].values, sub_batch['Step'].values
            plt.scatter(X, Y, color=c, alpha=0.015, s=15)
            popt, _ = curve_fit(inverse_model, X, Y, maxfev=5000)
            px = np.linspace(X.min(), X.max(), 100)
            plt.plot(px, inverse_model(px, *popt), color=c, linestyle='-', linewidth=3, label=model_name)
    plt.title("Combined Empirical Batch Forensics\nMethod: 80/20 Train-Test Split", fontsize=16)
    plt.xlabel("Input: Single Batch Next Token Probability", fontweight='bold')
    plt.ylabel("Target: Gradient Updates (Steps)", fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    leg = plt.legend(loc='upper left')
    for lh in leg.legend_handles: lh.set_alpha(1)
    plt.savefig(os.path.join(OUTPUT_IMG_DIR, "6_combined_batch_cloud.png"), dpi=300)
    plt.close()

    # ==========================================
    # PART 3: PRINT TERMINAL REPORT
    # ==========================================
    print("\n" + "="*110)
    print(f"{'FINAL FORENSICS REPORT: BASELINE (CV) vs BATCHES (TRAIN/TEST)':^110}")
    print("="*110)
    print(f"{'Model Name':<12} | {'Base R² (CV)':<13} | {'Base MAE':<10} | {'Batch R² (Test)':<15} | {'Batch MAE':<10} | {'Batch Eq'}")
    print("-" * 110)
    for item in stats_report:
        print(f"{item['Model']:<12} | {item['Base_R2']:<13.4f} | +/- {item['Base_MAE']:<6.1f} | {item['Batch_R2']:<15.4f} | +/- {item['Batch_MAE']:<6.1f} | {item['Batch_Eq']}")
    print("-" * 110 + "\n✅ All 12 plots saved to: " + OUTPUT_IMG_DIR + "\n")

if __name__ == "__main__":
    run_full_analysis_and_plotting()