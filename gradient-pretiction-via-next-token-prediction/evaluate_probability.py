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
OUTPUT_IMG_DIR = os.path.join(WORK_DIR, "plots_comprehensive")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# ==========================================
# FORENSICS MODEL LOGIC (Exponential)
# ==========================================
def inverse_model(prob, a, b):
    # Equation: Step = a * e^(b * Prob)
    return a * np.exp(b * prob)

# ==========================================
# MAIN LOGIC
# ==========================================
def run_full_analysis():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found at {CSV_PATH}")
        return

    print(f"📂 Loading massive batch-level data from {CSV_PATH}...")
    df_batches = pd.read_csv(CSV_PATH)
    
    # Fix Final Model Step
    if 1 in df_batches['Step'].values:
        max_step = df_batches['Step'].max()
        df_batches.loc[df_batches['Step'] == 1, 'Step'] = max_step + 500

    # Derive the 60-point baseline by averaging the batches per step
    print(f"⚙️ Deriving 60-point Baseline dataset for comparison...")
    df_baseline = df_batches.groupby(['Model', 'Step'])['Batch_Probability'].mean().reset_index()
    
    models = df_batches['Model'].unique()
    colors = {'Model 124M': 'blue', 'Model 30M': 'green', 'Model 7M': 'red'}
    
    stats_report = []

    print(f"📊 Starting Dual-Experiment Analysis...")

    for model_name in models:
        c = colors.get(model_name, 'black')
        
        # ---------------------------------------------------------
        # DATASET 1: THE BASELINE (60 Points)
        # ---------------------------------------------------------
        subset_base = df_baseline[df_baseline['Model'] == model_name]
        X_base = subset_base["Batch_Probability"].values
        Y_base = subset_base["Step"].values
        max_step_base = Y_base.max()

        # Experiment 1: 5-Fold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores_cv, mae_scores_cv = [], []
        
        for train_idx, test_idx in kf.split(X_base):
            X_train_cv, X_test_cv = X_base[train_idx], X_base[test_idx]
            y_train_cv, y_test_cv = Y_base[train_idx], Y_base[test_idx]
            try:
                popt_cv, _ = curve_fit(inverse_model, X_train_cv, y_train_cv, maxfev=5000)
                y_pred_cv = inverse_model(X_test_cv, *popt_cv)
                r2_scores_cv.append(r2_score(y_test_cv, y_pred_cv))
                mae_scores_cv.append(mean_absolute_error(y_test_cv, y_pred_cv))
            except:
                pass
        
        base_r2 = np.mean(r2_scores_cv) if r2_scores_cv else 0
        base_mae = np.mean(mae_scores_cv) if mae_scores_cv else 0
        
        # Fit on all 60 points JUST for the visual curve equation
        popt_base_all, _ = curve_fit(inverse_model, X_base, Y_base, maxfev=5000)
        a_base, b_base = popt_base_all

        # ---------------------------------------------------------
        # DATASET 2: THE EMPIRICAL BATCHES (60,000 Points)
        # ---------------------------------------------------------
        subset_batch = df_batches[df_batches['Model'] == model_name]
        X_batch = subset_batch["Batch_Probability"].values
        Y_batch = subset_batch["Step"].values

        # Experiment 2: 80/20 Train/Test Split
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_batch, Y_batch, test_size=0.2, random_state=42)
        
        try:
            popt_batch, _ = curve_fit(inverse_model, X_train_b, y_train_b, maxfev=5000)
            y_pred_batch = inverse_model(X_test_b, *popt_batch)
            batch_r2 = r2_score(y_test_b, y_pred_batch)
            batch_mae = mean_absolute_error(y_test_b, y_pred_batch)
            a_batch, b_batch = popt_batch
        except:
            batch_r2, batch_mae, a_batch, b_batch = 0, 0, 0, 0

        # Store Stats
        stats_report.append({
            "Model": model_name,
            "Base_R2": base_r2, "Base_MAE": base_mae, "Base_Eq": f"Step = {a_base:.2f} * e^({b_base:.2f}*p)",
            "Batch_R2": batch_r2, "Batch_MAE": batch_mae, "Batch_Eq": f"Step = {a_batch:.2f} * e^({b_batch:.2f}*p)"
        })

        # ==========================================
        # GRAPH GENERATION (3 Variants)
        # ==========================================
        x_line = np.linspace(X_batch.min(), X_batch.max(), 100)

        # 1. Separated Graph A: Baseline Only (Similar to previous presentation)
        plt.figure(figsize=(12, 7))
        plt.scatter(X_base[:-1], Y_base[:-1], color=c, alpha=0.8, label='Checkpoints (Mean)', s=60)
        plt.scatter(X_base[-1], Y_base[-1], color='yellow', edgecolor='black', marker='*', s=400, label='Full Epoch', zorder=10)
        plt.plot(x_line, inverse_model(x_line, *popt_base_all), color='black', linestyle='--', linewidth=2.5, label=f'Model (CV R²={base_r2:.3f})')
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel("Input: Next Token Probability", fontsize=13, fontweight='bold')
        plt.ylabel("Predicted: Gradient Updates (Steps)", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=11)
        plt.title(f"{model_name} Baseline (60 Points)\nStep = {a_base:.2f} * e^({b_base:.2f} * Prob)", fontsize=15)
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"1_separated_baseline_{model_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Separated Graph B: Batch Cloud Only (The New Method)
        plt.figure(figsize=(12, 7))
        # Plot all points EXCEPT the last epoch as a light cloud
        mask_not_last = Y_batch < max_step_base
        plt.scatter(X_batch[mask_not_last], Y_batch[mask_not_last], color=c, alpha=0.03, s=15, label='Batch Samples')
        
        # Plot the final epoch points with a darker shade to highlight the variance at the end
        mask_last = Y_batch == max_step_base
        plt.scatter(X_batch[mask_last], Y_batch[mask_last], color='darkred', alpha=0.15, s=20, label='Final Epoch Batches')
        
        if batch_r2 > 0:
            plt.plot(x_line, inverse_model(x_line, *popt_batch), color='black', linestyle='--', linewidth=2.5, label=f'Model (Test R²={batch_r2:.3f})')
        
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel("Input: Batch Next Token Probability", fontsize=13, fontweight='bold')
        plt.ylabel("Predicted: Gradient Updates (Steps)", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        leg = plt.legend(loc='upper left', fontsize=11)
        for lh in leg.legend_handles: 
            lh.set_alpha(1) # Make legend icons fully opaque
            
        plt.title(f"{model_name} Batch Analysis (Unseen Test Data)\nStep = {a_batch:.2f} * e^({b_batch:.2f} * Prob)", fontsize=15)
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"2_separated_batches_{model_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Combined Graph (Just in case you want to compare visually)
        plt.figure(figsize=(12, 7))
        plt.scatter(X_batch, Y_batch, color='gray', alpha=0.02, s=10, label='Batch Cloud')
        plt.scatter(X_base[:-1], Y_base[:-1], color=c, alpha=0.9, s=60, edgecolor='white', label='Checkpoints (Mean)')
        plt.scatter(X_base[-1], Y_base[-1], color='yellow', edgecolor='black', marker='*', s=300, label='Full Epoch Mean', zorder=10)
        plt.plot(x_line, inverse_model(x_line, *popt_base_all), color='black', linestyle='--', linewidth=2)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.title(f"{model_name} Combined View", fontsize=15)
        leg = plt.legend(loc='upper left', fontsize=11)
        for lh in leg.legend_handles: lh.set_alpha(1)
        plt.savefig(os.path.join(OUTPUT_IMG_DIR, f"3_combined_view_{model_name.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Generated 3 graphs for {model_name}")

    # --- PRINT FINAL REPORT TABLE ---
    print("\n" + "="*110)
    print(f"{'FINAL FORENSICS REPORT: BASELINE (CV) vs BATCHES (TRAIN/TEST)':^110}")
    print("="*110)
    print(f"{'Model Name':<12} | {'Base R² (CV)':<13} | {'Base MAE':<10} | {'Batch R² (Test)':<15} | {'Batch MAE':<10} | {'Batch Formula'}")
    print("-" * 110)
    
    for item in stats_report:
        print(f"{item['Model']:<12} | {item['Base_R2']:<13.4f} | +/- {item['Base_MAE']:<6.1f} | {item['Batch_R2']:<15.4f} | +/- {item['Batch_MAE']:<6.1f} | {item['Batch_Eq']}")
    
    print("-" * 110)
    print("Use these values to show the exact improvement of the empirical approach.")
    print("="*110 + "\n")

if __name__ == "__main__":
    run_full_analysis()