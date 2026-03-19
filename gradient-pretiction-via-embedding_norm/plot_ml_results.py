import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. SETUP
# ==========================================
BASE_PATH = os.getcwd() 
if not BASE_PATH.endswith("gradient-pretiction-via-embedding_norm"):
    WORK_DIR = os.path.join(BASE_PATH, "gradient-pretiction-via-embedding_norm")
else:
    WORK_DIR = BASE_PATH

INPUT_CSV = os.path.join(WORK_DIR, "embedding_norms_empirical.csv")
PLOTS_DIR = os.path.join(WORK_DIR, "plots_empirical_ml")

# Create fresh directory
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 2. PLOTTING HELPER
# ==========================================
def plot_results(x_data, y_true, y_pred, title, x_label, y_label, metrics_text, filename, is_true_vs_pred=False):
    plt.figure(figsize=(9, 7))
    
    if is_true_vs_pred:
        # For Multi-feature Bins: True vs Predicted
        plt.scatter(x_data, y_pred, alpha=0.6, color='blue', edgecolors='k', label='Model Prediction')
        # The 0-error baseline
        min_val = min(x_data.min(), y_pred.min())
        max_val = max(x_data.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='0-Error Line (Perfect Match)')
    else:
        # For Single-feature Global Norm: Feature vs Step
        plt.scatter(x_data, y_true, alpha=0.6, color='gray', label='True Step')
        plt.plot(x_data, y_pred, color='blue', linewidth=2, label='Model Prediction Line')

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # Add metrics text box inside the plot
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
            
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=300)
    plt.close()

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def evaluate_and_plot():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ CSV not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    df['Bin_ID'] = df['Bin_ID'].astype(str)
    models = ['7M', '30M', '124M']

    for model in models:
        print(f"\nEvaluating Model: {model}...")
        model_df = df[df['Model'] == model]

        # ---------------------------------------------------------
        # TASK 1: Global Baseline (CV for R2, fit over all for Plot)
        # ---------------------------------------------------------
        global_df = model_df[model_df['Method'] == 'Global'].sort_values(by='Feature_Value')
        if not global_df.empty:
            X_g = global_df[['Feature_Value']].values
            y_g = global_df['Step'].values
            
            # CV for scientific metric
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(LinearRegression(), X_g, y_g, cv=kf, scoring='r2')
            
            # Fit on all to plot the line vs reality (as requested: X=Norm, Y=Step)
            reg_g = LinearRegression().fit(X_g, y_g)
            y_pred_g = reg_g.predict(X_g)
            mae_g = mean_absolute_error(y_g, y_pred_g)
            
            text_g = (f"Model Architecture: {model}\n"
                      f"Data: Global Frobenius Norm\n"
                      f"Evaluation: 5-Fold CV\n"
                      f"R² (Mean): {cv_scores.mean():.4f}\n"
                      f"MAE: {mae_g:.2f} steps")
                      
            plot_results(
                x_data=X_g, y_true=y_g, y_pred=y_pred_g,
                title=f"{model} - Global Embedding Norm vs Training Step",
                x_label="Embedding Norm (Frobenius)", y_label="Training Step",
                metrics_text=text_g, filename=f"{model}_1_Global_Baseline.png", is_true_vs_pred=False
            )

        # ---------------------------------------------------------
        # TASK 2: Frequency Bins (80/20 Train/Test Split)
        # ---------------------------------------------------------
        freq_df = model_df[model_df['Method'] == 'Frequency']
        if not freq_df.empty:
            X_f = freq_df[['Feature_Value', 'Bin_ID']].values
            y_f = freq_df['Step'].values
            
            X_train, X_test, y_train, y_test = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
            reg_f = LinearRegression().fit(X_train, y_train)
            y_pred_test = reg_f.predict(X_test)
            
            r2_f = r2_score(y_test, y_pred_test)
            mae_f = mean_absolute_error(y_test, y_pred_test)
            
            text_f = (f"Model Architecture: {model}\n"
                      f"Data: Frequency Bins (~3000 samples)\n"
                      f"Evaluation: Unseen Test Set (20%)\n"
                      f"R²: {r2_f:.4f}\n"
                      f"MAE: {mae_f:.2f} steps")
            
            plot_results(
                x_data=y_test, y_true=y_test, y_pred=y_pred_test,
                title=f"{model} - Frequency Bins ML Prediction Errors",
                x_label="True Training Step", y_label="Predicted Training Step",
                metrics_text=text_f, filename=f"{model}_2_Frequency_Bins_ML.png", is_true_vs_pred=True
            )

        # ---------------------------------------------------------
        # TASK 3: Random Bins Control (80/20 Train/Test Split)
        # ---------------------------------------------------------
        rand_df = model_df[model_df['Method'] == 'Random']
        if not rand_df.empty:
            X_r = rand_df[['Feature_Value', 'Bin_ID']].values
            y_r = rand_df['Step'].values
            
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
            reg_r = LinearRegression().fit(X_train_r, y_train_r)
            y_pred_test_r = reg_r.predict(X_test_r)
            
            r2_r = r2_score(y_test_r, y_pred_test_r)
            mae_r = mean_absolute_error(y_test_r, y_pred_test_r)
            
            text_r = (f"Model Architecture: {model}\n"
                      f"Data: Random Bins Control (~3000 samples)\n"
                      f"Evaluation: Unseen Test Set (20%)\n"
                      f"R²: {r2_r:.4f}\n"
                      f"MAE: {mae_r:.2f} steps")
            
            plot_results(
                x_data=y_test_r, y_true=y_test_r, y_pred=y_pred_test_r,
                title=f"{model} - Random Bins Control ML Prediction Errors",
                x_label="True Training Step", y_label="Predicted Training Step",
                metrics_text=text_r, filename=f"{model}_3_Random_Bins_Control_ML.png", is_true_vs_pred=True
            )

    print(f"\n✅ Clean scientific plots generated successfully in: {PLOTS_DIR}")

if __name__ == "__main__":
    evaluate_and_plot()