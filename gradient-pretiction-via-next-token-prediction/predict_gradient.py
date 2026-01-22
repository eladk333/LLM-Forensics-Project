import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = os.getcwd() 
CSV_PATH = os.path.join(BASE_PATH, "probability_results_server.csv")

# ==========================================
# FORENSICS MODEL LOGIC
# ==========================================
def inverse_model(prob, a, b):
    return a * np.exp(b * prob)

# ==========================================
# MAIN LOGIC
# ==========================================
def run_full_analysis():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV not found at {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Fix Final Model Step
    if 1 in df['Step'].values:
        max_step = df['Step'].max()
        df.loc[df['Step'] == 1, 'Step'] = max_step + 500

    df = df.sort_values(by=["Model", "Step"])
    models = df['Model'].unique()
    colors = {'Model 124M': 'blue', 'Model 30M': 'green', 'Model 7M': 'red'}
    
    # List to store statistics for the final report
    stats_report = []

    print(f"📊 Starting Analysis and Graph Generation...")

    # --- GRAPH GENERATION LOOP ---
    for model_name in models:
        subset = df[df['Model'] == model_name]
        if subset.empty: continue

        X_probs = subset["Avg_Probability"].values
        Y_steps = subset["Step"].values
        
        c = colors.get(model_name, 'black')
        
        # Start Plot
        plt.figure(figsize=(12, 7))
        
        # 1. Plot all points (except the last one)
        plt.scatter(X_probs[:-1], Y_steps[:-1], color=c, alpha=0.6, label='Checkpoints', s=60)

        # 2. Plot the LAST point as a Yellow Star
        plt.scatter(X_probs[-1], Y_steps[-1], 
                    color='yellow', edgecolor='black', marker='*', s=400, 
                    label='Full Epoch (Final State)', zorder=10)

        # 3. Curve Fitting & Statistics
        try:
            popt, _ = curve_fit(inverse_model, X_probs, Y_steps, maxfev=5000)
            a, b = popt
            
            # Generate line for plot
            x_line = np.linspace(X_probs.min(), X_probs.max(), 100)
            y_pred_line = inverse_model(x_line, *popt)
            
            # Calculate R2
            y_pred_actual = inverse_model(X_probs, *popt)
            r2 = r2_score(Y_steps, y_pred_actual)
            
            # Plot the line
            label_fit = f'Prediction Model (R²={r2:.3f})'
            plt.plot(x_line, y_pred_line, color='black', linestyle='--', linewidth=2.5, label=label_fit)
            
            # Set Title with Equation
            equation = f'Step = {a:.2f} * e^({b:.2f} * Prob)'
            plt.title(f"{model_name} Forensics Analysis\n{equation}", fontsize=15)
            
            # Store stats for the final report
            stats_report.append({
                "Model": model_name,
                "R2": r2,
                "Max_Step": int(Y_steps.max()),
                "Equation": f"Step = {a:.2f} * e^({b:.2f} * p)"
            })

        except Exception as e:
            print(f"⚠️ Fit failed for {model_name}: {e}")
            plt.title(f"{model_name} (Raw Data)", fontsize=15)
            stats_report.append({
                "Model": model_name,
                "R2": 0.0,
                "Max_Step": 0,
                "Equation": "Fit Failed"
            })

        # 4. Final Formatting
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        plt.xlabel("Input: Next Token Probability", fontsize=13, fontweight='bold')
        plt.ylabel("Predicted: Gradient Updates (Steps)", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=11)
        
        # Save Graph
        filename = f"forensics_clean_{model_name.replace(' ', '_')}.png"
        output_path = os.path.join(BASE_PATH, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Graph Saved: {filename}")

    # --- PRINT FINAL REPORT TABLE ---
    print("\n" + "="*80)
    print(f"{'FINAL FORENSICS REPORT':^80}")
    print("="*80)
    print(f"{'Model Name':<15} | {'Accuracy (R²)':<15} | {'Max Steps':<12} | {'Formula'}")
    print("-" * 80)
    
    for item in stats_report:
        print(f"{item['Model']:<15} | {item['R2']:<15.4f} | {item['Max_Step']:<12,} | {item['Equation']}")
    
    print("-" * 80)
    print("Use these values for your presentation.")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_full_analysis()