import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ==========================================
# CONFIGURATION
# ==========================================
# Uses the current working directory (Server compatible)
BASE_PATH = os.getcwd() 
CSV_PATH = os.path.join(BASE_PATH, "probability_results_server.csv")
OUTPUT_IMAGE = os.path.join(BASE_PATH, "final_graph_server.png")

# ==========================================
# PLOTTING LOGIC
# ==========================================
def plot_graph():
    # 1. Validation
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at {CSV_PATH}")
        print("   Please run the data extraction script first.")
        return

    # 2. Load Data
    print(f"📂 Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # 🔧 FIX: Handle 'Final Model' appearing as Step 1
    # If 'final_model_1_epoch.pt' was read as Step 1, we move it to the end.
    if 1 in df['Step'].values:
        max_step = df['Step'].max()
        new_step = max_step + 500
        print(f"🔧 Correcting 'Step 1' (Final Model) -> Reassigning to Step {new_step}")
        df.loc[df['Step'] == 1, 'Step'] = new_step

    # Sort data for clean plotting
    df = df.sort_values(by=["Model", "Step"])
    
    # 3. Setup Plot
    plt.figure(figsize=(12, 7))
    
    # Define colors for consistency
    colors = {
        'Model 124M': 'blue', 
        'Model 30M': 'green', 
        'Model 7M': 'red'
    }
    
    models = df['Model'].unique()
    print(f"📊 Found data for models: {models}")

    # 4. Iterate over each model to plot
    for model_name in models:
        subset = df[df['Model'] == model_name]
        
        if subset.empty:
            continue

        steps = subset["Step"].values
        probs = subset["Avg_Probability"].values
        
        # Get color (default to black if unknown)
        c = colors.get(model_name, 'black')
        
        # A. Plot Raw Data Points
        plt.scatter(steps, probs, color=c, alpha=0.6, label=f'{model_name} (Observed)')
        
        # B. Regression: Fit Logarithmic Curve
        # Hypothesis: Probability grows logarithmically: y = a + b * ln(x)
        try:
            def log_func(x, a, b): 
                return a + b * np.log(x)
            
            popt, _ = curve_fit(log_func, steps, probs)
            probs_pred = log_func(steps, *popt)
            r2 = r2_score(probs, probs_pred)
            
            # Only draw the line if the fit is decent (R^2 > 0.5)
            if r2 > 0.5:
                plt.plot(steps, probs_pred, color=c, linestyle='--', linewidth=2, 
                         label=f'{model_name} Log Fit (R²={r2:.2f})')
            else:
                print(f"   ⚠️ Weak correlation for {model_name} (R²={r2:.2f}), skipping trendline.")
                
        except Exception as e:
            print(f"   ⚠️ Could not fit curve for {model_name}: {e}")

    # 5. Graph Styling
    plt.title("LLM Confidence Growth (True Token Probability)", fontsize=16)
    plt.xlabel("Gradient Updates (Training Steps)", fontsize=12)
    plt.ylabel("Average Confidence (Probability)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    
    # 6. Save Output (Server-friendly, no GUI required)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\n✅ Success! Graph saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_graph()