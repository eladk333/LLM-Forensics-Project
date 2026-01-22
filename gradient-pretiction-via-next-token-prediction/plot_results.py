import pandas as pd
import matplotlib.pyplot as plt
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
# HELPER FUNCTION
# ==========================================
def log_func(x, a, b): 
    """ Logarithmic function for curve fitting: y = a + b * ln(x) """
    return a + b * np.log(x)

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
    
    # Define colors for consistency
    colors = {
        'Model 124M': 'blue', 
        'Model 30M': 'green', 
        'Model 7M': 'red'
    }
    
    models = df['Model'].unique()
    print(f"📊 Found data for models: {models}")

    # 3. Iterate over each model to create SEPARATE graphs
    for model_name in models:
        subset = df[df['Model'] == model_name]
        
        if subset.empty:
            continue

        print(f"🎨 Generating graph for: {model_name}...")

        steps = subset["Step"].values
        probs = subset["Avg_Probability"].values
        
        # Get color (default to black if unknown)
        c = colors.get(model_name, 'black')
        
        # --- Start New Figure ---
        plt.figure(figsize=(10, 6))
        
        # A. Plot Raw Data Points
        plt.scatter(steps, probs, color=c, alpha=0.6, label='Observed Data')
        
        # B. Mark the "Full Epoch" (Last Point)
        last_step = steps[-1]
        last_prob = probs[-1]
        
        # Add annotation with arrow
        plt.annotate('Full Epoch', 
                     xy=(last_step, last_prob), 
                     xytext=(-60, 30),            
                     textcoords='offset points',  
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                     fontsize=10, fontweight='bold', color='black',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        # C. Regression: Fit Logarithmic Curve
        try:
            popt, _ = curve_fit(log_func, steps, probs)
            probs_pred = log_func(steps, *popt)
            r2 = r2_score(probs, probs_pred)
            
            # Create a label with the R^2 score
            label_fit = f'Logarithmic Fit (R²={r2:.3f})'
            
            # Plot the trendline
            plt.plot(steps, probs_pred, color='black', linestyle='--', linewidth=2, label=label_fit)
            
            # Optional: Add the equation text to the plot
            equation_text = f'y = {popt[0]:.2f} + {popt[1]:.4f}ln(x)'
            plt.title(f"{model_name}\nEquation: {equation_text}", fontsize=14)
            
        except Exception as e:
            print(f"   ⚠️ Could not fit curve for {model_name}: {e}")
            plt.title(f"{model_name} (Raw Data)", fontsize=14)

        # Graph Styling
        plt.xlabel("Gradient Updates (Training Steps)", fontsize=12)
        plt.ylabel("Next Token Probability", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # Save Output (Unique filename per model)
        safe_filename = f"graph_{model_name.replace(' ', '_')}.png"
        output_path = os.path.join(BASE_PATH, safe_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() # Close the figure to free memory
        
        print(f"   ✅ Saved: {output_path}")

    print("\n🎉 All graphs generated successfully.")

if __name__ == "__main__":
    plot_graph()