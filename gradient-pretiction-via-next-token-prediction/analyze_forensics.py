import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
import os

# CONFIG
BASE_PATH = os.getcwd()
INPUT_CSV = os.path.join(BASE_PATH, "forensic_merged_features.csv")

def main():
    if not os.path.exists(INPUT_CSV):
        print("❌ Dataset not found. Run extract_forensic_features.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Fix 'Final' steps (999999 -> Real Max + 500)
    for size in df['Model_Size'].unique():
        mask = (df['Model_Size'] == size)
        max_step = df.loc[mask & (df['Step'] < 999999), 'Step'].max()
        if pd.notna(max_step):
            df.loc[mask & (df['Step'] == 999999), 'Step'] = max_step + 500
    
    model_sizes = df['Model_Size'].unique()
    
    # Define Feature Sets for Comparison
    feature_sets = {
        'My Method (Static)': ['Embedding_Norm'],
        'My Method (Dynamic)': ['Avg_Next_Token_Prob'],
        'Friend\'s Method': ['Weight_Skew', 'Weight_Kurtosis', 'Weight_Variance'],
        'Combined (All)': ['Embedding_Norm', 'Avg_Next_Token_Prob', 'Weight_Skew']
    }
    
    # Setup Plot
    plt.figure(figsize=(10, 4 * len(model_sizes)))
    
    for i, size in enumerate(model_sizes):
        subset = df[df['Model_Size'] == size]
        if len(subset) < 5: continue
        
        ax = plt.subplot(len(model_sizes), 1, i+1)
        y_true = subset['Step']
        
        print(f"\n📊 Results for Model {size}:")
        
        # Plot Ground Truth reference
        ax.plot(y_true, y_true, 'k--', label='Ground Truth', alpha=0.3)
        
        colors = ['blue', 'green', 'orange', 'red']
        
        for (name, cols), color in zip(feature_sets.items(), colors):
            X = subset[cols]
            
            # Use Polynomial Regression (Degree 2) to allow for curves
            model = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
            model.fit(X, y_true)
            y_pred = model.predict(X)
            r2 = r2_score(y_true, y_pred)
            
            print(f"   > {name:<20} | R² = {r2:.4f}")
            
            # Plot (Sort for clean lines)
            sorted_idx = np.argsort(y_true)
            ax.plot(y_true.iloc[sorted_idx], y_pred[sorted_idx], 
                    label=f"{name} ($R^2$={r2:.3f})", color=color, linewidth=2, alpha=0.8)
            
        ax.set_title(f"Forensic Age Prediction ({size})")
        ax.set_xlabel("Actual Steps (Ground Truth)")
        ax.set_ylabel("Predicted Steps (Model Output)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    output_img = os.path.join(BASE_PATH, "final_forensic_comparison.png")
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"\n✅ Graph saved to '{output_img}'")

if __name__ == "__main__":
    main()