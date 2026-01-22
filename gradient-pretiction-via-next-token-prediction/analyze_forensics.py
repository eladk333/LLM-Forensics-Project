import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(CURRENT_DIR, "forensic_merged_features.csv")
OUTPUT_TABLE_IMG = os.path.join(CURRENT_DIR, "r2_comparison_table.png")
OUTPUT_COMBINED_IMG = os.path.join(CURRENT_DIR, "combined_features_analysis.png")

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Dataset not found at {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV)
    
    # 1. הגדרת הסטים של הפיצ'רים בדיוק כמו בשקופיות שלהם
    feature_configs = {
        'Embedded Norm': ['Embedded_Norm'],
        'Logit Norm': ['Logit_Norm'],
        'Weight Variance': ['Weight_Variance'],
        'Next-Token Prob (Your Feature)': ['Next_Token_Prob'],
        'Friends Combined (Static)': ['Embedded_Norm', 'Logit_Norm', 'Weight_Variance', 'L1_Norm'],
        'Full Fusion (Our Work)': ['Embedded_Norm', 'Logit_Norm', 'Weight_Variance', 'L1_Norm', 'Next_Token_Prob']
    }
    
    model_sizes = ['7M', '30M', '124M']
    results = []

    # חישוב R2 לכל קומבינציה (כמו בטבלה של Slide 4)
    for size in model_sizes:
        subset = df[df['Model_Size'] == size]
        if subset.empty: continue
        y = subset['Step']
        
        for name, cols in feature_configs.items():
            X = subset[cols]
            
            # Linear Regression
            lin = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
            r2_lin = r2_score(y, lin.predict(X))
            
            # MLP (Neural Network)
            mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50,50), max_iter=2000, random_state=42)).fit(X, y)
            r2_mlp = r2_score(y, mlp.predict(X))
            
            results.append({'Size': size, 'Method': name, 'Linear': r2_lin, 'MLP': r2_mlp})

    # --- יצירת הטבלה (כמו בשקופית 4) ---
    res_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 R^2 COMPARISON TABLE (Matches Friend's Format)")
    print("="*50)
    pivot = res_df.pivot(index='Method', columns='Size', values=['Linear', 'MLP'])
    print(pivot.round(4))

    # --- יצירת גרף Combined (כמו בשקופית 6) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, size in enumerate(model_sizes):
        subset = df[df['Model_Size'] == size].sort_values('Step')
        ax = axes[i]
        y_true = subset['Step']
        
        # חיזוי של ה-Full Fusion (השילוב של כולם)
        X_full = subset[feature_configs['Full Fusion (Our Work)']]
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X_full, y_true)
        y_pred = model.predict(X_full)
        
        ax.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Predicted vs Actual')
        ax.plot(y_true, y_true, 'r--', label='Perfect Match')
        
        ax.set_title(f"Combined Features: {size} Model\n(Fusion R² = {r2_score(y_true, y_pred):.3f})", fontsize=14)
        ax.set_xlabel("Actual Gradient Updates")
        ax.set_ylabel("Predicted Updates")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_COMBINED_IMG)
    print(f"\n✅ Combined analysis graph saved to: {OUTPUT_COMBINED_IMG}")

if __name__ == "__main__":
    main()