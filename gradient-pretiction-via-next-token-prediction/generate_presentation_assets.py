import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# הגדרות
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(CURRENT_DIR, "forensic_merged_features.csv")
ASSETS_DIR = os.path.join(CURRENT_DIR, "presentation_assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

def save_feature_slide(df, feature_col, title_name):
    """יוצר תמונה עם 3 גרפים (אחד לכל מודל) עבור פיצ'ר ספציפי"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    model_sizes = ['7M', '30M', '124M']
    
    for i, size in enumerate(model_sizes):
        subset = df[df['Model_Size'] == size].sort_values('Step')
        if subset.empty: continue
        
        X = subset[[feature_col]]
        y = subset['Step']
        
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        ax = axes[i]
        ax.scatter(y, y_pred, color='blue', alpha=0.5)
        ax.plot(y, y, 'r--', alpha=0.8)
        ax.set_title(f"Model {size}\nR² = {r2:.3f}")
        ax.set_xlabel("Actual Updates")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f"Feature Analysis: {title_name}", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ASSETS_DIR, f"slide_{feature_col}.png"))
    plt.close()

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # 1. יצירת שקופית לכל פיצ'ר בנפרד (כמו Slide 2, 3, 5 שלהם)
    features_to_plot = {
        'Embedded_Norm': 'Embedded Norm (L2)',
        'Logit_Norm': 'Logit Norm (De-embedded)',
        'Weight_Variance': 'Weight Variance',
        'Next_Token_Prob': 'Next-Token Probability (Dynamic)'
    }
    
    for col, name in features_to_plot.items():
        print(f"Generating slide for {name}...")
        save_feature_slide(df, col, name)

    # 2. יצירת שקופית Combined (כמו Slide 6 שלהם)
    print("Generating Combined slide...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    full_features = ['Embedded_Norm', 'Logit_Norm', 'Weight_Variance', 'L1_Norm', 'Next_Token_Prob']
    
    for i, size in enumerate(['7M', '30M', '124M']):
        subset = df[df['Model_Size'] == size].sort_values('Step')
        X = subset[full_features]
        y = subset['Step']
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        ax = axes[i]
        ax.scatter(y, y_pred, color='purple', alpha=0.6)
        ax.plot(y, y, 'r--')
        ax.set_title(f"Combined {size}\nR² = {r2:.4f}")
        ax.grid(True, alpha=0.2)
        
    plt.suptitle("Multi-Feature Fusion Results", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ASSETS_DIR, "slide_combined.png"))
    plt.close()

    print(f"\n✅ All assets generated in: {ASSETS_DIR}")
    print("Download them using scp or FileZilla and drop them into your PPT.")

if __name__ == "__main__":
    main()