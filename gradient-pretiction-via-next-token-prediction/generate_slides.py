import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

INPUT_CSV = "forensic_merged_features.csv"
SLIDES_DIR = "presentation_assets"
os.makedirs(SLIDES_DIR, exist_ok=True)

def create_triple_panel_plot(df, feature_col, title):
    """Creates a 3-panel slide (7M, 30M, 124M) matching the friends' style."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sizes = ['7M', '30M', '124M']
    
    for i, size in enumerate(sizes):
        subset = df[df['Model_Size'] == size].sort_values('Step')
        if subset.empty: continue
        
        X, y = subset[[feature_col]], subset['Step']
        model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
        r2 = r2_score(y, model.predict(X))
        
        axes[i].scatter(y, X, color='blue', alpha=0.6)
        axes[i].set_title(f"Model {size}\n$R^2 = {r2:.3f}$")
        axes[i].set_xlabel("Gradient Updates")
        axes[i].set_ylabel(title)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SLIDES_DIR, f"slide_{feature_col}.png"))
    plt.close()

def main():
    df = pd.read_csv(INPUT_CSV)
    
    # Feature mapping for slides
    features = {
        'Embedding_Norm': 'Embedded Norm (L2)',
        'Logit_Norm': 'Logit Norm',
        'Weight_Variance': 'Weight Variance',
        'Avg_Probability': 'Next-Token Probability (Dynamic)'
    }

    print("Generating feature slides...")
    for col, name in features.items():
        create_triple_panel_plot(df, col, name)

    # Combined Summary Table (Slide 4 style)
    table_data = []
    for size in ['7M', '30M', '124M']:
        subset = df[df['Model_Size'] == size]
        for col in features.keys():
            X, y = subset[[col]], subset['Step']
            # Linear and MLP R2
            lin_r2 = r2_score(y, make_pipeline(StandardScaler(), LinearRegression()).fit(X, y).predict(X))
            mlp_r2 = r2_score(y, make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000)).fit(X, y).predict(X))
            table_data.append({'Size': size, 'Feature': features[col], 'Linear': lin_r2, 'MLP': mlp_r2})

    print("\n--- R^2 Comparison Table ---")
    summary = pd.DataFrame(table_data).pivot(index='Feature', columns='Size', values=['Linear', 'MLP'])
    print(summary.round(4))

if __name__ == "__main__":
    main()