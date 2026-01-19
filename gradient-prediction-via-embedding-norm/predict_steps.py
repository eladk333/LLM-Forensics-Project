import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
MY_DRIVE_BASE = r"G:/My Drive"
PROJECT_FOLDER = "LLM_erez_property"

INPUT_CSV = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "gradient_norm_results.csv")
OUTPUT_IMG_DIR = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "plots")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: {INPUT_CSV} not found. Please run 'extract_gradient_norms.py' first.")
    exit()

print(f"📂 Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
model_names = df['Model'].unique()

# ==========================================
# 2. GENERATE COMBINED GRAPH (ALL MODELS)
# ==========================================
print("\n📊 Generating Combined Correlation Graph...")
plt.figure(figsize=(12, 8))

for model_name in model_names:
    subset = df[df['Model'] == model_name].sort_values('Step')
    if not subset.empty:
        plt.plot(subset['Step'], subset['Embedding_Norm'], marker='o', linestyle='-', label=model_name, markersize=4)

plt.title("Correlation: Training Steps vs. Embedding Norm (All Models)")
plt.xlabel("Gradient Updates (Training Steps)")
plt.ylabel("Embedding Matrix Norm (L2)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

combined_plot_path = os.path.join(OUTPUT_IMG_DIR, "combined_correlation_plot.png")
plt.savefig(combined_plot_path)
plt.close()
print(f"   📷 Combined plot saved to: {combined_plot_path}")

# ==========================================
# 3. TRAIN PREDICTORS & PLOT INDIVIDUAL GRAPHS
# ==========================================
print("\n🤖 Training Prediction Models & Generating Individual Plots...")
print("=" * 60)

for model_name in model_names:
    print(f"\n📌 Analyzing: {model_name}")

    # Filter data
    model_data = df[df['Model'] == model_name].sort_values('Embedding_Norm')
    
    X = model_data[['Embedding_Norm']].values
    y = model_data['Step'].values

    if len(X) < 5:
        print(f"   ⚠️ Not enough data points ({len(X)}). Skipping.")
        continue

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(X_train_poly, y_train)

    # Predict
    y_pred = regressor.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   ✅ R² Score: {r2:.5f}")
    print(f"   📉 MAE: +/- {mae:.1f} Steps")

    # --- INDIVIDUAL PLOT ---
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
    
    # Smooth line
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = regressor.predict(X_range_poly)
    
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Prediction (Poly Deg 2)')
    
    plt.title(f"Prediction Model: {model_name}\nR2: {r2:.4f}, MAE: {mae:.1f}")
    plt.xlabel("Embedding Norm (L2)")
    plt.ylabel("Training Steps (Gradient Updates)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    safe_name = model_name.replace(" ", "_")
    save_path = os.path.join(OUTPUT_IMG_DIR, f"prediction_{safe_name}.png")
    plt.savefig(save_path)
    plt.close() 
    print(f"   📷 Individual plot saved to: {save_path}")

print(f"\n✅ DONE! All graphs saved in: {OUTPUT_IMG_DIR}")