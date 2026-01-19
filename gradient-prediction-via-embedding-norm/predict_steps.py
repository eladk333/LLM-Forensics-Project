import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MY_DRIVE_BASE = r"G:/My Drive"
PROJECT_FOLDER = "LLM_erez_property"
INPUT_CSV = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "gradient_norm_results.csv")
OUTPUT_IMG_DIR = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "plots")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

if not os.path.exists(INPUT_CSV):
    print(f"❌ Error: {INPUT_CSV} not found. Run extract_gradient_norms.py first.")
    exit()

print(f"📂 Loading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)
model_names = df['Model'].unique()

# ==========================================
# 2. GENERATE COMBINED GRAPH (X=Norm, Y=Steps)
# ==========================================
print("\n📊 Generating Combined Graph...")
plt.figure(figsize=(12, 8))

for model_name in model_names:
    subset = df[df['Model'] == model_name].sort_values('Embedding_Norm')
    if not subset.empty:
        plt.plot(subset['Embedding_Norm'], subset['Step'], marker='o', linestyle='-', label=model_name, markersize=4)

plt.title("The Forensic Tool: Predicting Model Age from Embedding Norm")
plt.xlabel("Embedding Matrix Norm (L2) [Input]")
plt.ylabel("Gradient Updates (Training Steps) [Target]")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig(os.path.join(OUTPUT_IMG_DIR, "combined_correlation_plot.png"))
plt.close()
print(f"   📷 Combined plot saved.")

# ==========================================
# 3. TRAIN PREDICTORS (Norm -> Steps)
# ==========================================
print("\n🤖 Training Forensic Models (Input: Norm -> Output: Steps)...")
print("=" * 60)

for model_name in model_names:
    print(f"\n📌 Analyzing: {model_name}")

    model_data = df[df['Model'] == model_name].sort_values('Embedding_Norm')
    
    # CORRECT LOGIC: Predict STEPS (y) from NORM (X)
    X = model_data[['Embedding_Norm']].values 
    y = model_data['Step'].values             

    if len(X) < 5:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(X_train_poly, y_train)

    y_pred = regressor.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   ✅ R² Score: {r2:.5f}")
    print(f"   📉 MAE: +/- {mae:.1f} Steps")

    # --- PLOT INDIVIDUAL PREDICTION ---
    plt.figure(figsize=(10, 6))
    
    plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
    
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = regressor.predict(X_range_poly)
    
    plt.plot(X_range, y_range_pred, color='red', linewidth=2, label='Prediction Model')
    
    # Annotate Full Epoch
    last_x = X.max()
    last_y = y[np.argmax(X)]
    plt.annotate('Full Epoch', xy=(last_x, last_y), xytext=(-20, 10), 
                 textcoords='offset points', ha='right', fontsize=10, fontweight='bold',
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

    stats_text = f"Forensic Accuracy:\n$R^2$ = {r2:.4f}\nMAE = {mae:.1f} Steps"
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f"Forensic Model: {model_name}")
    plt.xlabel("Embedding Norm (L2) [Input]")
    plt.ylabel("Predicted Steps [Output]")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(OUTPUT_IMG_DIR, f"prediction_{model_name.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close() 
    print(f"   📷 Plot saved: {save_path}")

print(f"\n✅ DONE! All graphs and stats are aligned (X=Norm, Y=Steps).")