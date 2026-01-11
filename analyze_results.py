import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_SIZE = '7M'
BASE_FOLDER = '/content/drive/MyDrive/llm'

FREQ_FILE = os.path.join(BASE_FOLDER, 'wiki_token_frequencies.csv')
FEATURE_FILE = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'final_frequency_dataset.csv')

# ==========================================
# 2. LOAD & MERGE DATA
# ==========================================
print(f"📂 Loading Data...")
if not os.path.exists(FREQ_FILE) or not os.path.exists(FEATURE_FILE):
    print("❌ Error: Files not found.")
    exit()

df_freq = pd.read_csv(FREQ_FILE)
df_feat = pd.read_csv(FEATURE_FILE)
df = pd.merge(df_freq, df_feat, on='token_id', how='inner')

print(f"📊 Dataset size: {len(df)}")

# ==========================================
# 3. PREPARE DATA
# ==========================================
X = df[['embedding_norm']]
y = df['log_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TRAIN MODELS
# ==========================================
print("-" * 50)

# --- A. Linear Regression ---
print("📈 Training Linear Regression...")
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# --- B. Neural Network (MLP) ---
print("🤖 Training Neural Network (MLP)...")
mlp_model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(100, 100),
        activation='tanh',
        solver='adam',
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )
)
mlp_model.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE
# ==========================================
# Predictions
lin_preds = lin_model.predict(X_test)
mlp_preds = mlp_model.predict(X_test)

# Scores
lin_r2 = r2_score(y_test, lin_preds)
mlp_r2 = r2_score(y_test, mlp_preds)

print("\n" + "="*30)
print("📊 RESULTS COMPARISON")
print("="*30)
print(f"1. Linear Regression R²: {lin_r2:.4f}")
print(f"2. Neural Network R²:    {mlp_r2:.4f}")

# ==========================================
# 6. VISUALIZE (Side-by-Side)
# ==========================================
plt.figure(figsize=(14, 6)) # Wider figure for two plots 

# --- Plot 1: Linear Regression ---
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=lin_preds, alpha=0.3, color='blue')
# Perfect prediction line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Log Frequency")
plt.ylabel("Predicted")
plt.title(f"Linear Regression\nR²: {lin_r2:.3f}")
plt.grid(True)
plt.legend()

# --- Plot 2: Neural Network ---
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=mlp_preds, alpha=0.3, color='green')
# Perfect prediction line
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Perfect Prediction")
plt.xlabel("Actual Log Frequency")
plt.ylabel("Predicted")
plt.title(f"Neural Network (MLP)\nR²: {mlp_r2:.3f}")
plt.grid(True)
plt.legend()

plt.suptitle(f"Model Comparison: {MODEL_SIZE} Checkpoint", fontsize=16)
plt.tight_layout()
plt.show()

# ==========================================
# 7. VISUALIZE THE LEARNED CURVE (Norm vs Freq)
# ==========================================
plt.figure(figsize=(10, 6))

# 1. Scatter the real data (Background)
plt.scatter(X_test, y_test, color='gray', alpha=0.1, label='Actual Data')

# 2. Sort X values to draw a clean line
# We need to sort the inputs so the line connects smoothly from left to right
sorted_indices = X_test.squeeze().argsort()
X_sorted = X_test.iloc[sorted_indices]
pred_sorted = mlp_preds[sorted_indices]

# 3. Plot the Neural Net's learned path
plt.plot(X_sorted, pred_sorted, color='red', lw=3, label='Neural Net Learned Curve')

plt.xlabel("Embedding Norm (Input)")
plt.ylabel("Log Frequency (Output)")
plt.title("The S-Shape: What the Neural Network Actually Learned")
plt.legend()
plt.grid(True)
plt.show()