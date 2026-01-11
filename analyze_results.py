import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
MODEL_SIZE = '7M'
BASE_FOLDER = 'G:/My Drive/llm'
FREQ_FILE = os.path.join(BASE_FOLDER, 'wiki_token_frequencies.csv')
FEATURE_FILE = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'norm_comparison_dataset.csv')

print(f"📂 Loading Data...")
if not os.path.exists(FREQ_FILE) or not os.path.exists(FEATURE_FILE):
    # Fallback for demo
    print("⚠️ Files not found. Generating Dummy Data...")
    df = pd.DataFrame({
        'embedding_norm': np.random.uniform(5, 15, 500),
        'logit_norm': np.random.uniform(5, 15, 500),
        'log_count': np.random.uniform(2, 10, 500)
    })
    df['log_count'] += 0.5 * df['embedding_norm']
else:
    df_freq = pd.read_csv(FREQ_FILE)
    df_feat = pd.read_csv(FEATURE_FILE)
    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')

y = df['log_count']
features = ['embedding_norm', 'logit_norm']

# ==========================================
# 2. PRE-CALCULATE MODELS
# ==========================================
print("⚙️  Training Models...")
storage = {}

# --- Train Individual Models (Slide 1 & 2) ---
for feat in features:
    X = df[[feat]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simple models for the single-feature slides
    lin = LinearRegression().fit(X_train, y_train)
    mlp = make_pipeline(StandardScaler(), MLPRegressor((100, 50), activation='tanh', max_iter=1000, random_state=42))
    mlp.fit(X_train, y_train)
    
    # Store sorted data for clean lines
    sorted_idx = X_test[feat].argsort()
    storage[feat] = {
        'X_test': X_test.iloc[sorted_idx],
        'y_test': y_test.iloc[sorted_idx],
        'lin_pred': lin.predict(X_test.iloc[sorted_idx]),
        'mlp_pred': mlp.predict(X_test.iloc[sorted_idx]),
        'lin_r2': r2_score(y_test, lin.predict(X_test)),
        'mlp_r2': r2_score(y_test, mlp.predict(X_test))
    }

# --- Train Combined Model (Slide 3) ---
# We use BOTH features to train, but we will plot against them individually
X_comb = df[features]
X_train, X_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.2, random_state=42)

mlp_comb = make_pipeline(StandardScaler(), MLPRegressor((100, 50), activation='tanh', max_iter=1000, random_state=42))
mlp_comb.fit(X_train, y_train)
comb_preds = mlp_comb.predict(X_test)
comb_r2 = r2_score(y_test, comb_preds)

storage['combined'] = {
    'X_test': X_test, 
    'y_test': y_test, 
    'preds': comb_preds, # Predictions from the COMBINED model
    'r2': comb_r2
}

print("✅ Training Complete. Launching Slideshow...")

# ==========================================
# 3. SLIDESHOW VISUALIZATION LOGIC
# ==========================================
class SlideshowViewer:
    def __init__(self):
        self.ind = 0
        self.fig = plt.figure(figsize=(14, 8))
        self.plot_functions = [self.plot_feat1, self.plot_feat2, self.plot_combined_projected]
        self.update() 

    def plot_feat1(self):
        self._plot_single_feature('embedding_norm', 1)

    def plot_feat2(self):
        self._plot_single_feature('logit_norm', 2)

    def _plot_single_feature(self, feat, slide_num):
        ax = self.fig.add_subplot(111)
        data = storage[feat]
        
        # Plot Actual Data (Grey Cloud)
        sns.scatterplot(x=data['X_test'][feat], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual Data')
        
        # Plot Single-Feature Models (Lines)
        ax.plot(data['X_test'][feat], data['lin_pred'], color='blue', lw=2, linestyle='--', label=f"Linear (R2={data['lin_r2']:.3f})")
        ax.plot(data['X_test'][feat], data['mlp_pred'], color='red', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
        
        ax.set_title(f"Slide {slide_num}: Prediction using ONLY {feat}", fontsize=16)
        ax.set_xlabel(feat)
        ax.set_ylabel("Log Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_combined_projected(self):
        # Create 2 subplots side-by-side
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        data = storage['combined']
        X = data['X_test']
        y = data['y_test']
        preds = data['preds']
        
        # --- LEFT PLOT: Combined Model vs Embedding Norm ---
        # 1. Actual Data
        sns.scatterplot(x=X['embedding_norm'], y=y, ax=ax1, alpha=0.2, color='gray', label='Actual Data')
        # 2. Combined Model Predictions (scattered because they depend on the OTHER feature too)
        sns.scatterplot(x=X['embedding_norm'], y=preds, ax=ax1, alpha=0.6, color='red', s=15, label='Combined Model Preds')
        
        ax1.set_title("View 1: Projected on Embedding Norm", fontsize=12)
        ax1.set_xlabel("Embedding Norm")
        ax1.set_ylabel("Log Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- RIGHT PLOT: Combined Model vs Logit Norm ---
        # 1. Actual Data
        sns.scatterplot(x=X['logit_norm'], y=y, ax=ax2, alpha=0.2, color='gray', label='Actual Data')
        # 2. Combined Model Predictions
        sns.scatterplot(x=X['logit_norm'], y=preds, ax=ax2, alpha=0.6, color='red', s=15, label='Combined Model Preds')
        
        ax2.set_title("View 2: Projected on Logit Norm", fontsize=12)
        ax2.set_xlabel("Logit Norm")
        ax2.set_ylabel("Log Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Slide 3: Combined Model Predictions (R2: {data['r2']:.3f})\n(Note: Red dots form a cloud because the model uses BOTH features)", fontsize=16)

    def update(self):
        self.fig.clf() 
        self.plot_functions[self.ind]()
        
        # Re-add buttons
        plt.subplots_adjust(bottom=0.2)
        ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        
        self.bprev = Button(ax_prev, 'Previous')
        self.bprev.on_clicked(self.prev)
        self.bnext = Button(ax_next, 'Next')
        self.bnext.on_clicked(self.next)
        
        plt.draw()

    def next(self, event):
        self.ind = (self.ind + 1) % len(self.plot_functions)
        self.update()

    def prev(self, event):
        self.ind = (self.ind - 1) % len(self.plot_functions)
        self.update()

viewer = SlideshowViewer()
plt.show()