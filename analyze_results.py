import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
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
FEATURE_FILE = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'final_frequency_dataset.csv')

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
# 2. PRE-CALCULATE MODELS (80% Train / 20% Test)
# ==========================================
print("⚙️  Training Models (80% Train / 20% Test)...")
storage = {}

# --- A. Single Feature Models ---
for feat in features:
    X = df[[feat]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lin = LinearRegression().fit(X_train, y_train)
    mlp = make_pipeline(StandardScaler(), MLPRegressor((100, 50), activation='tanh', max_iter=1000, random_state=42))
    mlp.fit(X_train, y_train)
    
    # Sort for clean lines
    sorted_idx = X_test[feat].argsort()
    storage[feat] = {
        'X_test': X_test.iloc[sorted_idx],
        'y_test': y_test.iloc[sorted_idx],
        'lin_pred': lin.predict(X_test.iloc[sorted_idx]),
        'mlp_pred': mlp.predict(X_test.iloc[sorted_idx]),
        'lin_r2': r2_score(y_test, lin.predict(X_test)),
        'mlp_r2': r2_score(y_test, mlp.predict(X_test))
    }

# --- B. Combined Model (Trained on Both, Projected on One) ---
X_comb = df[features]
X_train, X_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.2, random_state=42)

mlp_comb = make_pipeline(StandardScaler(), MLPRegressor((100, 50), activation='tanh', max_iter=1000, random_state=42))
mlp_comb.fit(X_train, y_train)
comb_preds = mlp_comb.predict(X_test)
comb_r2 = r2_score(y_test, comb_preds)

storage['combined'] = {
    'X_test': X_test, 
    'y_test': y_test, 
    'preds': comb_preds,
    'r2': comb_r2
}

print("✅ Training Complete. Launching Menu...")

# ==========================================
# 3. MENU VISUALIZATION LOGIC
# ==========================================
class MenuViewer:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 8))
        
        # 1. Setup the Menu Area (Left Side)
        # [left, bottom, width, height]
        self.menu_ax = plt.axes([0.02, 0.6, 0.15, 0.25], facecolor='#f0f0f0')
        self.radio = RadioButtons(self.menu_ax, 
                                  ('Spread (Embedding)', 'Spread (Logit)', 'Combined Model'))
        
        # Style the menu slightly
        for circle in self.radio.circles:
            circle.set_radius(0.05)

        # 2. Map options to functions
        self.funcs = {
            'Spread (Embedding)': self.plot_feat1,
            'Spread (Logit)': self.plot_feat2,
            'Combined Model': self.plot_combined
        }

        # 3. Initial Draw
        self.radio.on_clicked(self.update_plot)
        self.update_plot('Spread (Embedding)')

    def update_plot(self, label):
        # Clear OLD plots, but keep the MENU (menu_ax)
        for ax in self.fig.axes:
            if ax != self.menu_ax:
                ax.remove()
        
        # Execute the correct plotting function
        plot_func = self.funcs[label]
        plot_func()
        
        # Refresh
        self.fig.canvas.draw_idle()

    def plot_feat1(self):
        # Create axes on the right side
        ax = self.fig.add_axes([0.25, 0.1, 0.70, 0.8]) 
        self._plot_single_feature(ax, 'embedding_norm', 
                                  title="Visualizing the Spread of the Data (Embedding Norm)")

    def plot_feat2(self):
        ax = self.fig.add_axes([0.25, 0.1, 0.70, 0.8])
        self._plot_single_feature(ax, 'logit_norm', 
                                  title="Visualizing the Spread of the Data (Logit Norm)")

    def _plot_single_feature(self, ax, feat, title):
        data = storage[feat]
        
        # Actual Data Cloud
        sns.scatterplot(x=data['X_test'][feat], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual Data')
        
        # Regressors
        ax.plot(data['X_test'][feat], data['lin_pred'], color='blue', lw=2, linestyle='--', label=f"Linear (R2={data['lin_r2']:.3f})")
        ax.plot(data['X_test'][feat], data['mlp_pred'], color='red', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(feat)
        ax.set_ylabel("Log Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_combined(self):
        # Create 2 subplots side-by-side on the right
        # add_axes([left, bottom, width, height])
        ax1 = self.fig.add_axes([0.25, 0.1, 0.33, 0.8])
        ax2 = self.fig.add_axes([0.62, 0.1, 0.33, 0.8])
        
        data = storage['combined']
        X = data['X_test']
        
        # View 1
        sns.scatterplot(x=X['embedding_norm'], y=data['y_test'], ax=ax1, alpha=0.2, color='gray')
        sns.scatterplot(x=X['embedding_norm'], y=data['preds'], ax=ax1, alpha=0.6, color='red', s=15, label='Combined Preds')
        ax1.set_title("Projected on Embedding Norm")
        ax1.set_xlabel("Embedding Norm")
        
        # View 2
        sns.scatterplot(x=X['logit_norm'], y=data['y_test'], ax=ax2, alpha=0.2, color='gray')
        sns.scatterplot(x=X['logit_norm'], y=data['preds'], ax=ax2, alpha=0.6, color='red', s=15, label='Combined Preds')
        ax2.set_title("Projected on Logit Norm")
        ax2.set_xlabel("Logit Norm")
        
        self.fig.text(0.5, 0.95, f"Combined Model Predictions (R2: {data['r2']:.3f})", ha='center', fontsize=16)

# Launch
viewer = MenuViewer()
plt.show()