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

# --- Train Individual Models ---
for feat in features:
    X = df[[feat]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lin = LinearRegression().fit(X_train, y_train)
    
    # Corrected MLPRegressor with hidden_layer_sizes
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42))
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

# --- Train Combined Model ---
X_comb = df[features]
X_train, X_test, y_train, y_test = train_test_split(X_comb, y, test_size=0.2, random_state=42)

# Corrected MLPRegressor with hidden_layer_sizes
mlp_comb = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42))
mlp_comb.fit(X_train, y_train)
comb_preds = mlp_comb.predict(X_test)
comb_r2 = r2_score(y_test, comb_preds)

storage['combined'] = {
    'X_test': X_test, 
    'y_test': y_test, 
    'preds': comb_preds,
    'r2': comb_r2
}

print("✅ Training Complete. Launching Viewer...")

# ==========================================
# 3. MENU VISUALIZATION LOGIC
# ==========================================
class SlideshowViewer:
    def __init__(self):
        # Initial selection
        self.current_selection = 'Embedding Norm'
        self.options = ['Embedding Norm', 'Logit Norm', 'Combined Model']
        
        self.fig = plt.figure(figsize=(14, 8))
        self.update()

    def plot_feat1(self):
        self._plot_single_feature('embedding_norm', 1)

    def plot_feat2(self):
        self._plot_single_feature('logit_norm', 2)

    def _plot_single_feature(self, feat, slide_num):
        ax = self.fig.add_subplot(111)
        data = storage[feat]
        
        sns.scatterplot(x=data['X_test'][feat], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual Data')
        ax.plot(data['X_test'][feat], data['lin_pred'], color='blue', lw=2, linestyle='--', label=f"Linear (R2={data['lin_r2']:.3f})")
        ax.plot(data['X_test'][feat], data['mlp_pred'], color='red', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
        
        ax.set_title(f"Data Spread: {feat} vs Log Frequency ({MODEL_SIZE})", fontsize=16)
        ax.set_xlabel(feat)
        ax.set_ylabel("Log Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_combined_projected(self):
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        data = storage['combined']
        X = data['X_test']
        y = data['y_test']
        preds = data['preds']
        
        # Plot 1: Embedding Norm
        sns.scatterplot(x=X['embedding_norm'], y=y, ax=ax1, alpha=0.2, color='gray', label='Actual Data')
        sns.scatterplot(x=X['embedding_norm'], y=preds, ax=ax1, alpha=0.6, color='red', s=15, label='Model Preds')
        ax1.set_title("Projection: Embedding Norm", fontsize=12)
        ax1.set_xlabel("Embedding Norm")
        ax1.set_ylabel("Log Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Logit Norm
        sns.scatterplot(x=X['logit_norm'], y=y, ax=ax2, alpha=0.2, color='gray', label='Actual Data')
        sns.scatterplot(x=X['logit_norm'], y=preds, ax=ax2, alpha=0.6, color='red', s=15, label='Model Preds')
        ax2.set_title("Projection: Logit Norm", fontsize=12)
        ax2.set_xlabel("Logit Norm")
        ax2.set_ylabel("Log Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Combined Model (Using Both Features) - R2: {data['r2']:.3f}", fontsize=16)

    def update(self):
        self.fig.clf() # Clear the window
        
        # Reserve space on the left for the menu
        plt.subplots_adjust(left=0.25)
        
        # Draw the main plot based on selection
        if self.current_selection == 'Embedding Norm':
            self.plot_feat1()
        elif self.current_selection == 'Logit Norm':
            self.plot_feat2()
        elif self.current_selection == 'Combined Model':
            self.plot_combined_projected()
            
        # Draw the Menu (Radio Buttons)
        # Position: [left, bottom, width, height]
        ax_menu = plt.axes([0.02, 0.4, 0.18, 0.25], facecolor='#f0f0f0')
        ax_menu.set_title("Select Picture", fontsize=10, pad=10)
        
        # Create Radio Button Widget
        # We need to find the index of the current selection to keep it highlighted
        active_idx = self.options.index(self.current_selection)
        self.radio = RadioButtons(ax_menu, self.options, active=active_idx)
        
        # Link the click event
        self.radio.on_clicked(self.on_click)
        
        plt.draw()

    def on_click(self, label):
        self.current_selection = label
        self.update()

viewer = SlideshowViewer()
plt.show()