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
MODEL_SIZE = '30M'
BASE_MODELS_FOLDER = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\models'
FREQ_FILE = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\datasets\wiki\wiki_token_frequencies.csv'

# Path to the specific feature file
FEATURE_FILE = os.path.join(BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'final_frequency_dataset.csv')

print(f"📂 Loading Data...")
if not os.path.exists(FREQ_FILE) or not os.path.exists(FEATURE_FILE):
    print("⚠️ Files not found. Generating Dummy Data...")
    # Dummy data for demonstration
    df = pd.DataFrame({
        'embedding_norm': np.random.uniform(5, 15, 500),
        'logit_norm': np.random.uniform(5, 15, 500),
        'log_count': np.random.uniform(2, 10, 500)
    })
    # Add some correlation so models work
    df['log_count'] += 0.6 * df['embedding_norm']
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

def train_and_store(label, X_data, y_data):
    """Helper to train both Linear and MLP models and store results."""
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    # 1. Linear Regression
    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)

    # 2. MLP Regressor
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42))
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred)

    # Sort for clean line plotting (only matters for single feature)
    if X_test.shape[1] == 1:
        sort_idx = X_test.iloc[:, 0].argsort()
        X_test_sorted = X_test.iloc[sort_idx]
        y_test_sorted = y_test.iloc[sort_idx]
        lin_pred_sorted = lin_pred[sort_idx]
        mlp_pred_sorted = mlp_pred[sort_idx]
    else:
        # No sorting for combined features (multi-dimensional)
        X_test_sorted = X_test
        y_test_sorted = y_test
        lin_pred_sorted = lin_pred
        mlp_pred_sorted = mlp_pred

    return {
        'X_test': X_test_sorted,
        'y_test': y_test_sorted, # True values (Ground Truth)
        
        # Predictions (Sorted for line plots, unsorted for scatter)
        'lin_pred': lin_pred_sorted, 
        'mlp_pred': mlp_pred_sorted,
        
        # Raw Predictions (Unsorted - needed for Pred vs Actual scatter)
        'lin_pred_raw': lin_pred,
        'mlp_pred_raw': mlp_pred,
        'y_test_raw': y_test,
        
        'lin_r2': lin_r2,
        'mlp_r2': mlp_r2
    }

# --- Train Individual Feature Models ---
for feat in features:
    storage[feat] = train_and_store(feat, df[[feat]], y)

# --- Train Combined Model ---
storage['combined'] = train_and_store('combined', df[features], y)

print("✅ Training Complete. Launching Viewer...")

# ==========================================
# 3. MENU VISUALIZATION LOGIC
# ==========================================
class SlideshowViewer:
    def __init__(self):
        # Menu Options
        self.options = [
            'Embedding: Data Spread',
            'Logit: Data Spread',
            'Combined: Data Spread',
            '-----------------------', # Separator (non-selectable logic handled below)
            'Embedding: Pred vs Actual',
            'Logit: Pred vs Actual',
            'Combined: Pred vs Actual'
        ]
        self.current_selection = 'Embedding: Data Spread'
        
        self.fig = plt.figure(figsize=(15, 8))
        self.update()

    # --- PLOT TYPE 1: DATA SPREAD (Original View) ---
    def _plot_data_spread(self, key, title_label):
        data = storage[key]
        
        if key == 'combined':
            # Special case for combined data spread (Two projection subplots)
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            # Left: Embedding Projection
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['y_test'], ax=ax1, alpha=0.2, color='gray', label='Actual Data')
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['mlp_pred'], ax=ax1, alpha=0.6, color='red', s=15, label='MLP Preds')
            ax1.set_title(f"Projected on Embedding Norm", fontsize=12)
            ax1.set_xlabel("Embedding Norm")
            ax1.set_ylabel("Log Frequency")
            
            # Right: Logit Projection
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['y_test'], ax=ax2, alpha=0.2, color='gray', label='Actual Data')
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['mlp_pred'], ax=ax2, alpha=0.6, color='red', s=15, label='MLP Preds')
            ax2.set_title(f"Projected on Logit Norm", fontsize=12)
            ax2.set_xlabel("Logit Norm")
            
            plt.suptitle(f"Data Spread: Combined Model ({MODEL_SIZE}) - MLP R2: {data['mlp_r2']:.3f}", fontsize=16)
        else:
            # Single Feature View
            ax = self.fig.add_subplot(111)
            # Scatter Actuals
            sns.scatterplot(x=data['X_test'][key], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual Data')
            # Plot Model Lines
            ax.plot(data['X_test'][key], data['lin_pred'], color='blue', lw=2, linestyle='--', label=f"Linear (R2={data['lin_r2']:.3f})")
            ax.plot(data['X_test'][key], data['mlp_pred'], color='red', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
            
            ax.set_title(f"Data Spread: {key} vs Log Frequency ({MODEL_SIZE})", fontsize=16)
            ax.set_xlabel(key)
            ax.set_ylabel("Log Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # --- PLOT TYPE 2: ACTUAL VS PREDICTED (Square Aspect Ratio) ---
    def _plot_actual_vs_predicted(self, key, title_label):
        data = storage[key]
        
        y_true = data['y_test_raw']
        lin_pred = data['lin_pred_raw']
        mlp_pred = data['mlp_pred_raw']
        
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        def plot_diagonal(ax, y_t, y_p):
            min_val = min(y_t.min(), y_p.min())
            max_val = max(y_t.max(), y_p.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Fit')

        # 1. Linear Regression Plot
        sns.scatterplot(x=y_true, y=lin_pred, ax=ax1, alpha=0.3, color='blue', edgecolor='w', s=50)
        plot_diagonal(ax1, y_true, lin_pred)
        ax1.set_title(f"Linear Regression\n(R2: {data['lin_r2']:.3f})", fontsize=14)
        ax1.set_xlabel("Actual Log Frequency")
        ax1.set_ylabel("Predicted Log Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_box_aspect(1) # <--- THIS MAKES IT SQUARE

        # 2. MLP Plot
        sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.3, color='green', edgecolor='w', s=50)
        plot_diagonal(ax2, y_true, mlp_pred)
        ax2.set_title(f"Neural Network (MLP)\n(R2: {data['mlp_r2']:.3f})", fontsize=14)
        ax2.set_xlabel("Actual Log Frequency")
        ax2.set_ylabel("Predicted Log Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_box_aspect(1) # <--- THIS MAKES IT SQUARE

        plt.suptitle(f"Prediction Accuracy: {title_label}", fontsize=16)

        
    def update(self):
        self.fig.clf() 
        plt.subplots_adjust(left=0.25) # Space for menu
        
        sel = self.current_selection
        
        # Logic to route the selection to the correct plotter
        if 'Data Spread' in sel:
            if 'Embedding' in sel: self._plot_data_spread('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_data_spread('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_data_spread('combined', 'Combined')
            
        elif 'Pred vs Actual' in sel:
            if 'Embedding' in sel: self._plot_actual_vs_predicted('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_actual_vs_predicted('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_actual_vs_predicted('combined', 'Combined Features')

        # Draw Menu
        ax_menu = plt.axes([0.02, 0.4, 0.18, 0.35], facecolor='#f0f0f0')
        ax_menu.set_title("Analysis Mode", fontsize=12, pad=10, weight='bold')
        
        # Handle index for separator line (skip it if clicked)
        try:
            active_idx = self.options.index(self.current_selection)
        except ValueError:
            active_idx = 0
            
        self.radio = RadioButtons(ax_menu, self.options, active=active_idx)
        self.radio.on_clicked(self.on_click)
        
        plt.draw()

    def on_click(self, label):
        if '---' in label: return # Ignore separator clicks
        self.current_selection = label
        self.update()

viewer = SlideshowViewer()
plt.show()