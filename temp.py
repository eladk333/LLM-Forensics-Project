import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import os

# ==========================================
# 0. PUBLICATION STYLE SETTINGS
# ==========================================
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['figure.dpi'] = 140
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
MODEL_SIZE = '7M'
BASE_FOLDER = 'G:/My Drive/llm'
FREQ_FILE = os.path.join(BASE_FOLDER, 'wiki_token_frequencies.csv')
FEATURE_FILE = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}', 'norm_comparison_dataset.csv')

print(f"📂 Loading Data...")
if not os.path.exists(FREQ_FILE) or not os.path.exists(FEATURE_FILE):
    print("⚠️ Files not found. Generating Dummy Data...")
    df = pd.DataFrame({
        'embedding_norm': np.random.uniform(5, 15, 500),
        'logit_norm': np.random.uniform(5, 15, 500),
        'log_count': np.random.uniform(2, 10, 500)
    })
    df['log_count'] += 0.6 * df['embedding_norm']
else:
    df_freq = pd.read_csv(FREQ_FILE)
    df_feat = pd.read_csv(FEATURE_FILE)
    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')

y = df['log_count']
features = ['embedding_norm', 'logit_norm']

# ==========================================
# 2. PRE-CALCULATE MODELS (NON-RANDOM SPLIT)
# ==========================================
print("⚙️  Training Models (Split: Train on Low Freq, Test on High Freq)...")
storage = {}

def train_and_store(label, X_data, y_data):
    # --- STEP 1: COMBINE AND SORT BY FREQUENCY ---
    # We combine X and y temporarily to ensure they stay aligned when we sort
    combined_df = X_data.copy()
    combined_df['target_y'] = y_data
    
    # Sort: Low Frequency (top) -> High Frequency (bottom)
    combined_df = combined_df.sort_values(by='target_y', ascending=True)
    
    # --- STEP 2: MANUAL SPLIT (80/20) ---
    split_index = int(len(combined_df) * 0.8)
    
    # Train = The first 80% (Lowest Frequencies)
    train_data = combined_df.iloc[:split_index]
    
    # Test = The last 20% (Highest Frequencies)
    test_data = combined_df.iloc[split_index:]
    
    # Separate X and y back out
    X_train = train_data.drop('target_y', axis=1)
    y_train = train_data['target_y']
    X_test = test_data.drop('target_y', axis=1)
    y_test = test_data['target_y']
    
    # --- STEP 3: TRAIN MODELS ---
    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)

    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42))
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred)

    # Sorting for clean plotting lines (X-axis sort)
    if X_test.shape[1] == 1:
        sort_idx = X_test.iloc[:, 0].argsort()
        X_test_sorted = X_test.iloc[sort_idx]
        y_test_sorted = y_test.iloc[sort_idx]
        lin_pred_sorted = lin_pred[sort_idx]
        mlp_pred_sorted = mlp_pred[sort_idx]
    else:
        X_test_sorted = X_test
        y_test_sorted = y_test
        lin_pred_sorted = lin_pred
        mlp_pred_sorted = mlp_pred

    return {
        'X_test': X_test_sorted,
        'y_test': y_test_sorted,
        'lin_pred': lin_pred_sorted, 
        'mlp_pred': mlp_pred_sorted,
        'lin_pred_raw': lin_pred,
        'mlp_pred_raw': mlp_pred,
        'y_test_raw': y_test,
        'lin_r2': lin_r2,
        'mlp_r2': mlp_r2
    }

for feat in features:
    storage[feat] = train_and_store(feat, df[[feat]], y)

storage['combined'] = train_and_store('combined', df[features], y)

print("✅ Training Complete. Launching Viewer...")

# ==========================================
# 3. MENU VISUALIZATION LOGIC
# ==========================================
class SlideshowViewer:
    def __init__(self):
        self.options = [
            'Embedding: Data Spread',
            'Logit: Data Spread',
            'Combined: Data Spread',
            '-----------------------',
            'Embedding: Pred vs Actual',
            'Logit: Pred vs Actual',
            'Combined: Pred vs Actual'
        ]
        self.current_selection = 'Embedding: Data Spread'
        self.fig = plt.figure(figsize=(15, 8))
        self.update()

    def _plot_data_spread(self, key, title_label):
        data = storage[key]
        if key == 'combined':
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['y_test'], ax=ax1, alpha=0.4, color='gray', label='Test Data (High Freq)', edgecolor=None)
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['mlp_pred'], ax=ax1, alpha=0.6, color='#d62728', s=15, label='MLP Preds', edgecolor=None)
            ax1.set_title(f"Projected on Embedding Norm", fontsize=12, fontweight='bold')
            ax1.set_xlabel("Embedding Norm")
            ax1.set_ylabel("Log Frequency")
            
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['y_test'], ax=ax2, alpha=0.4, color='gray', label='Test Data (High Freq)', edgecolor=None)
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['mlp_pred'], ax=ax2, alpha=0.6, color='#d62728', s=15, label='MLP Preds', edgecolor=None)
            ax2.set_title(f"Projected on Logit Norm", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Logit Norm")
            
            plt.suptitle(f"Data Spread: Combined Model ({MODEL_SIZE}) - Test on Highest 20%", fontsize=16, fontweight='bold')
        else:
            ax = self.fig.add_subplot(111)
            sns.scatterplot(x=data['X_test'][key], y=data['y_test'], ax=ax, alpha=0.4, color='gray', label='Test Data (High Freq)', edgecolor=None)
            ax.plot(data['X_test'][key], data['lin_pred'], color='#1f77b4', lw=2.5, linestyle='--', label=f"Linear ($R^2$={data['lin_r2']:.3f})")
            ax.plot(data['X_test'][key], data['mlp_pred'], color='#d62728', lw=3, label=f"MLP ($R^2$={data['mlp_r2']:.3f})")
            ax.set_title(f"Data Spread: {key} vs Log Frequency (Test on Top 20%)", fontsize=16, fontweight='bold')
            ax.set_xlabel(key, fontsize=12)
            ax.set_ylabel("Log Frequency", fontsize=12)
            ax.legend(frameon=True, fancybox=False, edgecolor='black')

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
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Identity Line')

        sns.scatterplot(x=y_true, y=lin_pred, ax=ax1, alpha=0.4, color='#1f77b4', edgecolor='w', s=60)
        plot_diagonal(ax1, y_true, lin_pred)
        ax1.set_title(f"Linear Regression\n($R^2$: {data['lin_r2']:.3f})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Observed Log Frequency", fontsize=12)
        ax1.set_ylabel("Predicted Log Frequency", fontsize=12)
        ax1.set_box_aspect(1)

        sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.4, color='#2ca02c', edgecolor='w', s=60)
        plot_diagonal(ax2, y_true, mlp_pred)
        ax2.set_title(f"Neural Network (MLP)\n($R^2$: {data['mlp_r2']:.3f})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Observed Log Frequency", fontsize=12)
        ax2.set_ylabel("Predicted Log Frequency", fontsize=12)
        ax2.set_box_aspect(1)

        plt.suptitle(f"Model Performance: {title_label} (Extrapolation Test)", fontsize=16, fontweight='bold')

    def update(self):
        self.fig.clf() 
        plt.subplots_adjust(left=0.25, right=0.95, top=0.85, bottom=0.15) 
        sel = self.current_selection
        if 'Data Spread' in sel:
            if 'Embedding' in sel: self._plot_data_spread('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_data_spread('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_data_spread('combined', 'Combined')
        elif 'Pred vs Actual' in sel:
            if 'Embedding' in sel: self._plot_actual_vs_predicted('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_actual_vs_predicted('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_actual_vs_predicted('combined', 'Combined Features')

        ax_menu = plt.axes([0.02, 0.4, 0.18, 0.35], facecolor='#f8f9fa')
        ax_menu.set_title("Analysis Mode", fontsize=12, pad=10, fontweight='bold', color='#333333')
        try:
            active_idx = self.options.index(self.current_selection)
        except ValueError:
            active_idx = 0
        self.radio = RadioButtons(ax_menu, self.options, active=active_idx, activecolor='#1f77b4')
        self.radio.on_clicked(self.on_click)
        plt.draw()

    def on_click(self, label):
        if '---' in label: return 
        self.current_selection = label
        self.update()

viewer = SlideshowViewer()
plt.show()