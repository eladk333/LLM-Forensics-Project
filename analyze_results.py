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
MODEL_SIZES = ['7M', '30M', '124M'] # List of all models to process
BASE_MODELS_FOLDER = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\models'
FREQ_FILE = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\datasets\wiki\wiki_token_frequencies.csv'

# Dictionary to hold data for ALL models
# Structure: global_storage['7M']['embedding_norm']...
global_storage = {} 

print(f"📂 Loading Data from: {BASE_MODELS_FOLDER}")

# Load the common frequency file once
if os.path.exists(FREQ_FILE):
    df_freq = pd.read_csv(FREQ_FILE)
    print(f"✅ Frequency file loaded ({len(df_freq)} tokens)")
else:
    print(f"❌ Frequency file not found at {FREQ_FILE}")
    df_freq = None

# Iterate over all model sizes and train models for each
for size in MODEL_SIZES:
    print(f"\n--- Processing Model: {size} ---")
    
    feature_file = os.path.join(BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'final_frequency_dataset.csv')
    
    if not os.path.exists(feature_file):
        print(f"⚠️  Feature file not found for {size} (Skipping...)")
        continue
        
    if df_freq is None:
        print("Cannot process without frequency file.")
        continue

    # Load and Merge
    df_feat = pd.read_csv(feature_file)
    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    
    if df.empty:
        print(f"⚠️  Merged dataframe is empty for {size}!")
        continue

    y = df['log_count']
    features = ['embedding_norm', 'logit_norm']
    
    # Initialize storage for this specific model size
    global_storage[size] = {}

    # --- Helper Function for Training ---
    def train_and_store(X_data, y_data):
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

    # Train for this model size
    print(f"   Training models for {size}...")
    for feat in features:
        global_storage[size][feat] = train_and_store(df[[feat]], y)
    
    global_storage[size]['combined'] = train_and_store(df[features], y)
    print(f"   ✅ Done.")

if not global_storage:
    print("\n❌ No models were successfully loaded. Exiting.")
    exit()

print("\n✅ All Training Complete. Launching Viewer...")

# ==========================================
# 3. MENU VISUALIZATION LOGIC
# ==========================================
class SlideshowViewer:
    def __init__(self):
        self.available_models = list(global_storage.keys())
        self.current_model = self.available_models[0] # Default to first loaded model
        
        self.view_options = [
            'Embedding: Data Spread',
            'Logit: Data Spread',
            'Combined: Data Spread',
            '-----------------------',
            'Embedding: Pred vs Actual',
            'Logit: Pred vs Actual',
            'Combined: Pred vs Actual'
        ]
        self.current_view = 'Embedding: Data Spread'
        
        self.fig = plt.figure(figsize=(16, 9))
        self.update()

    def _plot_data_spread(self, key, title_label):
        data = global_storage[self.current_model][key]
        
        if key == 'combined':
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            # Left: Embedding Projection
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['y_test'], ax=ax1, alpha=0.2, color='gray')
            sns.scatterplot(x=data['X_test']['embedding_norm'], y=data['mlp_pred'], ax=ax1, alpha=0.6, color='red', s=15, label='MLP Preds')
            ax1.set_title(f"Projected on Embedding Norm", fontsize=12)
            ax1.set_xlabel("Embedding Norm")
            ax1.set_ylabel("Log Frequency")
            
            # Right: Logit Projection
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['y_test'], ax=ax2, alpha=0.2, color='gray')
            sns.scatterplot(x=data['X_test']['logit_norm'], y=data['mlp_pred'], ax=ax2, alpha=0.6, color='red', s=15, label='MLP Preds')
            ax2.set_title(f"Projected on Logit Norm", fontsize=12)
            ax2.set_xlabel("Logit Norm")
            
            plt.suptitle(f"Data Spread: {self.current_model} - Combined (MLP R2: {data['mlp_r2']:.3f})", fontsize=16)
        else:
            ax = self.fig.add_subplot(111)
            sns.scatterplot(x=data['X_test'][key], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual Data')
            ax.plot(data['X_test'][key], data['lin_pred'], color='blue', lw=2, linestyle='--', label=f"Linear (R2={data['lin_r2']:.3f})")
            ax.plot(data['X_test'][key], data['mlp_pred'], color='red', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
            
            ax.set_title(f"Data Spread: {self.current_model} - {title_label}", fontsize=16)
            ax.set_xlabel(key)
            ax.set_ylabel("Log Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_actual_vs_predicted(self, key, title_label):
        data = global_storage[self.current_model][key]
        y_true = data['y_test_raw']
        lin_pred = data['lin_pred_raw']
        mlp_pred = data['mlp_pred_raw']
        
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        def plot_diagonal(ax, y_t, y_p):
            min_val = min(y_t.min(), y_p.min())
            max_val = max(y_t.max(), y_p.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, label='Perfect Fit')

        # Linear
        sns.scatterplot(x=y_true, y=lin_pred, ax=ax1, alpha=0.3, color='blue', edgecolor='w', s=50)
        plot_diagonal(ax1, y_true, lin_pred)
        ax1.set_title(f"Linear Regression ({self.current_model})\nR2: {data['lin_r2']:.3f}", fontsize=14)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_box_aspect(1)

        # MLP
        sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.3, color='green', edgecolor='w', s=50)
        plot_diagonal(ax2, y_true, mlp_pred)
        ax2.set_title(f"Neural Network MLP ({self.current_model})\nR2: {data['mlp_r2']:.3f}", fontsize=14)
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.set_box_aspect(1)

        plt.suptitle(f"Prediction Accuracy: {title_label} ({self.current_model})", fontsize=16)

    def update(self):
        self.fig.clf() 
        plt.subplots_adjust(left=0.3) # More space for two menus
        
        sel = self.current_view
        
        # Plotting Logic
        if 'Data Spread' in sel:
            if 'Embedding' in sel: self._plot_data_spread('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_data_spread('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_data_spread('combined', 'Combined')
        elif 'Pred vs Actual' in sel:
            if 'Embedding' in sel: self._plot_actual_vs_predicted('embedding_norm', 'Embedding Norm')
            elif 'Logit' in sel: self._plot_actual_vs_predicted('logit_norm', 'Logit Norm')
            elif 'Combined' in sel: self._plot_actual_vs_predicted('combined', 'Combined Features')

        # --- MENU 1: VIEW SELECTOR ---
        ax_view = plt.axes([0.02, 0.45, 0.2, 0.4], facecolor='#f0f0f0')
        ax_view.set_title("1. Analysis View", fontsize=11, weight='bold')
        try: active_view_idx = self.view_options.index(self.current_view)
        except: active_view_idx = 0
        self.radio_view = RadioButtons(ax_view, self.view_options, active=active_view_idx)
        self.radio_view.on_clicked(self.on_view_click)

        # --- MENU 2: MODEL SELECTOR ---
        ax_model = plt.axes([0.02, 0.15, 0.2, 0.2], facecolor='#e6e6e6')
        ax_model.set_title("2. Select Model", fontsize=11, weight='bold')
        try: active_model_idx = self.available_models.index(self.current_model)
        except: active_model_idx = 0
        self.radio_model = RadioButtons(ax_model, self.available_models, active=active_model_idx)
        self.radio_model.on_clicked(self.on_model_click)

        plt.draw()

    def on_view_click(self, label):
        if '---' in label: return
        self.current_view = label
        self.update()

    def on_model_click(self, label):
        self.current_model = label
        self.update()

if __name__ == "__main__":
    viewer = SlideshowViewer()
    plt.show()