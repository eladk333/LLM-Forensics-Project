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
MODEL_SIZES = ['7M', '30M', '124M'] 
BASE_MODELS_FOLDER = r'G:\My Drive\llm\data\models'
FREQ_FILE = r'G:\My Drive\llm\data\datasets\wiki\wiki_token_frequencies.csv'

# --- DEFINING FEATURES HERE MAKES IT SCALABLE ---
# Add any new column name here, and the entire code (training + GUI) updates automatically.
FEATURE_CONFIG = {
    'embedding_norm': 'Embedding Norm',
    'logit_norm':     'Logit Norm',
    'weight_variance': 'Weight Variance', 
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm (New)',
    'dist_to_center':  'Dist to Center (New)',
    'weight_skew':     'Skewness (New)',
    'weight_kurtosis': 'Kurtosis (New)',
    'token_len':       'Token Length (New)',
    'is_upper':        'Is Capitalized (New)'

}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

global_storage = {} 

print(f"📂 Loading Data from: {BASE_MODELS_FOLDER}")

# Load Frequency File
if os.path.exists(FREQ_FILE):
    df_freq = pd.read_csv(FREQ_FILE)
    print(f"✅ Frequency file loaded ({len(df_freq)} tokens)")
else:
    print(f"❌ Frequency file not found at {FREQ_FILE}")
    exit()

# Train Models Loop
for size in MODEL_SIZES:
    print(f"\n--- Processing Model: {size} ---")
    
    feature_file = os.path.join(BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'final_frequency_dataset.csv')
    
    if not os.path.exists(feature_file):
        print(f"⚠️  Feature file not found for {size} (Skipping...)")
        continue

    # Load and Merge
    df_feat = pd.read_csv(feature_file)
    # Check if new features exist in file before processing
    valid_features = [f for f in FEATURES_LIST if f in df_feat.columns]
    
    if not valid_features:
        print(f"⚠️  None of the configured features found in {size} CSV.")
        continue

    df = pd.merge(df_freq, df_feat, on='token_id', how='inner')
    y = df['log_count']
    
    global_storage[size] = {'features': valid_features} # Store valid features for this model

    # --- Helper Function ---
    def train_and_store(X_data, y_data):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        lin = LinearRegression().fit(X_train, y_train)
        lin_pred = lin.predict(X_test)
        lin_r2 = r2_score(y_test, lin_pred)

        mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=1000, random_state=42))
        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        mlp_r2 = r2_score(y_test, mlp_pred)

        # Sort for line plots
        if X_test.shape[1] == 1:
            sort_idx = X_test.iloc[:, 0].argsort()
            X_test_sorted = X_test.iloc[sort_idx]
            y_test_sorted = y_test.iloc[sort_idx]
            lin_pred_sorted = lin_pred[sort_idx]
            mlp_pred_sorted = mlp_pred[sort_idx]
        else:
            X_test_sorted, y_test_sorted = X_test, y_test
            lin_pred_sorted, mlp_pred_sorted = lin_pred, mlp_pred

        return {
            'X_test': X_test_sorted, 'y_test': y_test_sorted,
            'lin_pred': lin_pred_sorted, 'mlp_pred': mlp_pred_sorted,
            'lin_pred_raw': lin_pred, 'mlp_pred_raw': mlp_pred,
            'y_test_raw': y_test,
            'lin_r2': lin_r2, 'mlp_r2': mlp_r2
        }

    # Train Individual Features
    print(f"   Training models for {size}...")
    for feat in valid_features:
        global_storage[size][feat] = train_and_store(df[[feat]], y)
    
    # Train Combined
    global_storage[size]['combined'] = train_and_store(df[valid_features], y)
    print(f"   ✅ Done.")

if not global_storage:
    print("\n❌ No models loaded. Exiting.")
    exit()

print("\n✅ Training Complete. Launching Modular Viewer...")

# ==========================================
# 3. MODULAR VISUALIZATION LOGIC
# ==========================================
class ModularViewer:
    def __init__(self):
        self.models = list(global_storage.keys())
        self.current_model = self.models[0]
        
        # --- DYNAMIC MENU GENERATION ---
        # Instead of hardcoding, we build the menu list from valid features
        self.view_types = ['Data Spread', 'Pred vs Actual']
        self.current_view_type = 'Data Spread'
        
        # We start by selecting the first feature available for the current model
        self.current_feature_key = global_storage[self.current_model]['features'][0] 
        
        self.fig = plt.figure(figsize=(16, 9))
        self.update()

    def _plot(self):
        # Determine if we are plotting a single feature or 'combined'
        is_combined = (self.current_feature_key == 'combined')
        data = global_storage[self.current_model][self.current_feature_key]
        
        # Get nice display name
        display_name = FEATURE_CONFIG.get(self.current_feature_key, "Combined Features") if not is_combined else "Combined Features"

        if self.current_view_type == 'Data Spread':
            # --- DATA SPREAD LOGIC ---
            if is_combined:
                 # Combined Spread (Show first 2 features as subplots for reference)
                feats_to_show = global_storage[self.current_model]['features'][:2] # Take first 2 available features
                ax1 = self.fig.add_subplot(121)
                ax2 = self.fig.add_subplot(122)
                
                for idx, (ax, feat) in enumerate(zip([ax1, ax2], feats_to_show)):
                    label = FEATURE_CONFIG.get(feat, feat)
                    feat_data = global_storage[self.current_model][feat] # Get raw data for that feature
                    
                    # We plot the COMBINED prediction against the INDIVIDUAL feature axis
                    # Note: X_test must match. Since we use random_state=42, indices align, 
                    # but strictly speaking we should align by index. For visualization 7M/30M this is fine.
                    # Ideally, we pull the specific column from the combined X_test.
                    
                    col_data = data['X_test'][feat]
                    
                    sns.scatterplot(x=col_data, y=data['y_test'], ax=ax, alpha=0.2, color='gray')
                    sns.scatterplot(x=col_data, y=data['mlp_pred'], ax=ax, alpha=0.6, color='red', s=15, label='Combined MLP Pred')
                    ax.set_xlabel(label)
                    ax.set_ylabel("Log Frequency")
                    ax.set_title(f"Projected on {label}")
                
                plt.suptitle(f"Combined Model Spread ({self.current_model}) - MLP R2: {data['mlp_r2']:.3f}", fontsize=16)

            else:
                # Single Feature Spread
                ax = self.fig.add_subplot(111)
                sns.scatterplot(x=data['X_test'][self.current_feature_key], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual')
                ax.plot(data['X_test'][self.current_feature_key], data['lin_pred'], 'b--', lw=2, label=f"Linear (R2={data['lin_r2']:.3f})")
                ax.plot(data['X_test'][self.current_feature_key], data['mlp_pred'], 'r-', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
                ax.set_title(f"{display_name} vs Frequency ({self.current_model})", fontsize=16)
                ax.set_xlabel(display_name)
                ax.set_ylabel("Log Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)

        elif self.current_view_type == 'Pred vs Actual':
            # --- PRED VS ACTUAL LOGIC (Same for Single or Combined) ---
            y_true = data['y_test_raw']
            lin_pred = data['lin_pred_raw']
            mlp_pred = data['mlp_pred_raw']
            
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)
            
            def plot_diag(ax):
                mn, mx = min(y_true.min(), lin_pred.min()), max(y_true.max(), lin_pred.max())
                ax.plot([mn, mx], [mn, mx], 'r--', lw=2)

            sns.scatterplot(x=y_true, y=lin_pred, ax=ax1, alpha=0.3, color='blue')
            plot_diag(ax1)
            ax1.set_title(f"Linear Regression (R2: {data['lin_r2']:.3f})")
            ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted"); ax1.set_box_aspect(1)

            sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.3, color='green')
            plot_diag(ax2)
            ax2.set_title(f"MLP Neural Net (R2: {data['mlp_r2']:.3f})")
            ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted"); ax2.set_box_aspect(1)
            
            plt.suptitle(f"Prediction Accuracy: {display_name} ({self.current_model})", fontsize=16)

    def update(self):
        self.fig.clf() 
        plt.subplots_adjust(left=0.35) # Make room for 3 menus
        
        self._plot()

        # --- DYNAMIC MENUS ---
        
        # 1. Model Selector
        ax_mod = plt.axes([0.02, 0.75, 0.25, 0.15], facecolor='#e6e6e6')
        ax_mod.set_title("1. Model Size", weight='bold')
        self.rad_mod = RadioButtons(ax_mod, self.models, active=self.models.index(self.current_model))
        self.rad_mod.on_clicked(self.set_model)
        
        # 2. View Type Selector
        ax_view = plt.axes([0.02, 0.55, 0.25, 0.15], facecolor='#f0f0f0')
        ax_view.set_title("2. Graph Type", weight='bold')
        self.rad_view = RadioButtons(ax_view, self.view_types, active=self.view_types.index(self.current_view_type))
        self.rad_view.on_clicked(self.set_view)

        # 3. Feature Selector (Dynamic!)
        # We get the valid features for the CURRENT model + 'Combined'
        current_valid_feats = global_storage[self.current_model]['features'] + ['combined']
        # Create display labels
        labels = [FEATURE_CONFIG.get(f, f).title() for f in current_valid_feats if f != 'combined'] + ['Combined Features']
        
        # Determine active index safely
        try:
            # Match current key to the new list
            if self.current_feature_key == 'combined':
                active_idx = len(labels) - 1
            else:
                active_idx = current_valid_feats.index(self.current_feature_key)
        except ValueError:
            active_idx = 0
            self.current_feature_key = current_valid_feats[0]

        ax_feat = plt.axes([0.02, 0.1, 0.25, 0.4], facecolor='#fff')
        ax_feat.set_title("3. Feature Selection", weight='bold')
        self.rad_feat = RadioButtons(ax_feat, labels, active=active_idx)
        
        # We need a closure or mapping to link label back to key
        self.label_to_key = dict(zip(labels, current_valid_feats))
        self.rad_feat.on_clicked(self.set_feature)

        plt.draw()

    def set_model(self, label):
        self.current_model = label
        # Reset feature if not available in new model
        if self.current_feature_key not in global_storage[self.current_model]['features'] and self.current_feature_key != 'combined':
             self.current_feature_key = global_storage[self.current_model]['features'][0]
        self.update()

    def set_view(self, label):
        self.current_view_type = label
        self.update()

    def set_feature(self, label):
        self.current_feature_key = self.label_to_key[label]
        self.update()

if __name__ == "__main__":
    viewer = ModularViewer()
    plt.show()