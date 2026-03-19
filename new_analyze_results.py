import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Button
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# Paths
MODEL_SIZES = ['7M', '30M', '124M'] 
BASE_MODELS_FOLDER = r'G:\My Drive\llm\data\models'
FREQ_FILE = r'G:\My Drive\llm\data\datasets\wiki\wiki_token_frequencies.csv'

# Feautres
FEATURE_CONFIG = {
    'embedding_norm': 'Embedding Norm',
    'logit_norm':     'Logit Norm',
    'weight_variance': 'Weight Variance', 
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
    # 'dist_to_center':  'Dist to Center',
    # 'weight_skew':     'Skewness (Doesnt help)',
    # 'weight_kurtosis': 'Kurtosis (Doesnt help)',
    'token_len':       'Token Length',
    'is_upper':        'Is Capitalized'
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# Like cache to avoid calculating stuff again
global_storage = {}   
global_dataframes = {}  




# Train the 2 predictor models  
def train_and_store(X_data, y_data):
    # Handles empty feature selection
    if X_data.shape[1] == 0:
        return None 

    # Splits the data into 80 percent for training and 20 percent for testing
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42) # seed 42 for it not to change between runs
    
    # Linear Model
    lin = LinearRegression().fit(X_train, y_train) # Trains the model
    lin_pred = lin.predict(X_test) # The prediction of the model y_hat
    lin_mae = mean_absolute_error(y_test, lin_pred) # Average absolute error in log_count units
    lin_rmse = mean_squared_error(y_test, lin_pred, squared=False) # RMSE penalizes large errors more

    # MLP Model    
    mlp = make_pipeline( # Creates the model
        StandardScaler(), # To normalize the data
          MLPRegressor( # The MLP config settings
              hidden_layer_sizes=(100, 50), # Two layers, first with 100 neurons and second with 50
                activation='tanh', # Activation function
                  max_iter=2000, # Let the model train for a long time to avoid underfitting
                  early_stopping=True, # To prevent overfitting
                  validation_fraction=0.1,
                  n_iter_no_change=10, # If no improvment after 10 rounds we stop training
                  random_state=42))
    
    mlp.fit(X_train, y_train) # Train the model
    mlp_pred = mlp.predict(X_test) # The prediction of the model y_hat
    mlp_mae = mean_absolute_error(y_test, mlp_pred) # Average absolute error in log_count units
    mlp_rmse = mean_squared_error(y_test, mlp_pred, squared=False) # RMSE penalizes large errors more

    # If only have 1 feature we can sort the x so we can see the data spread
    if X_test.shape[1] == 1:
        sort_idx = X_test.iloc[:, 0].argsort()
        X_test_sorted = X_test.iloc[sort_idx]
        y_test_sorted = y_test.iloc[sort_idx]
        lin_pred_sorted = lin_pred[sort_idx]
        mlp_pred_sorted = mlp_pred[sort_idx]
    else: # Can't sort for multiple x        
        X_test_sorted, y_test_sorted = X_test, y_test
        lin_pred_sorted, mlp_pred_sorted = lin_pred, mlp_pred

    return {
        'X_test': X_test_sorted, 'y_test': y_test_sorted,
        'lin_pred': lin_pred_sorted, 'mlp_pred': mlp_pred_sorted,
        'lin_pred_raw': lin_pred, 'mlp_pred_raw': mlp_pred,
        'y_test_raw': y_test,
        'lin_mae': lin_mae, 'lin_rmse': lin_rmse,
        'mlp_mae': mlp_mae, 'mlp_rmse': mlp_rmse
    }




class ModularViewer:
    def __init__(self):
        self.models = list(global_storage.keys())
        self.current_model = self.models[0]
        self.view_types = ['Data Spread', 'Pred vs Actual']
        self.current_view_type = 'Data Spread'
        
        # Start with 'combined' view
        self.current_feature_key = 'combined'
        
        # Setup Figure
        self.fig = plt.figure(figsize=(16, 9))
        self.update()

    def _plot(self):
        # Determine if we are plotting a single feature or 'combined'
        is_combined = (self.current_feature_key == 'combined')
        
        # Handle case where training failed (e.g. 0 features selected)
        if global_storage[self.current_model].get(self.current_feature_key) is None:
            plt.clf()
            plt.text(0.5, 0.5, "No features selected for training.\nPlease select at least one feature and click 'Retrain'.", 
                     ha='center', va='center', fontsize=14)
            return

        data = global_storage[self.current_model][self.current_feature_key]
        display_name = FEATURE_CONFIG.get(self.current_feature_key, "Combined Features") if not is_combined else "Combined Features"

        if self.current_view_type == 'Data Spread':
            # --- DATA SPREAD LOGIC ---
            if is_combined:
                # Combined Spread: Show the features that were actually used
                active_feats = global_storage[self.current_model].get('active_combined_features', [])
                
                # Show up to 2 features for reference
                feats_to_show = active_feats[:2] 
                
                if not feats_to_show:
                    plt.text(0.5, 0.5, "No active features to display.", ha='center')
                    return

                ax1 = self.fig.add_subplot(121)
                ax2 = self.fig.add_subplot(122) if len(feats_to_show) > 1 else None
                
                axes_list = [ax1] if ax2 is None else [ax1, ax2]

                for idx, (ax, feat) in enumerate(zip(axes_list, feats_to_show)):
                    label = FEATURE_CONFIG.get(feat, feat)
                    
                    # Get the X data for this specific feature from the Test set
                    # We look up the raw value in the original X_test stored in the result
                    col_data = data['X_test'][feat]
                    
                    sns.scatterplot(x=col_data, y=data['y_test'], ax=ax, alpha=0.2, color='gray')
                    sns.scatterplot(x=col_data, y=data['mlp_pred'], ax=ax, alpha=0.6, color='red', s=15, label='Combined Model Pred')
                    ax.set_xlabel(label)
                    ax.set_ylabel("Log Frequency")
                    ax.set_title(f"Projected on {label}")
                
                feature_count = len(active_feats)
                plt.suptitle(f"Combined Model ({feature_count} features) - MLP MAE: {data['mlp_mae']:.3f}  RMSE: {data['mlp_rmse']:.3f}", fontsize=16)

            else:
                # Single Feature Spread
                ax = self.fig.add_subplot(111)
                sns.scatterplot(x=data['X_test'][self.current_feature_key], y=data['y_test'], ax=ax, alpha=0.3, color='gray', label='Actual')
                ax.plot(data['X_test'][self.current_feature_key], data['lin_pred'], 'b--', lw=2, label=f"Linear (MAE={data['lin_mae']:.3f})")
                ax.plot(data['X_test'][self.current_feature_key], data['mlp_pred'], 'r-', lw=3, label=f"MLP (MAE={data['mlp_mae']:.3f})")
                ax.set_title(f"{display_name} vs Frequency ({self.current_model})", fontsize=16)
                ax.set_xlabel(display_name)
                ax.set_ylabel("Log Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)

        elif self.current_view_type == 'Pred vs Actual':
            # --- PRED VS ACTUAL LOGIC ---
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
            ax1.set_title(f"Linear Regression (MAE: {data['lin_mae']:.3f}  RMSE: {data['lin_rmse']:.3f})")
            ax1.set_xlabel("Actual"); ax1.set_ylabel("Predicted"); ax1.set_box_aspect(1)

            sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.3, color='green')
            plot_diag(ax2)
            ax2.set_title(f"MLP Neural Net (MAE: {data['mlp_mae']:.3f}  RMSE: {data['mlp_rmse']:.3f})")
            ax2.set_xlabel("Actual"); ax2.set_ylabel("Predicted"); ax2.set_box_aspect(1)
            
            plt.suptitle(f"Prediction Accuracy: {display_name} ({self.current_model})", fontsize=16)

    def update(self):

        if hasattr(self, 'check'):
            self.check = None
        self.fig.clf() 
        # Adjust layout to make room for controls on Left AND Right
        plt.subplots_adjust(left=0.25, right=0.8) 
        
        self._plot()

        # --- LEFT SIDEBAR CONTROLS ---
        
        # 1. Model Selector
        ax_mod = plt.axes([0.02, 0.75, 0.18, 0.15], facecolor='#e6e6e6')
        ax_mod.set_title("1. Model Size", weight='bold')
        self.rad_mod = RadioButtons(ax_mod, self.models, active=self.models.index(self.current_model))
        self.rad_mod.on_clicked(self.set_model)
        
        # 2. View Type Selector
        ax_view = plt.axes([0.02, 0.55, 0.18, 0.15], facecolor='#f0f0f0')
        ax_view.set_title("2. Graph Type", weight='bold')
        self.rad_view = RadioButtons(ax_view, self.view_types, active=self.view_types.index(self.current_view_type))
        self.rad_view.on_clicked(self.set_view)

        # 3. Feature Focus Selector
        current_valid_feats = global_storage[self.current_model]['features'] + ['combined']
        labels = [FEATURE_CONFIG.get(f, f).title() for f in current_valid_feats if f != 'combined'] + ['Combined Features']
        
        try:
            if self.current_feature_key == 'combined':
                active_idx = len(labels) - 1
            else:
                active_idx = current_valid_feats.index(self.current_feature_key)
        except ValueError:
            active_idx = 0
            self.current_feature_key = current_valid_feats[0]

        ax_feat = plt.axes([0.02, 0.05, 0.18, 0.45], facecolor='#fff')
        ax_feat.set_title("3.Features", weight='bold')
        self.rad_feat = RadioButtons(ax_feat, labels, active=active_idx)
        self.label_to_key = dict(zip(labels, current_valid_feats))
        self.rad_feat.on_clicked(self.set_feature)

        # --- RIGHT SIDEBAR CONTROLS (Combined Config) ---
        # Only show/enable these if 'Combined Features' is selected
        if self.current_feature_key == 'combined':
            
            # Title
            plt.figtext(0.82, 0.90, "Combined Config", fontsize=12, weight='bold')
            
            # 4. Checkboxes for Features
            # Get all potential features for this model
            avail_feats = global_dataframes[self.current_model]['valid_features']
            avail_labels = [FEATURE_CONFIG.get(f, f) for f in avail_feats]
            
            # Determine which are currently active in the stored model
            active_feats = global_storage[self.current_model].get('active_combined_features', avail_feats)
            actives = [f in active_feats for f in avail_feats]

            ax_check = plt.axes([0.82, 0.20, 0.16, 0.65], frame_on=False)
            self.check = CheckButtons(ax_check, avail_labels, actives)
            
            # Map labels back to feature keys for the callback
            self.check_label_to_key = dict(zip(avail_labels, avail_feats))

            # 5. Retrain Button
            ax_btn = plt.axes([0.82, 0.05, 0.15, 0.08])
            self.btn = Button(ax_btn, 'Retrain Model', color='lightblue', hovercolor='skyblue')
            self.btn.on_clicked(self.retrain_combined)

        plt.draw()

    def set_model(self, label):
        self.current_model = label
        if self.current_feature_key not in global_storage[self.current_model]['features'] and self.current_feature_key != 'combined':
             self.current_feature_key = global_storage[self.current_model]['features'][0]
        self.update()

    def set_view(self, label):
        self.current_view_type = label
        self.update()

    def set_feature(self, label):
        self.current_feature_key = self.label_to_key[label]
        self.update()

    def retrain_combined(self, event):
        """Callback to retrain the combined model with checked features."""
        
        # 1. Get checked status
        # CheckButtons.get_status() returns a list of booleans matching the labels
        status = self.check.get_status()
        
        # 2. Filter features
        avail_feats = global_dataframes[self.current_model]['valid_features']
        selected_features = [f for f, s in zip(avail_feats, status) if s]
        
        print(f"\n🔄 Retraining Combined Model for {self.current_model}...")
        print(f"   Selected: {selected_features}")
        
        if not selected_features:
            print("   ⚠️ No features selected!")
            global_storage[self.current_model]['combined'] = None
        else:
            # 3. Retrain
            df = global_dataframes[self.current_model]['df']
            y = global_dataframes[self.current_model]['y']
            
            new_results = train_and_store(df[selected_features], y)
            
            global_storage[self.current_model]['combined'] = new_results
            global_storage[self.current_model]['active_combined_features'] = selected_features
            print(f"   ✅ Retraining Complete. MAE: {new_results['mlp_mae']:.4f}  RMSE: {new_results['mlp_rmse']:.4f}")
                        

        # 5. Refresh Plot
        self.update()

        

if __name__ == "__main__":

    # Load token frequency file
    if os.path.exists(FREQ_FILE):
        df_freq = pd.read_csv(FREQ_FILE)        
    else:
        print(f"File not found at {FREQ_FILE}")
        exit()

    # 2. Train Models Loop
    for size in MODEL_SIZES:
        print(f"\nProcessing Model: {size}")
        
        # Path for the feature file
        feature_file = os.path.join(BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'model_features.csv')
        
        if not os.path.exists(feature_file):
            print(f"Feature file not found for {size}. Skipping")
            continue

        
        df_features = pd.read_csv(feature_file) # Load feature file into df
        valid_features = [f for f in FEATURES_LIST if f in df_features.columns] # Filter for only features we have 
                

        df = pd.merge(df_freq, df_features, on='token_id', how='inner') # Merge the freq file with the features file
        y = df['log_count'] # Define the log count column as the traget of our predictor models
        
        # Stores the data so we could retrain on less features later on
        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid_features}
        global_storage[size] = {'features': valid_features}

        # Train Individual Features        
        for feat in valid_features:
            global_storage[size][feat] = train_and_store(df[[feat]], y)
        
        # Train Initial Combined
        print(f"   Training combined model...")
        global_storage[size]['combined'] = train_and_store(df[valid_features], y)
        global_storage[size]['active_combined_features'] = valid_features

        print(f"   ✅ Done.")

    if not global_storage:
        print("\nNo models loaded.")
        exit()
    

    viewer = ModularViewer()
    plt.show()