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
from sklearn.metrics import r2_score
import os

# ---------------------------------------------------------
# 1. Paths & Mappings
# ---------------------------------------------------------
# Updated to your active local workspace
BASE_DIR = r'C:\Users\elad.k.int\LLM-Forensics-Project'
BASE_MODELS_FOLDER = os.path.join(BASE_DIR, 'data', 'models')

MODEL_SIZES = ['7M', '30M', '124M', '500M']

# Maps model size -> subfolder name
MODEL_FOLDER_MAP = {
    '7M':   'MinGPT_Checkpoints_7M',
    '30M':  'MinGPT_Checkpoints_30M',
    '124M': 'MinGPT_Checkpoints_124M',
    '500M': 'MinGPT_Checkpoints_500M',
}

# Maps model size -> correct frequency dataset
FREQ_FILE_MAP = {
    '7M':   os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '30M':  os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '124M': os.path.join(BASE_DIR, 'data', 'datasets', 'wiki', 'wiki_token_frequencies.csv'),
    '500M': os.path.join(BASE_DIR, 'data', 'datasets', 'openweb', 'openwebtext_token_frequencies.csv'),
}

# Features
FEATURE_CONFIG = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
    
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# Cache to avoid recalculating
global_storage = {}
global_dataframes = {}

# ---------------------------------------------------------
# 2. Training Logic
# ---------------------------------------------------------
def train_and_store(X_data, y_data):
    if X_data.shape[1] == 0:
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42)

    # Linear Pipeline
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred)

    if X_data.shape[1] > 1:
        print("\n   [Linear Regression] STANDARDIZED Feature Weights:")
        lin_step = lin.named_steps['linearregression']
        for feature_name, coef in zip(X_data.columns, lin_step.coef_):
            print(f"{FEATURE_CONFIG.get(feature_name, feature_name)}: {coef:.6f}")
        print(f"      -> Y-Intercept: {lin_step.intercept_:.6f}")

    # MLP Pipeline
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='tanh',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42))

    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred)

    if X_test.shape[1] == 1:
        sort_idx = X_test.iloc[:, 0].argsort()
        X_test_sorted     = X_test.iloc[sort_idx]
        y_test_sorted     = y_test.iloc[sort_idx]
        lin_pred_sorted   = lin_pred[sort_idx]
        mlp_pred_sorted   = mlp_pred[sort_idx]
    else:
        X_test_sorted, y_test_sorted     = X_test, y_test
        lin_pred_sorted, mlp_pred_sorted = lin_pred, mlp_pred

    return {
        'X_test': X_test_sorted, 'y_test': y_test_sorted,
        'lin_pred': lin_pred_sorted, 'mlp_pred': mlp_pred_sorted,
        'lin_pred_raw': lin_pred, 'mlp_pred_raw': mlp_pred,
        'y_test_raw': y_test,
        'lin_r2': lin_r2, 'mlp_r2': mlp_r2
    }

# ---------------------------------------------------------
# 3. GUI Logic
# ---------------------------------------------------------
class ModularViewer:
    def __init__(self):
        self.models = list(global_storage.keys())
        self.current_model = self.models[0]
        self.view_types = ['Data Spread', 'Pred vs Actual']
        self.current_view_type = 'Data Spread'
        self.current_feature_key = 'combined'

        self.fig = plt.figure(figsize=(16, 9))
        self.update()

    def _plot(self):
        is_combined = (self.current_feature_key == 'combined')

        if global_storage[self.current_model].get(self.current_feature_key) is None:
            plt.clf()
            plt.text(0.5, 0.5,
                     "No features selected for training.\nPlease select at least one feature and click 'Retrain'.",
                     ha='center', va='center', fontsize=14)
            return

        data = global_storage[self.current_model][self.current_feature_key]
        display_name = FEATURE_CONFIG.get(self.current_feature_key, "Combined Features") if not is_combined else "Combined Features"

        if self.current_view_type == 'Data Spread':
            if is_combined:
                active_feats = global_storage[self.current_model].get('active_combined_features', [])
                feats_to_show = active_feats[:2]

                if not feats_to_show:
                    plt.text(0.5, 0.5, "No active features to display.", ha='center')
                    return

                ax1 = self.fig.add_subplot(121)
                ax2 = self.fig.add_subplot(122) if len(feats_to_show) > 1 else None
                axes_list = [ax1] if ax2 is None else [ax1, ax2]

                for ax, feat in zip(axes_list, feats_to_show):
                    label = FEATURE_CONFIG.get(feat, feat)
                    col_data = data['X_test'][feat]
                    sns.scatterplot(x=col_data, y=data['y_test'], ax=ax, alpha=0.2, color='gray')
                    sns.scatterplot(x=col_data, y=data['mlp_pred'], ax=ax, alpha=0.6,
                                    color='red', s=15, label='Combined Model Pred')
                    ax.set_xlabel(label)
                    ax.set_ylabel("Log Frequency")
                    ax.set_title(f"Projected on {label}")

                feature_count = len(active_feats)
                plt.suptitle(
                    f"Combined Model ({feature_count} features) - MLP R2: {data['mlp_r2']:.3f}",
                    fontsize=16)

            else:
                ax = self.fig.add_subplot(111)
                sns.scatterplot(x=data['X_test'][self.current_feature_key], y=data['y_test'],
                                ax=ax, alpha=0.3, color='gray', label='Actual')
                ax.plot(data['X_test'][self.current_feature_key], data['lin_pred'],
                        'b--', lw=2, label=f"Linear (R2={data['lin_r2']:.3f})")
                ax.plot(data['X_test'][self.current_feature_key], data['mlp_pred'],
                        'r-', lw=3, label=f"MLP (R2={data['mlp_r2']:.3f})")
                ax.set_title(f"{display_name} vs Frequency ({self.current_model})", fontsize=16)
                ax.set_xlabel(display_name)
                ax.set_ylabel("Log Frequency")
                ax.legend()
                ax.grid(True, alpha=0.3)

        elif self.current_view_type == 'Pred vs Actual':
            y_true   = data['y_test_raw']
            lin_pred = data['lin_pred_raw']
            mlp_pred = data['mlp_pred_raw']

            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            def plot_diag(ax):
                lims = [min(y_true.min(), ax.get_xlim()[0]),
                        max(y_true.max(), ax.get_xlim()[1])]
                ax.plot(lims, lims, 'k--', alpha=0.5, lw=1)

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
        if hasattr(self, 'check'):
            self.check = None
        self.fig.clf()
        plt.subplots_adjust(left=0.25, right=0.8)

        self._plot()

        # LEFT SIDEBAR
        ax_mod = plt.axes([0.02, 0.75, 0.18, 0.15], facecolor='#e6e6e6')
        ax_mod.set_title("1. Model Size", weight='bold')
        self.rad_mod = RadioButtons(ax_mod, self.models, active=self.models.index(self.current_model))
        self.rad_mod.on_clicked(self.set_model)

        ax_view = plt.axes([0.02, 0.55, 0.18, 0.15], facecolor='#f0f0f0')
        ax_view.set_title("2. Graph Type", weight='bold')
        self.rad_view = RadioButtons(ax_view, self.view_types,
                                     active=self.view_types.index(self.current_view_type))
        self.rad_view.on_clicked(self.set_view)

        current_valid_feats = global_storage[self.current_model]['features'] + ['combined']
        labels = [FEATURE_CONFIG.get(f, f).title() for f in current_valid_feats if f != 'combined'] + ['Combined Features']

        try:
            active_idx = len(labels) - 1 if self.current_feature_key == 'combined' \
                         else current_valid_feats.index(self.current_feature_key)
        except ValueError:
            active_idx = 0
            self.current_feature_key = current_valid_feats[0]

        ax_feat = plt.axes([0.02, 0.05, 0.18, 0.45], facecolor='#fff')
        ax_feat.set_title("3. Features", weight='bold')
        self.rad_feat = RadioButtons(ax_feat, labels, active=active_idx)
        self.label_to_key = dict(zip(labels, current_valid_feats))
        self.rad_feat.on_clicked(self.set_feature)

        # RIGHT SIDEBAR (only for Combined)
        if self.current_feature_key == 'combined':
            plt.figtext(0.82, 0.90, "Combined Config", fontsize=12, weight='bold')

            avail_feats  = global_dataframes[self.current_model]['valid_features']
            avail_labels = [FEATURE_CONFIG.get(f, f) for f in avail_feats]
            active_feats = global_storage[self.current_model].get('active_combined_features', avail_feats)
            actives = [f in active_feats for f in avail_feats]

            ax_check = plt.axes([0.82, 0.20, 0.16, 0.65], frame_on=False)
            self.check = CheckButtons(ax_check, avail_labels, actives)
            self.check_label_to_key = dict(zip(avail_labels, avail_feats))

            ax_btn = plt.axes([0.82, 0.05, 0.15, 0.08])
            self.btn = Button(ax_btn, 'Retrain Model', color='lightblue', hovercolor='skyblue')
            self.btn.on_clicked(self.retrain_combined)

        plt.draw()

    def set_model(self, label):
        self.current_model = label
        if self.current_feature_key not in global_storage[self.current_model]['features'] \
                and self.current_feature_key != 'combined':
            self.current_feature_key = global_storage[self.current_model]['features'][0]
        self.update()

    def set_view(self, label):
        self.current_view_type = label
        self.update()

    def set_feature(self, label):
        self.current_feature_key = self.label_to_key[label]
        self.update()

    def retrain_combined(self, event):
        status = self.check.get_status()
        avail_feats = global_dataframes[self.current_model]['valid_features']
        selected_features = [f for f, s in zip(avail_feats, status) if s]

        print(f"\nRetraining Combined Model for {self.current_model}...")
        print(f"   Selected: {selected_features}")

        if not selected_features:
            print("No features selected!")
            global_storage[self.current_model]['combined'] = None
        else:
            df = global_dataframes[self.current_model]['df']
            y  = global_dataframes[self.current_model]['y']
            new_results = train_and_store(df[selected_features], y)
            global_storage[self.current_model]['combined'] = new_results
            global_storage[self.current_model]['active_combined_features'] = selected_features
            print(f"   ✅ Retraining Complete. R2: {new_results['mlp_r2']:.4f}")

        self.update()

# ---------------------------------------------------------
# 4. Main Execution Setup
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Initializing environment and caching frequencies...")
    # Pre-load both frequency files
    df_freq_cache = {}
    for freq_path in set(FREQ_FILE_MAP.values()):
        if not os.path.exists(freq_path):
            print(f"❌ Frequency file not found: {freq_path}")
            print("Run token_frequency.py first to generate it.")
            exit()
        df_freq_cache[freq_path] = pd.read_csv(freq_path)
        print(f"  ✅ Cached: {os.path.basename(freq_path)}")

    # Train models loop
    for size in MODEL_SIZES:
        print(f"\nProcessing Model: {size}")

        folder = MODEL_FOLDER_MAP[size]
        feature_file = os.path.join(BASE_MODELS_FOLDER, folder, 'model_features.csv')

        if not os.path.exists(feature_file):
            print(f"   ⚠️ Feature file not found at {feature_file}. Skipping.")
            continue

        df_features    = pd.read_csv(feature_file)
        valid_features = [f for f in FEATURES_LIST if f in df_features.columns]

        df_freq = df_freq_cache[FREQ_FILE_MAP[size]]
        df = pd.merge(df_freq, df_features, on='token_id', how='inner')
        
        # Robust fallback: use log_count if it exists, otherwise create it
        y = df['log_count'] if 'log_count' in df.columns else np.log1p(df['count'])

        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid_features}
        global_storage[size]    = {'features': valid_features}

        for feat in valid_features:
            global_storage[size][feat] = train_and_store(df[[feat]], y)

        print(f"   Training combined model...")
        global_storage[size]['combined'] = train_and_store(df[valid_features], y)
        global_storage[size]['active_combined_features'] = valid_features
        print(f"   Done.")

    if not global_storage:
        print("\nNo models loaded. Check that feature files exist.")
        exit()

    viewer = ModularViewer()
    plt.show()