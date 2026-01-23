import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons, Button
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import os

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Inputs
FEATURE_CSV = os.path.join(CURRENT_DIR, "forensic_features.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

# Feature Mapping (Display Name -> Column Name)
FEATURE_CONFIG = {
    'Embedding Norm (L2)': 'embedding_norm',
    'Logit Norm':          'logit_norm',
    'Weight Variance':     'weight_variance',
    'L1 Norm':             'l1_norm',
    'Next-Token Probability': 'Avg_Probability' # From your probability experiment
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())
MODEL_SIZES = ['7M', '30M', '124M']

# ==========================================
# 2. SPECIALIZED PREDICTION MODELS
# ==========================================
# Logic from predict_gradient.py (Exponential/Inverse Model)
def inverse_exp_func(prob, a, b):
    return a * np.exp(b * prob)

# Logic from 3.txt (Polynomial Regression)
def get_poly_model(degree=2):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())

# Standard Models
def get_linear_model():
    return make_pipeline(StandardScaler(), LinearRegression())

def get_mlp_model():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42))

# ==========================================
# 3. DATA LOADING & MERGING
# ==========================================
def load_and_merge_data():
    if not os.path.exists(FEATURE_CSV) or not os.path.exists(PROB_CSV):
        print("❌ Error: Missing input CSV files.")
        return None

    print("🔄 Merging Forensic Data...")
    df_feats = pd.read_csv(FEATURE_CSV)
    df_probs = pd.read_csv(PROB_CSV)

    # Standardize Model names
    df_probs['Model'] = df_probs['Model'].apply(lambda x: x if 'Model' in x else f"Model {x}")
    df_feats['Model'] = df_feats['Model'].astype(str)
    
    # Merge on Model and Step
    df_merged = pd.merge(df_feats, df_probs, on=['Model', 'Step'], how='inner')
    print(f"✅ Merged Dataset: {len(df_merged)} samples.")
    return df_merged

# ==========================================
# 4. ANALYSIS & PLOTTING LOGIC
# ==========================================
class ForensicAnalyzer:
    def __init__(self, df):
        self.df = df
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(left=0.3)
        
        self.selected_features = [FEATURES_LIST[0]] # Default: Embedding Norm
        self.selected_model_type = 'Linear'
        self.current_size = '124M'
        
        self.init_gui()
        self.update_plot()

    def init_gui(self):
        # 1. Model Size Selection
        ax_size = plt.axes([0.05, 0.75, 0.15, 0.15], facecolor='#f0f0f0')
        self.radio_size = RadioButtons(ax_size, MODEL_SIZES, active=2)
        self.radio_size.on_clicked(self.set_size)

        # 2. Regressor Selection
        ax_reg = plt.axes([0.05, 0.55, 0.15, 0.15], facecolor='#f0f0f0')
        self.radio_reg = RadioButtons(ax_reg, ['Linear', 'Polynomial', 'MLP', 'Exponential'], active=0)
        self.radio_reg.on_clicked(self.set_regressor)

        # 3. Feature Selection
        ax_feat = plt.axes([0.05, 0.1, 0.2, 0.4], facecolor='#f0f0f0')
        self.check_feat = CheckButtons(ax_feat, FEATURES_LIST, [True] + [False]*(len(FEATURES_LIST)-1))
        self.check_feat.on_clicked(self.set_features)

    def set_size(self, label):
        self.current_size = label
        self.update_plot()

    def set_regressor(self, label):
        self.selected_model_type = label
        self.update_plot()

    def set_features(self, label):
        # Toggle logic handled by CheckButtons visually, need to update internal state
        self.selected_features = [l for l, state in zip(FEATURES_LIST, self.check_feat.get_status()) if state]
        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()
        
        if not self.selected_features:
            self.ax.set_title("Select at least one feature")
            plt.draw()
            return

        # Prepare Data
        subset = self.df[self.df['Model'] == f"Model {self.current_size}"].sort_values('Step')
        cols = [FEATURE_CONFIG[f] for f in self.selected_features]
        X = subset[cols].values
        y = subset['Step'].values

        # Special Case: Exponential Model (only works for Probability)
        if self.selected_model_type == 'Exponential':
            if len(cols) == 1 and cols[0] == 'Avg_Probability':
                try:
                    popt, _ = curve_fit(inverse_exp_func, X.flatten(), y, maxfev=5000)
                    y_pred = inverse_exp_func(X.flatten(), *popt)
                    r2 = r2_score(y, y_pred)
                    self.ax.plot(y, y_pred, 'g-', label=f'Exp Fit (R²={r2:.3f})')
                    self.ax.set_title(f"Exponential Model (Probability Analysis)\nFormula: Steps = {popt[0]:.2e} * exp({popt[1]:.2f} * p)")
                except:
                    self.ax.set_title("Exponential Fit Failed")
            else:
                self.ax.set_title("Exponential Model only works with 'Next-Token Probability' alone.")
                plt.draw()
                return
        else:
            # Standard Models
            if self.selected_model_type == 'Linear':
                model = get_linear_model()
            elif self.selected_model_type == 'Polynomial':
                model = get_poly_model(degree=2)
            else: # MLP
                model = get_mlp_model()

            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            self.ax.plot(y, y_pred, 'b-', linewidth=2, label=f'Predicted (R²={r2:.3f})')
            self.ax.set_title(f"Forensic Prediction | Model: {self.current_size} | Regressor: {self.selected_model_type}")

        # Common Plotting
        self.ax.scatter(y, y, c='red', marker='--', alpha=0.5, label='Ground Truth') # Ideal line
        if self.selected_model_type != 'Exponential':
            self.ax.scatter(y, y_pred, alpha=0.6, label='Predictions')

        self.ax.set_xlabel("Actual Gradient Steps")
        self.ax.set_ylabel("Predicted Steps")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        plt.draw()

def main():
    df = load_and_merge_data()
    if df is not None:
        print("📊 Launching Analyzer GUI...")
        analyzer = ForensicAnalyzer(df)
        plt.show()

if __name__ == "__main__":
    main()