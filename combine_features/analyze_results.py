import matplotlib
matplotlib.use('TkAgg') # Use the basic and stable Windows graphics engine
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import os

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Paths to your EXACT original base files
FEATURE_CSV = os.path.join(CURRENT_DIR, "forensic_features.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

# Fallback just in case PROB_CSV was copied into the same folder
if not os.path.exists(PROB_CSV):
    PROB_CSV = os.path.join(CURRENT_DIR, "probability_results_server.csv")

# Feature Mapping - Explicitly defining the global features
FEATURE_CONFIG = {
    'Embedding Norm (Frobenius)': 'embedding_norm',
    'Logit Norm':                 'logit_norm',
    'Weight Variance':            'weight_variance',
    'L1 Norm':                    'l1_norm',
    'Next-Token Probability':     'Avg_Probability'
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())
MODEL_SIZES = ['7M', '30M', '124M']

# ==========================================
# 2. SPECIALIZED PREDICTION MODELS
# ==========================================
def inverse_exp_func(prob, a, b):
    return a * np.exp(b * prob)

def get_poly_model(degree=2):
    return make_pipeline(StandardScaler(), PolynomialFeatures(degree), LinearRegression())

def get_linear_model():
    return make_pipeline(StandardScaler(), LinearRegression())

def get_mlp_model():
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42))

# ==========================================
# 3. DATA LOADING & MERGING
# ==========================================
def load_and_merge_data():
    print("🔄 Loading Global Baseline Datasets...")
    
    if not os.path.exists(FEATURE_CSV):
        print(f"❌ Error: Missing forensic features file at: {FEATURE_CSV}")
        return None
        
    if not os.path.exists(PROB_CSV):
        print(f"❌ Error: Missing probability file at: {PROB_CSV}")
        return None

    df_feats = pd.read_csv(FEATURE_CSV)
    df_probs = pd.read_csv(PROB_CSV)

    # Standardize Model names to ensure perfect merge
    df_probs['Model'] = df_probs['Model'].apply(lambda x: x if 'Model' in x else f"Model {x}")
    df_feats['Model'] = df_feats['Model'].astype(str)
    
    # Merge the ~60 samples based on Model and Step
    df_merged = pd.merge(df_feats, df_probs, on=['Model', 'Step'], how='inner')
    print(f"✅ Merged Dataset Ready: {len(df_merged)} global samples loaded.")
    return df_merged

# ==========================================
# 4. ANALYSIS & PLOTTING LOGIC
# ==========================================
class ForensicAnalyzer:
    def __init__(self, df):
        self.df = df
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        plt.subplots_adjust(left=0.35)
        
        self.selected_features = [FEATURES_LIST[0]]
        self.selected_model_type = 'Linear'
        self.current_size = '124M'
        
        self.init_gui()
        self.update_plot()

    def init_gui(self):
        ax_size = plt.axes([0.05, 0.75, 0.15, 0.15], facecolor='#f0f0f0')
        self.radio_size = RadioButtons(ax_size, MODEL_SIZES, active=2)
        self.radio_size.on_clicked(self.set_size)

        ax_reg = plt.axes([0.05, 0.55, 0.15, 0.15], facecolor='#f0f0f0')
        self.radio_reg = RadioButtons(ax_reg, ['Linear', 'Polynomial', 'MLP', 'Exponential'], active=0)
        self.radio_reg.on_clicked(self.set_regressor)

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
        self.selected_features = [l for l, state in zip(FEATURES_LIST, self.check_feat.get_status()) if state]
        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()
        
        if not self.selected_features:
            self.ax.set_title("Select at least one feature")
            plt.draw()
            return

        subset = self.df[self.df['Model'] == f"Model {self.current_size}"].sort_values('Step')
        cols = [FEATURE_CONFIG[f] for f in self.selected_features]
        X = subset[cols].values
        y = subset['Step'].values

        # Exponential Model Logic (Specific for Probability)
        if self.selected_model_type == 'Exponential':
            if len(cols) == 1 and cols[0] == 'Avg_Probability':
                try:
                    popt, _ = curve_fit(inverse_exp_func, X.flatten(), y, maxfev=5000)
                    y_pred = inverse_exp_func(X.flatten(), *popt)
                    r2 = r2_score(y, y_pred)
                    self.ax.scatter(y, y_pred, color='green', alpha=0.6, label='Exp Predictions')
                    self.ax.set_title(f"Exponential Model (Probability)\nSteps = {popt[0]:.2e} * exp({popt[1]:.2f} * p)\nGlobal R²={r2:.3f}")
                except:
                    self.ax.set_title("Exponential Fit Failed")
            else:
                self.ax.set_title("Exponential Model only works with 'Next-Token Probability' alone.")
                plt.draw()
                return
        else:
            # ---------------------------------------------------------
            # THE YANAI SHIELD: 5-Fold Cross Validation for ML Models
            # ---------------------------------------------------------
            if self.selected_model_type == 'Linear':
                model = get_linear_model()
            elif self.selected_model_type == 'Polynomial':
                model = get_poly_model(degree=2)
            else: # MLP
                model = get_mlp_model()

            # Initialize K-Fold
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Predict each sample ONLY when it's in the unseen validation set
            y_pred_cv = cross_val_predict(model, X, y, cv=cv)
            
            # Calculate mean R^2 across the 5 folds
            cv_r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            mean_r2 = np.mean(cv_r2_scores)
            
            self.ax.scatter(y, y_pred_cv, color='blue', alpha=0.6, label='CV Predictions (Unseen Data)')
            self.ax.set_title(f"Forensic Multivariate Prediction | Model: {self.current_size} | {self.selected_model_type}\nRobust 5-Fold CV R² = {mean_r2:.3f}")

        # Common Plotting - The Ideal "Target" Line
        self.ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', alpha=0.5, label='Ideal (X=Y)')
        
        self.ax.set_xlabel("Actual Gradient Steps")
        self.ax.set_ylabel("Predicted Steps")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        plt.draw()

def main():
    df = load_and_merge_data()
    if df is not None:
        print("📊 Launching Validated Analyzer GUI...")
        analyzer = ForensicAnalyzer(df)
        plt.show()

if __name__ == "__main__":
    main()