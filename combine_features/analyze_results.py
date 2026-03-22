import matplotlib
matplotlib.use('TkAgg')
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
import warnings
from scipy.optimize import curve_fit
import os

# Suppress annoying convergence warnings to keep GUI smooth
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

FEATURE_CSV = os.path.join(CURRENT_DIR, "forensic_features.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

if not os.path.exists(PROB_CSV):
    PROB_CSV = os.path.join(CURRENT_DIR, "probability_results_server.csv")

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
# 2. MODELS
# ==========================================
def inverse_exp_func(prob, a, b):
    return a * np.exp(b * prob)

def get_model(model_type):
    if model_type == 'Linear':
        return make_pipeline(StandardScaler(), LinearRegression())
    elif model_type == 'Polynomial (D2)':
        return make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
    elif model_type == 'MLP':
        # Added 'adaptive' learning rate and more iterations for stability
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=5000, learning_rate='adaptive', random_state=42))
    return None

# ==========================================
# 3. DATA
# ==========================================
def load_and_merge_data():
    if not os.path.exists(FEATURE_CSV) or not os.path.exists(PROB_CSV):
        return None
    df_feats = pd.read_csv(FEATURE_CSV)
    df_probs = pd.read_csv(PROB_CSV)
    df_probs['Model'] = df_probs['Model'].apply(lambda x: x if 'Model' in x else f"Model {x}")
    df_feats['Model'] = df_feats['Model'].astype(str)
    return pd.merge(df_feats, df_probs, on=['Model', 'Step'], how='inner')

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
        self.radio_reg = RadioButtons(ax_reg, ['Linear', 'Polynomial (D2)', 'MLP', 'Exponential'], active=0)
        self.radio_reg.on_clicked(self.set_regressor)

        ax_feat = plt.axes([0.05, 0.1, 0.2, 0.4], facecolor='#f0f0f0')
        self.check_feat = CheckButtons(ax_feat, FEATURES_LIST, [True] + [False]*4)
        self.check_feat.on_clicked(self.set_features)

    def set_size(self, label): self.current_size = label; self.update_plot()
    def set_regressor(self, label): self.selected_model_type = label; self.update_plot()
    def set_features(self, label):
        self.selected_features = [l for l, s in zip(FEATURES_LIST, self.check_feat.get_status()) if s]
        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()
        if not self.selected_features: return
        
        subset = self.df[self.df['Model'] == f"Model {self.current_size}"].sort_values('Step')
        cols = [FEATURE_CONFIG[f] for f in self.selected_features]
        X, y = subset[cols].values, subset['Step'].values

        if self.selected_model_type == 'Exponential':
            if len(cols) == 1 and cols[0] == 'Avg_Probability':
                try:
                    popt, _ = curve_fit(inverse_exp_func, X.flatten(), y, maxfev=5000)
                    y_pred = inverse_exp_func(X.flatten(), *popt)
                    self.ax.scatter(y, y_pred, color='green', label=f'Exp (R²={r2_score(y, y_pred):.3f})')
                except: self.ax.set_title("Exp Fit Failed")
            else: self.ax.set_title("Exp works only with Probability alone")
        else:
            model = get_model(self.selected_model_type)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred = cross_val_predict(model, X, y, cv=cv)
            r2 = np.mean(cross_val_score(model, X, y, cv=cv))
            self.ax.scatter(y, y_pred, color='blue', alpha=0.6, label=f'CV Pred (R²={r2:.3f})')
            self.ax.set_title(f"Size: {self.current_size} | Model: {self.selected_model_type}")

        self.ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', alpha=0.5)
        self.ax.legend(); self.ax.grid(True, alpha=0.3); plt.draw()

if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None: ForensicAnalyzer(df); plt.show()