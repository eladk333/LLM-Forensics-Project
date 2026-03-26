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
# Added Zero-Shot mode
MODEL_SIZES = ['7M', '30M', '124M', 'Zero-Shot (->124M)']
EMBED_DIM_MAP = {'Model 7M': 128, 'Model 30M': 384, 'Model 124M': 768}

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
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=5000, learning_rate='adaptive', random_state=42))
    return None

# ==========================================
# 3. DATA & GUI
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
        ax_size = plt.axes([0.05, 0.70, 0.20, 0.20], facecolor='#f0f0f0')
        self.radio_size = RadioButtons(ax_size, MODEL_SIZES, active=2)
        self.radio_size.on_clicked(self.set_size)

        ax_reg = plt.axes([0.05, 0.50, 0.20, 0.15], facecolor='#f0f0f0')
        self.radio_reg = RadioButtons(ax_reg, ['Linear', 'Polynomial (D2)', 'MLP', 'Exponential'], active=0)
        self.radio_reg.on_clicked(self.set_regressor)

        ax_feat = plt.axes([0.05, 0.1, 0.25, 0.35], facecolor='#f0f0f0')
        self.check_feat = CheckButtons(ax_feat, FEATURES_LIST, [True] + [False]*4)
        self.check_feat.on_clicked(self.set_features)

    def set_size(self, label): self.current_size = label; self.update_plot()
    def set_regressor(self, label): self.selected_model_type = label; self.update_plot()
    def set_features(self, label):
        self.selected_features = [l for l, s in zip(FEATURES_LIST, self.check_feat.get_status()) if s]
        self.update_plot()

    def scale_features(self, df_subset, cols):
        """Applies physics-based scaling across architectures for Zero-Shot"""
        df_scaled = df_subset.copy()
        if 'embedding_norm' in cols:
            for m_name, n_embd in EMBED_DIM_MAP.items():
                mask = df_scaled['Model'] == m_name
                df_scaled.loc[mask, 'embedding_norm'] = df_scaled.loc[mask, 'embedding_norm'] / np.sqrt(n_embd)
        return df_scaled

    def update_plot(self, event=None):
        # ----------------------------------------------------
        # SAFETY NET: Try/Except to prevent GUI from freezing
        # ----------------------------------------------------
        try:
            self.ax.clear()
            if not self.selected_features:
                self.ax.set_title("⚠️ Please select at least one feature.", color='red')
                plt.draw()
                return
            
            cols = [FEATURE_CONFIG[f] for f in self.selected_features]
            
                        # ----------------------------------------------------
            # ZERO-SHOT LOGIC
            # ----------------------------------------------------
            if self.current_size == 'Zero-Shot (->124M)':
                
                cols = [FEATURE_CONFIG[f] for f in self.selected_features]
                
                # Allow Exponential ONLY when using Probability alone
                if self.selected_model_type == 'Exponential':
                    if len(cols) == 1 and cols[0] == 'Avg_Probability':
                        # Exponential Zero-Shot on Probability only
                        df_train = self.df[self.df['Model'].isin(['Model 7M', 'Model 30M'])]
                        df_test = self.df[self.df['Model'] == 'Model 124M']
                        
                        X_train = df_train['Avg_Probability'].values.reshape(-1, 1)
                        y_train = df_train['Step'].values
                        X_test = df_test['Avg_Probability'].values.reshape(-1, 1)
                        y_test = df_test['Step'].values
                        
                        # Fit exponential on training data (7M + 30M)
                        popt, _ = curve_fit(inverse_exp_func, X_train.flatten(), y_train, maxfev=10000, bounds=(0, [np.inf, np.inf]))
                        
                        y_pred = inverse_exp_func(X_test.flatten(), *popt)
                        r2 = r2_score(y_test, y_pred)
                        
                        self.ax.scatter(y_test, y_pred, color='purple', alpha=0.7, 
                                      label=f'Zero-Shot Exp (R²={r2:.3f})')
                        self.ax.set_title("Zero-Shot: Train 7M+30M -> Test 124M | Exponential on Probability only")
                        y_plot = y_test
                    else:
                        self.ax.set_title("⚠️ Exponential in Zero-Shot works only with Probability alone", color='red')
                        plt.draw()
                        return
                
                # Regular Linear / Polynomial / MLP for Zero-Shot (with normalization)
                else:
                    df_scaled = self.scale_features(self.df, cols)
                    df_train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
                    df_test = df_scaled[df_scaled['Model'] == 'Model 124M']
                    
                    X_train, y_train = df_train[cols].values, df_train['Step'].values
                    X_test, y_test = df_test[cols].values, df_test['Step'].values
                    
                    model = get_model(self.selected_model_type)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    self.ax.scatter(y_test, y_pred, color='purple', alpha=0.7, 
                                  label=f'Zero-Shot Pred (R²={r2:.3f})')
                    self.ax.set_title(f"Zero-Shot: Train 7M+30M -> Test 124M | Model: {self.selected_model_type}")
                    y_plot = y_test
            # ----------------------------------------------------
            # INTERNAL ARCHITECTURE LOGIC
            # ----------------------------------------------------
            else:
                subset = self.df[self.df['Model'] == f"Model {self.current_size}"].sort_values('Step')
                X, y = subset[cols].values, subset['Step'].values

                if self.selected_model_type == 'Exponential':
                    if len(cols) == 1 and cols[0] == 'Avg_Probability':
                        popt, _ = curve_fit(inverse_exp_func, X.flatten(), y, maxfev=5000)
                        y_pred = inverse_exp_func(X.flatten(), *popt)
                        self.ax.scatter(y, y_pred, color='green', label=f'Exp (R²={r2_score(y, y_pred):.3f})')
                        self.ax.set_title(f"Size: {self.current_size} | Exponential Fit")
                    else:
                        self.ax.set_title("⚠️ Exp works only with Probability alone", color='red')
                        plt.draw()
                        return
                    y_plot = y
                else:
                    model = get_model(self.selected_model_type)
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    y_pred = cross_val_predict(model, X, y, cv=cv)
                    r2 = np.mean(cross_val_score(model, X, y, cv=cv))
                    self.ax.scatter(y, y_pred, color='blue', alpha=0.6, label=f'5-Fold CV Pred (R²={r2:.3f})')
                    self.ax.set_title(f"Size: {self.current_size} | Model: {self.selected_model_type}")
                    y_plot = y

            self.ax.plot([y_plot.min(), y_plot.max()], [y_plot.min(), y_plot.max()], 'r--', alpha=0.5)
            self.ax.legend(); self.ax.grid(True, alpha=0.3)
            plt.draw()

        except Exception as e:
            # If anything fails, catch it, show it on screen, and don't freeze!
            self.ax.clear()
            self.ax.set_title("⚠️ Error Executing Request", color='red', fontsize=14, weight='bold')
            error_msg = f"An operation failed:\n{str(e)}\n\nTry changing your model or feature selection."
            self.ax.text(0.5, 0.5, error_msg, ha='center', va='center', fontsize=12, 
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
            plt.draw()

if __name__ == "__main__":
    df = load_and_merge_data()
    if df is not None: ForensicAnalyzer(df); plt.show()