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

# ──────────────────────────────────────────────
# PATHS & CONFIG
# ──────────────────────────────────────────────
MODEL_SIZES = ['7M', '30M', '124M']
BASE_MODELS_FOLDER = r'G:\My Drive\llm\data\models'
FREQ_FILE          = r'G:\My Drive\llm\data\datasets\wiki\wiki_token_frequencies.csv'

FEATURE_CONFIG = {
    'embedding_norm':  'Embedding Norm',
    'logit_norm':      'Logit Norm',
    'weight_variance': 'Weight Variance',
    'weight_mean':     'Weight Mean',
    'l1_norm':         'L1 Norm',
    'token_len':       'Token Length',
    'is_upper':        'Is Capitalized',
}
FEATURES_LIST = list(FEATURE_CONFIG.keys())

# ──────────────────────────────────────────────
# STORAGE  (populated at startup)
# global_dataframes[size]  -> {df, y, valid_features}
# results[train_size][test_size] -> {lin_r2, mlp_r2, y_test, lin_pred, mlp_pred,
#                                    y_test_raw, lin_pred_raw, mlp_pred_raw, X_test}
# ──────────────────────────────────────────────
global_dataframes = {}
results = {}   # results[train_model][test_model]

# ──────────────────────────────────────────────
# ML HELPERS
# ──────────────────────────────────────────────

def _fit_and_evaluate(X_train, y_train, X_test, y_test):
    """Train Linear + MLP on (X_train, y_train) and evaluate on (X_test, y_test)."""
    if X_train.shape[1] == 0:
        return None

    # ── Linear ──
    lin = make_pipeline(StandardScaler(), LinearRegression())
    lin.fit(X_train, y_train)
    lin_pred_raw = lin.predict(X_test)
    lin_r2 = r2_score(y_test, lin_pred_raw)

    # ── MLP ──
    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='tanh',
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
    )
    mlp.fit(X_train, y_train)
    mlp_pred_raw = mlp.predict(X_test)
    mlp_r2 = r2_score(y_test, mlp_pred_raw)

    # Sort by first feature when only one feature (nicer scatter lines)
    if X_test.shape[1] == 1:
        idx = X_test.iloc[:, 0].argsort()
        return {
            'X_test':        X_test.iloc[idx],
            'y_test':        y_test.iloc[idx],
            'lin_pred':      lin_pred_raw[idx],
            'mlp_pred':      mlp_pred_raw[idx],
            'lin_pred_raw':  lin_pred_raw,
            'mlp_pred_raw':  mlp_pred_raw,
            'y_test_raw':    y_test,
            'lin_r2':        lin_r2,
            'mlp_r2':        mlp_r2,
        }
    return {
        'X_test':       X_test,
        'y_test':       y_test,
        'lin_pred':     lin_pred_raw,
        'mlp_pred':     mlp_pred_raw,
        'lin_pred_raw': lin_pred_raw,
        'mlp_pred_raw': mlp_pred_raw,
        'y_test_raw':   y_test,
        'lin_r2':       lin_r2,
        'mlp_r2':       mlp_r2,
    }


def train_all():
    """
    For every train_model in MODEL_SIZES:
        - train the combined model on train_model's data (80/20 self-split)
        - test on train_model itself  (the held-out 20 %)
        - test on every other model   (full dataset as test set)

    results[train_model][test_model] = result_dict
    """
    for train_size in MODEL_SIZES:
        if train_size not in global_dataframes:
            print(f"  ⚠ {train_size} data missing — skipping as train source.")
            continue

        results[train_size] = {}

        gd_train      = global_dataframes[train_size]
        df_train_full = gd_train['df']
        y_train_full  = gd_train['y']
        feats_train   = gd_train['valid_features']

        # Split training data once (80/20) — the 80 % is ONLY used for fitting
        X_all = df_train_full[feats_train]
        X_tr, X_held, y_tr, y_held = train_test_split(
            X_all, y_train_full, test_size=0.2, random_state=42
        )

        # ── self-test (held-out 20 %) ──────────────────────────────────────
        print(f"  Training on {train_size}, testing on {train_size} (held-out 20 %)...")
        results[train_size][train_size] = _fit_and_evaluate(X_tr, y_tr, X_held, y_held)
        r = results[train_size][train_size]
        if r:
            print(f"    Lin R2={r['lin_r2']:.4f}  MLP R2={r['mlp_r2']:.4f}")

        # ── cross-model tests ──────────────────────────────────────────────
        for test_size in MODEL_SIZES:
            if test_size == train_size:
                continue
            if test_size not in global_dataframes:
                print(f"  ⚠ {test_size} data missing — skipping as test target.")
                continue

            gd_test     = global_dataframes[test_size]
            df_test_all = gd_test['df']
            y_test_all  = gd_test['y']
            feats_test  = gd_test['valid_features']

            # Use only features available in BOTH models
            common = [f for f in FEATURES_LIST if f in set(feats_train) and f in set(feats_test)]
            if not common:
                print(f"  ⚠ No common features between {train_size} and {test_size}.")
                results[train_size][test_size] = None
                continue

            # Re-fit on 80 % of train_model using only common features
            X_tr_common   = df_train_full[common].iloc[X_tr.index]
            X_test_common = df_test_all[common]          # entire other model as test set

            print(f"  Training on {train_size}, testing on {test_size} ({len(common)} common features)...")
            results[train_size][test_size] = _fit_and_evaluate(
                X_tr_common, y_tr, X_test_common, y_test_all
            )
            r = results[train_size][test_size]
            if r:
                print(f"    Lin R2={r['lin_r2']:.4f}  MLP R2={r['mlp_r2']:.4f}")


# ──────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────

class CrossModelViewer:
    """
    Two radio-button selectors:
      • Train Model  — which model the predictor was trained on
      • Test  Model  — which model it is evaluated on
    Two graph view modes:
      • Pred vs Actual  — scatter of predicted vs true log-frequency
      • R2 Summary      — bar chart comparing Lin vs MLP across all 9 combos
    """

    VIEW_OPTIONS = ['Pred vs Actual', 'R2 Summary']

    def __init__(self):
        self.train_models = [s for s in MODEL_SIZES if s in results]
        self.test_models  = MODEL_SIZES[:]

        self.current_train = self.train_models[0]
        self.current_test  = self.train_models[0]   # default: self-test
        self.current_view  = 'Pred vs Actual'

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Cross-Model Frequency Prediction Analysis")
        self.update()

    # ── rendering ──────────────────────────────────────────────────────────

    def _draw_pred_vs_actual(self):
        """Two scatter plots: Linear and MLP predicted vs actual log-frequency."""
        data = results.get(self.current_train, {}).get(self.current_test)

        ax_area = self.fig.add_axes([0.25, 0.05, 0.72, 0.88])
        ax_area.axis('off')

        if data is None:
            ax_area.text(
                0.5, 0.5,
                f"No results available for\nTrain={self.current_train}  →  Test={self.current_test}",
                ha='center', va='center', fontsize=14, transform=ax_area.transAxes
            )
            return

        y_true    = data['y_test_raw']
        lin_pred  = data['lin_pred_raw']
        mlp_pred  = data['mlp_pred_raw']
        lin_r2    = data['lin_r2']
        mlp_r2    = data['mlp_r2']

        is_self = (self.current_train == self.current_test)
        test_label  = f"{self.current_test} (held-out 20 %)" if is_self else f"{self.current_test} (full dataset)"
        cross_note  = "" if is_self else "  ←  trained on a DIFFERENT model"

        ax1 = self.fig.add_axes([0.27, 0.12, 0.31, 0.70])
        ax2 = self.fig.add_axes([0.63, 0.12, 0.31, 0.70])

        def _diagonal(ax, y_true, y_pred):
            mn = min(y_true.min(), y_pred.min())
            mx = max(y_true.max(), y_pred.max())
            ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')

        # ── Linear ──
        sns.scatterplot(x=y_true, y=lin_pred, ax=ax1, alpha=0.25, color='steelblue', s=12)
        _diagonal(ax1, y_true, lin_pred)
        ax1.set_xlabel("Actual Log-Frequency of Token in WikiText-103", fontsize=10)
        ax1.set_ylabel("Predicted Log-Frequency", fontsize=10)
        ax1.set_title(
            f"Linear Regression  (R² = {lin_r2:.3f})\n"
            f"Trained on {self.current_train} → Tested on {test_label}{cross_note}",
            fontsize=10, pad=8
        )
        ax1.set_box_aspect(1)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)

        # ── MLP ──
        sns.scatterplot(x=y_true, y=mlp_pred, ax=ax2, alpha=0.25, color='seagreen', s=12)
        _diagonal(ax2, y_true, mlp_pred)
        ax2.set_xlabel("Actual Log-Frequency of Token in WikiText-103", fontsize=10)
        ax2.set_ylabel("Predicted Log-Frequency", fontsize=10)
        ax2.set_title(
            f"MLP Neural Network  (R² = {mlp_r2:.3f})\n"
            f"Trained on {self.current_train} → Tested on {test_label}{cross_note}",
            fontsize=10, pad=8
        )
        ax2.set_box_aspect(1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)

        self.fig.suptitle(
            f"How Well Do GPT Embedding Features Predict Token Frequency?\n"
            f"Predictor trained on {self.current_train}  →  evaluated on {self.current_test}",
            fontsize=13, fontweight='bold', y=0.98
        )

    def _draw_r2_summary(self):
        """
        Bar chart — for the currently selected TRAIN model, show R2 across all test targets.
        Groups: self-test / 7M / 30M / 124M  |  bars: Linear, MLP
        """
        ax = self.fig.add_axes([0.28, 0.12, 0.68, 0.74])

        train_results = results.get(self.current_train, {})
        test_labels   = []
        lin_r2_vals   = []
        mlp_r2_vals   = []

        for test_size in MODEL_SIZES:
            data = train_results.get(test_size)
            if data is None:
                continue
            is_self = (test_size == self.current_train)
            label = f"{test_size}\n(self-test\nheld-out 20 %)" if is_self else f"{test_size}\n(full dataset\ncross-model)"
            test_labels.append(label)
            lin_r2_vals.append(data['lin_r2'])
            mlp_r2_vals.append(data['mlp_r2'])

        if not test_labels:
            ax.text(0.5, 0.5, "No results to display.", ha='center', va='center')
            return

        x      = np.arange(len(test_labels))
        width  = 0.30
        bars1  = ax.bar(x - width/2, lin_r2_vals, width, label='Linear Regression', color='steelblue', alpha=0.85)
        bars2  = ax.bar(x + width/2, mlp_r2_vals, width, label='MLP Neural Network', color='seagreen',  alpha=0.85)

        # R2 value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, color='steelblue', fontweight='bold')
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=9, color='seagreen', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(test_labels, fontsize=10)
        ax.set_ylabel("R² Score  (1.0 = perfect prediction)", fontsize=11)
        ax.set_ylim(0, min(1.05, max(mlp_r2_vals + lin_r2_vals) + 0.12))
        ax.axhline(0, color='black', lw=0.8)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.35)

        ax.set_title(
            f"R² Summary: Predictor Trained on {self.current_train} Weights\n"
            f"Does a model trained on {self.current_train} embedding statistics generalize\n"
            f"to predict token frequencies on larger/smaller GPT models?",
            fontsize=11, pad=10
        )

        self.fig.suptitle(
            f"Cross-Model Generalization of Embedding-Based Frequency Prediction\n"
            f"Training model: {self.current_train}  |  All features: {', '.join(FEATURES_LIST[:5])} …",
            fontsize=12, fontweight='bold', y=0.98
        )

    # ── sidebar controls ────────────────────────────────────────────────────

    def _draw_controls(self):
        # 1. View type
        ax_view = self.fig.add_axes([0.01, 0.78, 0.18, 0.14], facecolor='#e8eaf6')
        ax_view.set_title("Graph Type", weight='bold', fontsize=9)
        self.rad_view = RadioButtons(
            ax_view, self.VIEW_OPTIONS,
            active=self.VIEW_OPTIONS.index(self.current_view)
        )
        self.rad_view.on_clicked(self._on_view)

        # 2. Train model
        ax_train = self.fig.add_axes([0.01, 0.55, 0.18, 0.18], facecolor='#e8f5e9')
        ax_train.set_title("Train Model\n(predictor fitted on)", weight='bold', fontsize=9)
        self.rad_train = RadioButtons(
            ax_train, self.train_models,
            active=self.train_models.index(self.current_train)
        )
        self.rad_train.on_clicked(self._on_train)

        # 3. Test model (only relevant for Pred vs Actual view)
        if self.current_view == 'Pred vs Actual':
            ax_test = self.fig.add_axes([0.01, 0.28, 0.18, 0.22], facecolor='#fff3e0')
            ax_test.set_title("Test Model\n(evaluated on)", weight='bold', fontsize=9)
            self.rad_test = RadioButtons(
                ax_test, self.test_models,
                active=self.test_models.index(self.current_test)
                       if self.current_test in self.test_models else 0
            )
            self.rad_test.on_clicked(self._on_test)

            # Legend note
            self.fig.text(
                0.01, 0.23,
                "ℹ  Same train/test model\n   = held-out 20% split\n\n"
                "   Different model\n   = full dataset used\n   as test set",
                fontsize=8, va='top', color='#555555'
            )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_view(self, label):
        self.current_view = label
        self.update()

    def _on_train(self, label):
        self.current_train = label
        self.update()

    def _on_test(self, label):
        self.current_test = label
        self.update()

    # ── main update ─────────────────────────────────────────────────────────

    def update(self):
        self.fig.clf()
        plt.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.07)

        if self.current_view == 'Pred vs Actual':
            self._draw_pred_vs_actual()
        else:
            self._draw_r2_summary()

        self._draw_controls()
        self.fig.canvas.draw_idle()


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == '__main__':

    # ── Load frequency file ──
    if not os.path.exists(FREQ_FILE):
        print(f"Frequency file not found: {FREQ_FILE}")
        exit(1)
    df_freq = pd.read_csv(FREQ_FILE)
    print(f"Loaded frequency file ({len(df_freq):,} tokens).")

    # ── Load each model's feature CSV ──
    for size in MODEL_SIZES:
        feat_file = os.path.join(
            BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'model_features.csv'
        )
        if not os.path.exists(feat_file):
            print(f"  ⚠ Feature file not found for {size}: {feat_file}  — skipping.")
            continue

        df_feat = pd.read_csv(feat_file)
        valid   = [f for f in FEATURES_LIST if f in df_feat.columns]
        df      = pd.merge(df_freq, df_feat, on='token_id', how='inner')
        y       = df['log_count']
        global_dataframes[size] = {'df': df, 'y': y, 'valid_features': valid}
        print(f"  Loaded {size}: {len(df):,} tokens, {len(valid)} features.")

    if not global_dataframes:
        print("No model data loaded. Exiting.")
        exit(1)

    # ── Train all combos ──
    print("\n── Training phase ──────────────────────────────")
    train_all()
    print("── Training complete ───────────────────────────\n")

    # ── Launch GUI ──
    viewer = CrossModelViewer()
    plt.show()