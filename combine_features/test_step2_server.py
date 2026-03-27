import pandas as pd
import numpy as np
import os
import time
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MATRICES = {
    'Original': os.path.join(BASE_DIR, 'step2_master_matrix.csv'),
    'Interaction': os.path.join(BASE_DIR, 'step2_interaction_matrix.csv')
}

DIRS = {
    'Original': os.path.join(BASE_DIR, 'micro_graphs-step2_original'),
    'Interaction': os.path.join(BASE_DIR, 'micro_graphs-step2_interaction')
}

EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}


# ==========================================
# HELPERS
# ==========================================
def setup_graphs_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📁 Created graphs directory at: {path}")


def get_algorithms():
    return {
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                            random_state=100, early_stopping=True),
        'Ridge': Ridge(alpha=50.0, random_state=42, solver='cholesky'),
        'RandomForest': RandomForestRegressor(n_estimators=50,
                                              random_state=42, n_jobs=8)
    }


def plot_actual_vs_predicted(y_true, y_pred, model_name,
                             algo_name, test_type, filename, save_dir):

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    plt.scatter(
        y_true, y_pred,
        alpha=0.4,
        color='royalblue',
        edgecolors='k',
        label=f'{algo_name} Prediction'
    )

    min_val, max_val = y_true.min(), y_true.max()
    plt.plot([min_val, max_val],
             [min_val, max_val],
             'r--', lw=3,
             label='Ideal Identity Line')

    plt.title(f"Architecture: {model_name} | Algo: {algo_name}\n{test_type}", fontsize=14)
    plt.xlabel('Ground Truth (Actual Training Steps)', fontsize=12)
    plt.ylabel('Model Prediction (Estimated Training Steps)', fontsize=12)

    stats_text = f"Accuracy Metrics:\n------------------\nR² Score: {r2:.4f}\nMAE: +/- {mae:.1f} steps"

    plt.gca().text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"{'Model':<12} | {'Algorithm':<15} | {'R²':>8} | {'MAE':>12}")
    print(f"{'-'*55}")


def print_result_row(model_name, algo_name, r2, mae):
    print(f"{model_name:<12} | {algo_name:<15} | {r2:>8.4f} | {mae:>12.1f}")


# ==========================================
# MAIN
# ==========================================
def run_all_micro_tests():
    summary = {}
    start_time = time.time()

    print(f"🚀 Started at: {datetime.datetime.now()}")
    print(f"📂 BASE_DIR: {BASE_DIR}\n")

    for dataset_name, matrix_path in MATRICES.items():
        if not os.path.exists(matrix_path):
            print(f"❌ Missing: {matrix_path}")
            continue

        print("\n" + "#"*80)
        print(f" RUNNING: {dataset_name.upper()} ")
        print("#"*80)

        should_generate_graphs = dataset_name == 'Interaction'
        save_dir = DIRS[dataset_name] if should_generate_graphs else None
        if should_generate_graphs:
            setup_graphs_dir(save_dir)

        df = pd.read_csv(matrix_path)
        feature_cols = [c for c in df.columns if c not in ['Model', 'Step']]
        bin_cols = [c for c in df.columns if c.startswith('Bin_')]
        models = ['7M', '30M', '124M']

        # =========================================================
        # TEST 1: Random Split
        # =========================================================
        print_test_header("TEST 1: Random 80/20 Split")
        for model_name in models:
            df_model = df[df['Model'] == model_name]
            if df_model.empty: continue

            X = df_model[feature_cols].values
            Y = df_model['Step'].values

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for algo_name, algo in get_algorithms().items():
                algo.fit(X_train, y_train)
                y_pred = algo.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                print_result_row(model_name, algo_name, r2, mae)

                if should_generate_graphs:
                    plot_actual_vs_predicted(y_test, y_pred, model_name, algo_name,
                                             "Random Split (Leakage Expected)",
                                             f"test1_{model_name}_{algo_name}.png", save_dir)

        # =========================================================
        # TEST 2: Grouped Clean - 5 Splits + FULL PLOT
        # =========================================================
        print_test_header("TEST 2: Grouped Split (5 splits average)")

        n_splits = 5

        for model_name in models:
            df_model = df[df['Model'] == model_name]
            if df_model.empty: continue

            X = df_model[feature_cols].values
            Y = df_model['Step'].values
            groups = df_model['Step'].values

            model_r2_scores = {'MLP': [], 'Ridge': [], 'RandomForest': []}

            gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

            # Separate collections per algorithm to prevent data mixing across models
            all_y_true = {'MLP': [], 'Ridge': [], 'RandomForest': []}
            all_y_pred = {'MLP': [], 'Ridge': [], 'RandomForest': []}

            for fold_idx, (train_idx, test_idx) in enumerate(gss.split(X, Y, groups)):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])

                algorithms = get_algorithms()
                for algo_name, algo in algorithms.items():
                    algo.fit(X_train, Y[train_idx])
                    y_pred = algo.predict(X_test)

                    r2 = r2_score(Y[test_idx], y_pred)
                    mae = mean_absolute_error(Y[test_idx], y_pred)

                    model_r2_scores[algo_name].append(r2)

                    # Collect predictions specifically for this algorithm
                    all_y_true[algo_name].extend(Y[test_idx])
                    all_y_pred[algo_name].extend(y_pred)

                    # Print only first fold to keep log clean
                    if fold_idx == 0:
                        print_result_row(model_name, algo_name, r2, mae)

            # Save average R² for summary (especially for 124M)
            if model_name == '124M':
                for algo_name in ['MLP', 'Ridge', 'RandomForest']:
                    avg_r2 = np.mean(model_r2_scores[algo_name])
                    summary.setdefault(dataset_name, {}).setdefault(algo_name, {})['Test2_Grouped_124M'] = avg_r2

            # Create FULL plot (all 5 splits combined) for this model
            if should_generate_graphs:
                for algo_name in ['MLP', 'Ridge', 'RandomForest']:
                    plot_actual_vs_predicted(
                        np.array(all_y_true[algo_name]),
                        np.array(all_y_pred[algo_name]),
                        model_name,
                        algo_name,
                        "Clean Grouped Evaluation (All 5 Splits Combined)",
                        f"test2_grouped_{model_name}_{algo_name}_full.png",
                        save_dir
                    )
        # =========================================================
        # TEST 5 (ZERO SHOT)
        # =========================================================
        print_test_header("TEST 5: Zero Shot")

        df_h = df.copy()

        for model_name, n_embd in EMBED_DIM_MAP.items():
            mask = df_h['Model'] == model_name
            df_h.loc[mask, bin_cols] /= np.sqrt(n_embd)

        train_df = df_h[df_h['Model'].isin(['7M', '30M'])]
        test_df = df_h[df_h['Model'] == '124M']

        X_train = train_df[feature_cols].values
        Y_train = train_df['Step'].values
        X_test = test_df[feature_cols].values
        Y_test = test_df['Step'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for algo_name, algo in get_algorithms().items():
            algo.fit(X_train, Y_train)
            y_pred = algo.predict(X_test)

            r2 = r2_score(Y_test, y_pred)
            mae = mean_absolute_error(Y_test, y_pred)

            print_result_row("124M (ZS)", algo_name, r2, mae)

            if should_generate_graphs:
                plot_actual_vs_predicted(
                    Y_test, y_pred,
                    "124M (Zero-Shot)", algo_name,
                    "Train 7M+30M → Test 124M",
                    f"test5_{algo_name}.png",
                    save_dir
                )

    print(f"\n✅ Done in {(time.time()-start_time)/60:.2f} minutes")
    return summary


if __name__ == "__main__":
    run_all_micro_tests()