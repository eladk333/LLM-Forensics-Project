import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ==========================================
# CONFIGURATION & ARCHITECTURE DATA
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MATRIX_PATH = os.path.join(CURRENT_DIR, 'step2_master_matrix.csv')

# Fixed Architecture values based on training scripts
EMBED_DIM_MAP = {'7M': 128, '30M': 384, '124M': 768}
LAYER_MAP = {'7M': 4, '30M': 6, '124M': 12}

def run_all_micro_tests():
    if not os.path.exists(MATRIX_PATH):
        print(f"Error: Master matrix not found at {MATRIX_PATH}")
        return

    print("Loading Master Matrix...")
    df = pd.read_csv(MATRIX_PATH)
    
    # Identify our features
    feature_cols = ['Batch_Probability'] + [col for col in df.columns if col.startswith('Bin_')]
    bin_cols = [col for col in df.columns if col.startswith('Bin_')]
    models = ['7M', '30M', '124M']
    
    def get_mlp():
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)

    # =====================================================================
    # TEST 1: RANDOM 80/20 SPLIT (Shows Data Leakage)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 1: RANDOM 80/20 SPLIT (DATA LEAKAGE EXPECTED)':^85}")
    print("="*85)
    print(f"{'Model':<10} | {'Test R2':<10} | {'Test MAE (Steps)':<18}")
    print("-" * 85)
    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        if df_model.empty: continue
        X = df_model[feature_cols].values
        Y = df_model['Step'].values
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        mlp = get_mlp()
        mlp.fit(scaler.fit_transform(X_train), y_train)
        y_pred = mlp.predict(scaler.transform(X_test))
        print(f"{model_name:<10} | {r2_score(y_test, y_pred):<10.4f} | +/- {mean_absolute_error(y_test, y_pred):<14.1f}")

    # =====================================================================
    # TEST 2: GROUPED 80/20 SPLIT (Clean, No Leakage)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 2: GROUPED BY CHECKPOINT (CLEAN EVALUATION)':^85}")
    print("="*85)
    print(f"{'Model':<10} | {'Test R2':<10} | {'Test MAE (Steps)':<18} | {'Checkpoints (Train/Test)'}")
    print("-" * 85)
    for model_name in models:
        df_model = df[df['Model'] == model_name].copy()
        if df_model.empty: continue
        X = df_model[feature_cols].values
        Y = df_model['Step'].values
        groups = df_model['Step'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, Y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        scaler = StandardScaler()
        mlp = get_mlp()
        mlp.fit(scaler.fit_transform(X_train), y_train)
        y_pred = mlp.predict(scaler.transform(X_test))
        print(f"{model_name:<10} | {r2_score(y_test, y_pred):<10.4f} | +/- {mean_absolute_error(y_test, y_pred):<14.1f} | {len(set(y_train))} Train / {len(set(y_test))} Test")

    # =====================================================================
    # TEST 3: CROSS-MODEL TRANSFER (Unnormalized - Will Fail)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 3: UNNORMALIZED CROSS-ARCHITECTURE (Train: 7M,30M -> Test: 124M)':^85}")
    print("="*85)
    df_train = df[df['Model'].isin(['7M', '30M'])].copy()
    df_test = df[df['Model'] == '124M'].copy()
    if not df_train.empty and not df_test.empty:
        scaler = StandardScaler()
        mlp = get_mlp()
        mlp.fit(scaler.fit_transform(df_train[feature_cols].values), df_train['Step'].values)
        y_pred = mlp.predict(scaler.transform(df_test[feature_cols].values))
        print(f"{'124M (Unseen)':<10} | {r2_score(df_test['Step'].values, y_pred):<10.4f} | +/- {mean_absolute_error(df_test['Step'].values, y_pred):<14.1f}")
    
    # =====================================================================
    # TEST 4: NORMALIZED CROSS-MODEL (Cheating - MinMaxScaler)
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 4: NORMALIZED CROSS-ARCHITECTURE (Cheating - MinMaxScaler)':^85}")
    print("="*85)
    df_norm_cheat = df.copy()
    for model_name in models:
        mask = df_norm_cheat['Model'] == model_name
        if mask.sum() > 0:
            df_norm_cheat.loc[mask, feature_cols] = MinMaxScaler().fit_transform(df_norm_cheat.loc[mask, feature_cols])
    df_train_cheat = df_norm_cheat[df_norm_cheat['Model'].isin(['7M', '30M'])]
    df_test_cheat = df_norm_cheat[df_norm_cheat['Model'] == '124M']
    if not df_train_cheat.empty and not df_test_cheat.empty:
        models_to_compare = {'Ridge (Linear)': Ridge(alpha=1.0), 'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42), 'MLP (Neural Net)': get_mlp()}
        print(f"{'Algorithm':<22} | {'Test R2':<10} | {'Test MAE (Steps)':<18}")
        print("-" * 85)
        for algo_name, algo in models_to_compare.items():
            algo.fit(df_train_cheat[feature_cols].values, df_train_cheat['Step'].values)
            y_pred = algo.predict(df_test_cheat[feature_cols].values)
            print(f"{algo_name:<22} | {r2_score(df_test_cheat['Step'].values, y_pred):<10.4f} | +/- {mean_absolute_error(df_test_cheat['Step'].values, y_pred):<14.1f}")

    # =====================================================================
    # TEST 5: HONEST CROSS-MODEL (Scaling by sqrt(N_EMBD))
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 5: HONEST CROSS-ARCHITECTURE (Scaling by sqrt(N_EMBD))':^85}")
    print("="*85)
    df_honest = df.copy()
    for model_name, n_embd in EMBED_DIM_MAP.items():
        mask = df_honest['Model'] == model_name
        if mask.sum() > 0:
            df_honest.loc[mask, bin_cols] = df_honest.loc[mask, bin_cols] / np.sqrt(n_embd)
    df_train_honest = df_honest[df_honest['Model'].isin(['7M', '30M'])]
    df_test_honest = df_honest[df_honest['Model'] == '124M']
    if not df_train_honest.empty and not df_test_honest.empty:
        scaler = StandardScaler()
        mlp_h = get_mlp()
        mlp_h.fit(scaler.fit_transform(df_train_honest[feature_cols].values), df_train_honest['Step'].values)
        y_pred_h = mlp_h.predict(scaler.transform(df_test_honest[feature_cols].values))
        print(f"{'124M (Honest)':<22} | {r2_score(df_test_honest['Step'].values, y_pred_h):<10.4f} | +/- {mean_absolute_error(df_test_honest['Step'].values, y_pred_h):<14.1f}")

    # =====================================================================
    # TEST 6: ADVANCED HONEST CROSS-MODEL (Scaling by sqrt(N_EMBD * N_LAYER))
    # =====================================================================
    print("\n" + "="*85)
    print(f"{'TEST 6: ADVANCED HONEST SCALING (sqrt(N_EMBD * N_LAYER))':^85}")
    print("="*85)
    df_adv = df.copy()
    for model_name in models:
        n_embd = EMBED_DIM_MAP[model_name]
        n_layer = LAYER_MAP[model_name]
        mask = df_adv['Model'] == model_name
        if mask.sum() > 0:
            df_adv.loc[mask, bin_cols] = df_adv.loc[mask, bin_cols] / np.sqrt(n_embd * n_layer)
    df_train_adv = df_adv[df_adv['Model'].isin(['7M', '30M'])]
    df_test_adv = df_adv[df_adv['Model'] == '124M']
    if not df_train_adv.empty and not df_test_adv.empty:
        scaler = StandardScaler()
        mlp_adv = get_mlp()
        mlp_adv.fit(scaler.fit_transform(df_train_adv[feature_cols].values), df_train_adv['Step'].values)
        y_pred_adv = mlp_adv.predict(scaler.transform(df_test_adv[feature_cols].values))
        print(f"{'124M (Advanced)':<22} | {r2_score(df_test_adv['Step'].values, y_pred_adv):<10.4f} | +/- {mean_absolute_error(df_test_adv['Step'].values, y_pred_adv):<14.1f}")

    print("-" * 85 + "\nAll Tests Complete.\n")

if __name__ == "__main__":
    run_all_micro_tests()