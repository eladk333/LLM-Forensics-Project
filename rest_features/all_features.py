import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

"""
=====================================================================================
LLM FORENSICS: ZERO-SHOT EXTRAPOLATION EXPERIMENT (7M+30M -> 124M)
=====================================================================================
"""

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "zero_shot_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REST_CSV = os.path.join(CURRENT_DIR, "extracted_rest_features.csv")
EMB_CSV = os.path.join(ROOT_DIR, "gradient-pretiction-via-embedding_norm", "embedding_norms_empirical.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

if not os.path.exists(PROB_CSV):
    PROB_CSV = os.path.join(ROOT_DIR, "gradient-pretiction-via-next-token-prediction", "probability_results_server.csv")

# Constants for normalization 
EMBED_DIM_MAP = {'Model 7M': 128, 'Model 30M': 384, 'Model 124M': 768}
VOCAB_SIZE = 50257

# ==========================================
# 2. DATA LOADING & MERGING
# ==========================================
def clean_model_name(val):
    s = str(val).strip()
    import re
    match = re.search(r'(\d+)\s*[mM]', s)
    if match:
        return f"Model {match.group(1)}M"
    match_num = re.search(r'(\d+)', s)
    if match_num:
        return f"Model {match_num.group(1)}M"
    return s

def load_and_merge_data():
    print("🔄 Loading data from CSV files...")
    
    if not all(os.path.exists(p) for p in [REST_CSV, EMB_CSV, PROB_CSV]):
        print("❌ Error: One or more CSV files are missing. Check paths.")
        return None

    df_rest = pd.read_csv(REST_CSV)
    df_emb = pd.read_csv(EMB_CSV)
    df_prob = pd.read_csv(PROB_CSV)

    for name, df in [('Rest', df_rest), ('Embedding', df_emb), ('Probability', df_prob)]:
        if 'Model' in df.columns:
            df['Model'] = df['Model'].apply(clean_model_name)
        else:
            return None
            
        if 'Step' in df.columns:
            df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
            df.dropna(subset=['Step'], inplace=True)
            df['Step'] = df['Step'].astype(int)
        else:
            return None

    if 'Method' in df_emb.columns:
        df_emb = df_emb[df_emb['Method'].astype(str).str.strip().str.lower() == 'global']

    if 'embedding_norm' in df_emb.columns:
        emb_col = 'embedding_norm'
    elif 'Feature_Value' in df_emb.columns:
        emb_col = 'Feature_Value'
    else:
        emb_col = df_emb.columns[-1] 
        
    df_emb[emb_col] = pd.to_numeric(df_emb[emb_col], errors='coerce')
    df_emb.dropna(subset=[emb_col], inplace=True)
    
    df_prob_sub = df_prob[['Model', 'Step', 'Avg_Probability']].drop_duplicates()
    df_emb_sub = df_emb[['Model', 'Step', emb_col]].drop_duplicates()
    df_rest_sub = df_rest.drop_duplicates()

    df_merged = pd.merge(df_rest_sub, df_prob_sub, on=['Model', 'Step'], how='inner')
    df_merged = pd.merge(df_merged, df_emb_sub, on=['Model', 'Step'], how='inner')
    df_merged.rename(columns={emb_col: 'embedding_norm'}, inplace=True)
    
    return df_merged

# ==========================================
# 3. RANK NORMALIZATION
# ==========================================
def normalize_structural_features(df):
    df_norm = df.copy()
    
    for m_name, dim in EMBED_DIM_MAP.items():
        mask = df_norm['Model'] == m_name
        if mask.any():
            df_norm.loc[mask, 'embedding_norm'] = df_norm.loc[mask, 'embedding_norm'] / np.sqrt(dim)
            df_norm.loc[mask, 'l1_norm'] = df_norm.loc[mask, 'l1_norm'] / (VOCAB_SIZE * dim)
            df_norm.loc[mask, 'logit_norm'] = df_norm.loc[mask, 'logit_norm'] / np.sqrt(VOCAB_SIZE * dim)
            
    return df_norm

# ==========================================
# 4. MODELING UTILITIES
# ==========================================
def get_ml_models(is_single_feature=False):
    models = {
        'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Polynomial (Degree 2)': make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression())
    }
    
    if not is_single_feature:
        models.update({
            'Ridge (L2)': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            'Random Forest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
            'MLP (Neural Net)': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=5000, random_state=42))
        })
    return models

def add_stats_box(ax, r2, mae, model_name):
    stats_text = f"Model: {model_name}\n----------------\nR² Score: {r2:.4f}\nMAE: ±{mae:.1f} steps"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ==========================================
# 5. ZERO-SHOT EXTRAPOLATION
# ==========================================
def run_zero_shot():
    df_raw = load_and_merge_data()
    if df_raw is None or len(df_raw) == 0:
        print("❌ Experiment stopped: Merged dataset is empty.")
        return
        
    df_scaled = normalize_structural_features(df_raw)
    
    train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
    test = df_scaled[df_scaled['Model'] == 'Model 124M']
    
    if len(train) == 0 or len(test) == 0:
        print("❌ Error: Train or Test split is empty!")
        return
        
    y_train = train['Step'].values
    y_test = test['Step'].values
    
    scenarios = {
        "ALL_5_FEATURES": ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance', 'logit_norm'],
        "WITHOUT_LOGIT_NORM": ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance'],
        "ONLY_WEIGHT_VARIANCE": ['weight_variance'],
        "ONLY_L1_NORM": ['l1_norm']
    }
    
    all_available_features = ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance', 'logit_norm']
    excluded_sets = [{'embedding_norm'}, {'embedding_norm', 'Avg_Probability'}]
    feat_acronyms = {
        'embedding_norm': 'EMB', 'Avg_Probability': 'PROB', 'l1_norm': 'L1',
        'weight_variance': 'VAR', 'logit_norm': 'LOGIT'
    }

    for r in range(1, len(all_available_features) + 1):
        for combo in itertools.combinations(all_available_features, r):
            combo_set = set(combo)
            if combo_set in excluded_sets:
                continue
            if any(set(existing_features) == combo_set for existing_features in scenarios.values()):
                continue
                
            scenario_name = "COMBO_" + "_".join([feat_acronyms[f] for f in combo])
            scenarios[scenario_name] = list(combo)

    final_results = []
    
    print(f"\nEvaluating {len(scenarios)} different feature combinations (No plotting yet)...")
    
    for scenario_name, features in scenarios.items():
        # הוחזר הפלט המפורט המקורי לכל ריצה
        print(f"\n{'='*70}")
        print(f"🚀 ZERO-SHOT SCENARIO: {scenario_name}")
        print(f"Features Used: {len(features)} {features}")
        print(f"{'='*70}")
        print(f"{'ML Algorithm':<25} | {'R² Score':<10} | {'MAE (Steps)':<12}")
        print("-" * 55)
        
        X_train = train[features].values
        X_test = test[features].values
        
        is_single = len(features) == 1
        ml_models = get_ml_models(is_single_feature=is_single)
        
        for model_name, model in ml_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # הדפסת התוצאות לכל אלגוריתם בנפרד כפי שהיה במקור
            print(f"{model_name:<25} | {r2:<10.4f} | {mae:<12.2f}")
            
            # שמירת הנתונים לשרטוט מאוחר יותר של הטופ 4
            final_results.append({
                'display_name': f"{scenario_name} ({model_name})",
                'r2': r2,
                'mae': mae,
                'scenario_name': scenario_name,
                'model_name': model_name,
                'y_pred': y_pred
            })

    print(f"\n{'='*80}\n🏆 GLOBAL ZERO-SHOT MODEL RANKING (BY LOWEST MAE)\n{'='*80}")
    final_results.sort(key=lambda x: x['mae'])
    
    print(f"{'Rank':<5} | {'Scenario (Model)':<55} | {'MAE':<10} | {'R²':<8}")
    print("-" * 90)
    for idx, item in enumerate(final_results, 1):
        print(f"#{idx:<4} | {item['display_name']:<55} | {item['mae']:<10.1f} | {item['r2']:<8.4f}")

    # יצירה ושמירה של גרפים אך ורק עבור 4 התוצאות הטובות ביותר
    print(f"\n📸 Generating and saving plots ONLY for the Top 4 models...")
    for idx, item in enumerate(final_results[:4], 1):
        r2 = item['r2']
        mae = item['mae']
        model_name = item['model_name']
        scenario_name = item['scenario_name']
        y_pred = item['y_pred']
        
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(y_test, y_pred, color='darkmagenta', alpha=0.7, label="124M Predictions")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Extrapolation (Y=X)")
        
        ax.set_title(f"Top {idx}: Zero-Shot Extrapolation [7M+30M -> 124M]\nScenario: {scenario_name}")
        ax.set_xlabel("Actual Steps (124M Odometer)")
        ax.set_ylabel("Predicted Steps by ML Algorithm")
        add_stats_box(ax, r2, mae, model_name)
        ax.legend(loc='lower right')
        
        clean_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"Top_{idx}_zs_{scenario_name}_{clean_model_name}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        
        print(f"Saved: {filename}")

if __name__ == "__main__":
    run_zero_shot()