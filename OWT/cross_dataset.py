import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# Suppress convergence and optimization warnings
warnings.filterwarnings("ignore")

"""
=====================================================================================
LLM FORENSICS: CROSS-DATASET ZERO-SHOT EXPERIMENT (WIKI -> OWT)
=====================================================================================
"""

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "cross_dataset_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIKI_FILE = os.path.join(CURRENT_DIR, "wiki_master_features.csv")
OWT_FILE = os.path.join(CURRENT_DIR, "owt_master_features.csv")

# Architecture constants for rank-normalization mapping
EMBED_DIM_MAP = {'Model 7M': 128, 'Model 30M': 384, 'Model 124M': 768}
VOCAB_SIZE = 50257

# ==========================================
# 2. NORMALIZATION (Architecture-Agnostic)
# ==========================================
def normalize_df(df):
    """
    Applies Rank-Normalization formulas to project absolute 
    tensor metrics into an architecture-agnostic scale space.
    """
    df_n = df.copy()
    for model_name, dim in EMBED_DIM_MAP.items():
        mask = df_n['Model'] == model_name
        if mask.any():
            # Divide Frobenius norm by sqrt(d_model)
            df_n.loc[mask, 'embedding_norm'] = df_n.loc[mask, 'embedding_norm'] / np.sqrt(dim)
            # Divide L1 norm by total parameters matrix volume (V * d_model)
            df_n.loc[mask, 'l1_norm'] = df_n.loc[mask, 'l1_norm'] / (VOCAB_SIZE * dim)
            # Divide Logit Frobenius norm by sqrt(V * d_model)
            df_n.loc[mask, 'logit_norm'] = df_n.loc[mask, 'logit_norm'] / np.sqrt(VOCAB_SIZE * dim)
    return df_n

# ==========================================
# 3. MODELING UTILITIES
# ==========================================
def get_ml_models(is_single_feature=False):
    """
    Returns standard scikit-learn regressor pipelines.
    Restricts multi-variable models if input space dimension is 1.
    """
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

def add_stats_box(ax, r2, mae, model_name, features):
    """
    Renders an academic bounding box detailing metrics and configuration.
    """
    features_str = ", ".join([f[:5] for f in features])
    stats_text = f"Model: {model_name}\nFeatures: {features_str}\n----------------\nR² Score: {r2:.4f}\nMAE: ±{mae:.1f} steps"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ==========================================
# 4. CROSS-DATASET EXTRAPOLATION ENGINE
# ==========================================
def run_cross_dataset_zero_shot():
    print("🔄 Loading and normalizing data from CSV files...")
    
    if not os.path.exists(WIKI_FILE) or not os.path.exists(OWT_FILE):
        print("❌ Error: Master CSV files are missing. Check paths.")
        return
        
    # Standardize data distributions dynamically prior to training/inference
    df_wiki = normalize_df(pd.read_csv(WIKI_FILE))
    df_owt = normalize_df(pd.read_csv(OWT_FILE))
    
    # Baseline anchor scenarios matching all_features.py setup
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

    # Generate complete combinations combinatorics to guarantee no blindspots
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
        print(f"\n{'='*70}")
        print(f"🚀 ZERO-SHOT SCENARIO: {scenario_name}")
        print(f"Features Used: {len(features)} {features}")
        print(f"{'='*70}")
        print(f"{'ML Algorithm':<25} | {'R² Score':<10} | {'MAE (Steps)':<12}")
        print("-" * 55)
        
        # Isolation: Train entirely on Wikipedia, test entirely on OpenWebText
        X_train = df_wiki[features].values
        y_train = df_wiki['Step'].values
        X_test = df_owt[features].values
        y_test = df_owt['Step'].values
        
        # Track original model strings for per-architecture plot coloring
        models_test = df_owt['Model'].values
        
        is_single = len(features) == 1
        ml_models = get_ml_models(is_single_feature=is_single)
        
        for model_name, model in ml_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{model_name:<25} | {r2:<10.4f} | {mae:<12.2f}")
            
            # --- NEW: Extract Feature Importance for Random Forest models ---
            rf_importances = None
            if model_name == 'Random Forest':
                rf_importances = model.named_steps['randomforestregressor'].feature_importances_.copy()
            # -------------------------------------------------------------
            
            final_results.append({
                'display_name': f"{scenario_name} ({model_name})",
                'r2': r2,
                'mae': mae,
                'scenario_name': scenario_name,
                'model_name': model_name,
                'y_pred': y_pred,
                'y_test': y_test,
                'models_test': models_test,
                'features': features,
                'importances': rf_importances  # Added to the dictionary
            })

    # Global compilation summary ranking block
    print(f"\n{'='*80}\n🏆 GLOBAL ZERO-SHOT MODEL RANKING (BY LOWEST MAE)\n{'='*80}")
    final_results.sort(key=lambda x: x['mae'])
    
    print(f"{'Rank':<5} | {'Scenario (Model)':<55} | {'MAE':<10} | {'R²':<8}")
    print("-" * 90)
    for idx, item in enumerate(final_results, 1):
        print(f"#{idx:<4} | {item['display_name']:<55} | {item['mae']:<10.1f} | {item['r2']:<8.4f}")

    print(f"\n📸 Generating and saving plots ONLY for the Top 4 models...")
    
    # Unified scientific palette for model tracking across domain shift
    color_map = {
        'Model 7M': 'royalblue',
        'Model 30M': 'darkorange',
        'Model 124M': 'forestgreen'
    }

    for idx, item in enumerate(final_results[:4], 1):
        r2 = item['r2']
        mae = item['mae']
        model_name = item['model_name']
        scenario_name = item['scenario_name']
        y_pred = item['y_pred']
        y_test_plot = item['y_test']
        models_test = item['models_test']
        features = item['features']
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Isolate scatter points by architecture to track target shift localization
        for arch_name, color in color_map.items():
            mask = models_test == arch_name
            if mask.any():
                ax.scatter(y_test_plot[mask], y_pred[mask], color=color, alpha=0.7, s=50, label=f"{arch_name} (OWT)")
                
        # Ideal identity timeline baseline (Y=X)
        ax.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'r--', linewidth=2, label="Perfect Extrapolation (Y=X)")
        
        ax.set_title(f"Top {idx}: Cross-Dataset Zero-Shot [Wiki -> OWT]\nScenario: {scenario_name}", fontsize=13)
        ax.set_xlabel("Actual Steps (OWT Odometer)", fontsize=11)
        ax.set_ylabel("Predicted Steps by ML Algorithm", fontsize=11)
        
        add_stats_box(ax, r2, mae, model_name, features)
        
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        clean_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"Top_{idx}_cross_zs_{scenario_name}_{clean_model_name}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")

    # =================================================================================
    # NEW BLOCK: Generating Feature Importance Graphs & Printing Stats for Top 10 RF
    # =================================================================================
    print(f"\n🔍 Extracting Feature Importance for Random Forest models in the Top 10...")
    for idx, item in enumerate(final_results[:10], 1):
        if item['model_name'] == 'Random Forest' and item['importances'] is not None:
            importances = item['importances']
            features_list = item['features']
            scenario_name = item['scenario_name']
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            sorted_features = [features_list[i] for i in indices]
            sorted_importances = importances[indices]
            
            # --- PRINT TO CONSOLE ---
            print(f"\nTop {idx} Rank | Scenario: {scenario_name}")
            for f_name, f_val in zip(sorted_features, sorted_importances):
                print(f"  -> {f_name:<20}: {f_val*100:.2f}%")
            # ------------------------
            
            plt.figure(figsize=(8, 5))
            colors = ['#005088', '#14B8A6', '#F59E0B', '#EF4444', '#8B5CF6'][:len(features_list)]
            bars = plt.bar(sorted_features, sorted_importances, color=colors, edgecolor='black', alpha=0.8)
            
            plt.title(f"Top {idx}: Random Forest Feature Importance\nScenario: {scenario_name}", fontsize=13, fontweight='bold', pad=15)
            plt.ylabel("Relative Importance (Gini Impurity)", fontsize=11)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.xticks(rotation=15, ha='right', fontsize=9)
            
            # Add percentage labels on top of bars
            for bar, v in zip(bars, sorted_importances):
                plt.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v*100:.1f}%", ha='center', fontweight='bold', fontsize=10)
                
            plt.tight_layout()
            filename_fi = f"Top_{idx}_FI_{scenario_name}.png"
            filepath_fi = os.path.join(OUTPUT_DIR, filename_fi)
            plt.savefig(filepath_fi, dpi=300)
            plt.close()
            print(f"Saved Feature Importance Plot: {filename_fi}")

if __name__ == "__main__":
    run_cross_dataset_zero_shot()