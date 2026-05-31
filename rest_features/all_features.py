import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

"""
=====================================================================================
LLM FORENSICS: ZERO-SHOT EXTRAPOLATION EXPERIMENT (7M+30M -> 124M)
=====================================================================================
EXPERIMENT OVERVIEW FOR PRESENTATION:
This script represents the final evaluation phase of our research. Our goal is to prove 
that the structural features (Norms, Variance) and behavioral features (Probability) 
of an LLM can act as a "universal clock" to predict how many Gradient Updates (Steps) 
the model has undergone.

The ultimate test is "Zero-Shot Extrapolation":
1. We train our Machine Learning algorithms ONLY on the small models (7M and 30M).
2. We ask the ML algorithm to predict the steps of the large model (124M).
3. Since the ML model has never seen the 124M architecture during training, achieving 
   a low Mean Absolute Error (MAE) proves that our features capture fundamental, 
   scale-invariant laws of neural network training, rather than just memorizing a specific model.
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
    """
    Standardizes model names into 'Model XM' format using regex.
    Handles inputs like '7M', 'Model 7M', 'Model_7M', 7, etc.
    This resolves the issue where '124M' and 'Model 124M' fail to merge.
    """
    s = str(val).strip()
    import re
    match = re.search(r'(\d+)\s*[mM]', s)
    if match:
        return f"Model {match.group(1)}M"
    # Fallback for purely numeric values
    match_num = re.search(r'(\d+)', s)
    if match_num:
        return f"Model {match_num.group(1)}M"
    return s

def load_and_merge_data():
    """
    DATA INTEGRATION LOGIC WITH ROBUST CLEANING:
    Loads data from the 3 CSV sources, aggressively sanitizes 'Model' and 'Step' 
    keys to prevent empty merges, and joins them into the master dataset.
    """
    print("🔄 Loading data from CSV files...")
    
    if not all(os.path.exists(p) for p in [REST_CSV, EMB_CSV, PROB_CSV]):
        print("❌ Error: One or more CSV files are missing. Check paths.")
        print(f"  Missing Rest Features CSV: {not os.path.exists(REST_CSV)}")
        print(f"  Missing Embedding Norms CSV: {not os.path.exists(EMB_CSV)}")
        print(f"  Missing Probability Results CSV: {not os.path.exists(PROB_CSV)}")
        return None

    # Load raw dataframes
    df_rest = pd.read_csv(REST_CSV)
    df_emb = pd.read_csv(EMB_CSV)
    df_prob = pd.read_csv(PROB_CSV)

    print(f"   Loaded rows: Rest Features={len(df_rest)}, Embedding Norms={len(df_emb)}, Probabilities={len(df_prob)}")

    # Clean and standardize the 'Model' and 'Step' column in all dataframes
    for name, df in [('Rest', df_rest), ('Embedding', df_emb), ('Probability', df_prob)]:
        if 'Model' in df.columns:
            df['Model'] = df['Model'].apply(clean_model_name)
        else:
            print(f"❌ Critical Error: 'Model' column not found in {name} CSV!")
            return None
            
        if 'Step' in df.columns:
            df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
            df.dropna(subset=['Step'], inplace=True)
            df['Step'] = df['Step'].astype(int)
        else:
            print(f"❌ Critical Error: 'Step' column not found in {name} CSV!")
            return None

    # Filter embedding norm to only use the 'Global' method
    if 'Method' in df_emb.columns:
        # Strip spaces and ignore case just to be completely safe
        df_emb = df_emb[df_emb['Method'].astype(str).str.strip().str.lower() == 'global']

    # Dynamically find the correct column for the numeric values
    if 'embedding_norm' in df_emb.columns:
        emb_col = 'embedding_norm'
    elif 'Feature_Value' in df_emb.columns:
        emb_col = 'Feature_Value'
    else:
        emb_col = df_emb.columns[-1] # Safest fallback is the last column
        
    df_emb[emb_col] = pd.to_numeric(df_emb[emb_col], errors='coerce')
    df_emb.dropna(subset=[emb_col], inplace=True)
    
    # Isolate relevant columns to prevent duplication conflicts during merge
    df_prob_sub = df_prob[['Model', 'Step', 'Avg_Probability']].drop_duplicates()
    df_emb_sub = df_emb[['Model', 'Step', emb_col]].drop_duplicates()
    df_rest_sub = df_rest.drop_duplicates()

    # Perform sequential inner joins
    print("   Merging DataFrames...")
    df_merged = pd.merge(df_rest_sub, df_prob_sub, on=['Model', 'Step'], how='inner')
    print(f"     -> After merging Rest and Probability: {len(df_merged)} rows match")
    
    df_merged = pd.merge(df_merged, df_emb_sub, on=['Model', 'Step'], how='inner')
    df_merged.rename(columns={emb_col: 'embedding_norm'}, inplace=True)
    
    print(f"✅ Master Merge complete. Total synchronized rows: {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("\n⚠️ Diagnostic Info: Showing a slice of keys from each table to inspect discrepancies:")
        print(f"   Rest unique models: {df_rest['Model'].unique()} | steps range: {df_rest['Step'].min()} to {df_rest['Step'].max()}")
        print(f"   Prob unique models: {df_prob['Model'].unique()} | steps range: {df_prob['Step'].min()} to {df_prob['Step'].max()}")
        print(f"   Embed unique models: {df_emb['Model'].unique()} | steps range: {df_emb['Step'].min()} to {df_emb['Step'].max()}")
        
    return df_merged

# ==========================================
# 3. RANK NORMALIZATION (CRITICAL FOR ZERO-SHOT)
# ==========================================
def normalize_structural_features(df):
    """
    EXPLANATION FOR PRESENTATION - THE "RANK NORMALIZATION" TECHNIQUE:
    Why do we need this? 
    The 124M model has a massive weight matrix compared to the 7M model. If we just sum 
    up the weights (like in L1 or L2 norms), the 124M model will naturally have much 
    higher values simply because it has more parameters, not necessarily because it 
    trained for more steps. This would break our Zero-Shot prediction.
    
    To fix this, we apply "Rank Normalization" - we neutralize the effect of the 
    architecture's size so that the ML models only look at the *density* or *average intensity* of the weights.
    
    Mathematical Methods Applied:
    1. Embedding Norm (L2): Divided by sqrt(d_model). L2 norm scales with the square root 
       of the dimensions, so this gives us the average magnitude per dimension.
    2. L1 Norm: Divided by (Vocab_Size * d_model). Since L1 is a pure sum of absolute values, 
       dividing by the total number of parameters gives us the Mean Absolute Weight.
    3. Logit Norm (Frobenius): Divided by sqrt(Vocab_Size * d_model). This neutralizes the 
       massive size of the unembedding matrix, providing a comparable intensity metric.
    4. Weight Variance: Left exactly as it is. Variance is already a statistical mean 
       (average squared deviation), so it is inherently scale-invariant!
    """
    df_norm = df.copy()
    
    for m_name, dim in EMBED_DIM_MAP.items():
        mask = df_norm['Model'] == m_name
        if mask.any():
            # L2 Embedding Norm -> Average magnitude per dimension
            df_norm.loc[mask, 'embedding_norm'] = df_norm.loc[mask, 'embedding_norm'] / np.sqrt(dim)
            
            # L1 Norm -> Mean Absolute Weight
            df_norm.loc[mask, 'l1_norm'] = df_norm.loc[mask, 'l1_norm'] / (VOCAB_SIZE * dim)
            
            # Logit Norm -> Normalized intensity of the output layer
            df_norm.loc[mask, 'logit_norm'] = df_norm.loc[mask, 'logit_norm'] / np.sqrt(VOCAB_SIZE * dim)
            
            # Variance requires NO normalization as it is an average by definition.
            
    return df_norm

# ==========================================
# 4. MODELING UTILITIES
# ==========================================
def get_ml_models():
    """
    ML ALGORITHMS CHOSEN FOR THE EXPERIMENT:
    1. Linear Regression: The absolute baseline. If the features have a simple linear 
       relationship with the training steps, this will perform well.
    2. Ridge (L2 Regularization): Highly important here! Since many of our structural 
       features (like L1 and Variance) grow together, they are highly correlated (Multicollinearity). 
       Ridge prevents the model from over-relying on one feature and crashing in Zero-Shot.
    3. Random Forest: A non-linear tree-based approach. Good at finding thresholds, but 
       often struggles with extrapolation outside its training domain.
    4. MLP (Multi-Layer Perceptron): A deep learning approach to map complex, non-linear 
       interactions between structural and behavioral features.
    """
    return {
        'Linear Regression': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge (L2)': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'Random Forest': make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        'MLP (Neural Net)': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=5000, random_state=42))
    }

def add_stats_box(ax, r2, mae, model_name):
    """Draws the performance metrics box on the chart."""
    stats_text = f"Model: {model_name}\n----------------\nR² Score: {r2:.4f}\nMAE: ±{mae:.1f} steps"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# ==========================================
# 5. ZERO-SHOT EXTRAPOLATION
# ==========================================
def run_zero_shot():
    """
    THE EXPERIMENT EXECUTION (ABLATION STUDY):
    Here we execute the core logic. We isolate the 124M model completely (it becomes 
    our pure Test Set). We train the ML algorithms only on 7M and 30M.
    
    Ablation Study (The 2 Scenarios):
    - SCENARIO 1: All 5 Features included.
    - SCENARIO 2: Removing the Logit Norm. 
      Why? Based on our hypothesis, the Logit Norm is a highly volatile feature. It 
      correlates strongly in small models (7M, 30M) but acts erratically in the 124M 
      model due to its massive capacity and weight decay effects. By removing it, 
      we test the hypothesis that Logit Norm is a 'Confounder' in large-scale generalization. 
      If Scenario 2 yields a lower MAE, the hypothesis is proven correct.
    """
    df_raw = load_and_merge_data()
    if df_raw is None or len(df_raw) == 0:
        print("❌ Experiment stopped: Merged dataset is empty.")
        return
        
    df_scaled = normalize_structural_features(df_raw)
    
    # Strictly isolate the training and testing sets to guarantee Zero-Shot integrity
    train = df_scaled[df_scaled['Model'].isin(['Model 7M', 'Model 30M'])]
    test = df_scaled[df_scaled['Model'] == 'Model 124M']
    
    if len(train) == 0:
        print("❌ Error: Training set (Model 7M/30M) has 0 samples after split!")
        return
    if len(test) == 0:
        print("❌ Error: Test set (Model 124M) has 0 samples after split!")
        return
        
    y_train = train['Step'].values
    y_test = test['Step'].values
    
    # Scenario definitions for the Ablation Study
    scenarios = {
        "ALL_5_FEATURES": ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance', 'logit_norm'],
        "WITHOUT_LOGIT_NORM": ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance']
    }
    
    ml_models = get_ml_models()
    
    for scenario_name, features in scenarios.items():
        print(f"\n{'='*70}")
        print(f"🚀 ZERO-SHOT SCENARIO: {scenario_name}")
        print(f"Features Used: {len(features)}")
        print(f"{'='*70}")
        print(f"{'ML Algorithm':<25} | {'R² Score':<10} | {'MAE (Steps)':<12}")
        print("-" * 55)
        
        X_train = train[features].values
        X_test = test[features].values
        
        for model_name, model in ml_models.items():
            # Fit the algorithm on small models, predict on the unseen massive model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluate the generalization capability
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{model_name:<25} | {r2:<10.4f} | {mae:<12.2f}")
            
            # Visualization: Plotting Predicted Steps vs Actual Steps
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(y_test, y_pred, color='darkmagenta', alpha=0.7, label="124M Predictions")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Extrapolation (Y=X)")
            
            ax.set_title(f"Zero-Shot Extrapolation [7M+30M -> 124M]\nScenario: {scenario_name}")
            ax.set_xlabel("Actual Steps (124M Odometer)")
            ax.set_ylabel("Predicted Steps by ML Algorithm")
            add_stats_box(ax, r2, mae, model_name)
            ax.legend(loc='lower right')
            
            # Save the experiment plot
            clean_model_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            filename = f"zs_{scenario_name}_{clean_model_name}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
            plt.close()

if __name__ == "__main__":
    run_zero_shot()