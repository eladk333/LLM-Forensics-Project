import os
import sys
import pandas as pd
import numpy as np

# ==========================================
# PATH CONFIGURATION (Relative to OWT folder)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Paths to the original Wikipedia feature files
REST_CSV = os.path.join(ROOT_DIR, "rest_features", "extracted_rest_features.csv")
EMB_CSV = os.path.join(ROOT_DIR, "gradient-pretiction-via-embedding_norm", "embedding_norms_empirical.csv")
PROB_CSV = os.path.join(ROOT_DIR, "probability_results_server.csv")

if not os.path.exists(PROB_CSV):
    PROB_CSV = os.path.join(ROOT_DIR, "gradient-pretiction-via-next-token-prediction", "probability_results_server.csv")

# Path for the new consolidated Wiki master file
OUTPUT_CSV = os.path.join(CURRENT_DIR, "wiki_master_features.csv")

# ==========================================
# DATA LOADING & MERGING LOGIC
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

def build_wiki_master():
    print("🔄 Loading Wikipedia features from original CSV files...")
    
    if not all(os.path.exists(p) for p in [REST_CSV, EMB_CSV, PROB_CSV]):
        print("❌ Error: One or more Wikipedia CSV files are missing. Check paths.")
        print(f"  Missing Rest Features CSV: {not os.path.exists(REST_CSV)}")
        print(f"  Missing Embedding Norms CSV: {not os.path.exists(EMB_CSV)}")
        print(f"  Missing Probability CSV: {not os.path.exists(PROB_CSV)}")
        return

    # Load raw dataframes
    df_rest = pd.read_csv(REST_CSV)
    df_emb = pd.read_csv(EMB_CSV)
    df_prob = pd.read_csv(PROB_CSV)

    print(f"  Loaded rows: Rest={len(df_rest)}, Embedding={len(df_emb)}, Probability={len(df_prob)}")

    # Clean and standardize the 'Model' and 'Step' column in all dataframes
    for name, df in [('Rest', df_rest), ('Embedding', df_emb), ('Probability', df_prob)]:
        df['Model'] = df['Model'].apply(clean_model_name)
        df['Step'] = pd.to_numeric(df['Step'], errors='coerce')
        df.dropna(subset=['Step'], inplace=True)
        df['Step'] = df['Step'].astype(int)

    # Filter embedding norm to only use the 'Global' method
    if 'Method' in df_emb.columns:
        df_emb = df_emb[df_emb['Method'].astype(str).str.strip().str.lower() == 'global']

    # Dynamically find the correct column for the numeric values
    if 'embedding_norm' in df_emb.columns:
        emb_col = 'embedding_norm'
    elif 'Feature_Value' in df_emb.columns:
        emb_col = 'Feature_Value'
    else:
        emb_col = df_emb.columns[-1] 
        
    df_emb[emb_col] = pd.to_numeric(df_emb[emb_col], errors='coerce')
    df_emb.dropna(subset=[emb_col], inplace=True)
    
    # Isolate relevant columns to prevent duplication conflicts during merge
    df_prob_sub = df_prob[['Model', 'Step', 'Avg_Probability']].drop_duplicates()
    df_emb_sub = df_emb[['Model', 'Step', emb_col]].drop_duplicates()
    df_rest_sub = df_rest.drop_duplicates()

    # Perform sequential inner joins
    print("  Merging DataFrames into master configuration...")
    df_merged = pd.merge(df_rest_sub, df_prob_sub, on=['Model', 'Step'], how='inner')
    df_merged = pd.merge(df_merged, df_emb_sub, on=['Model', 'Step'], how='inner')
    df_merged.rename(columns={emb_col: 'embedding_norm'}, inplace=True)
    
    # Ensure dataset label is clear
    df_merged['Dataset'] = 'Wikipedia'
    
    # Sort nicely by Model and Step
    df_merged.sort_values(by=['Model', 'Step'], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)

    print(f"✅ Master Merge complete. Total synchronized rows: {len(df_merged)}")
    
    # Save to OWT folder
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"💾 Saved Wiki Master dataset to: {OUTPUT_CSV}")

if __name__ == "__main__":
    build_wiki_master()