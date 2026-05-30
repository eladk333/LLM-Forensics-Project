import pandas as pd
import os

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
STEP2_MASTER = os.path.join(CURRENT_DIR, 'step2_master_matrix.csv')
STEP2S_MASTER = os.path.join(CURRENT_DIR, 'step2s_master_matrix.csv')

# Output files
STEP2_INTERACTION = os.path.join(CURRENT_DIR, 'step2_interaction_matrix.csv')
STEP2S_INTERACTION = os.path.join(CURRENT_DIR, 'step2s_interaction_matrix.csv')

def create_interaction_matrix(input_path, output_path, prob_col_name):
    """
    Loads the master matrix, multiplies all 'Bin_' columns by the probability column,
    drops the probability column, and saves the new interaction matrix.
    """
    if not os.path.exists(input_path):
        print(f"❌ Error: Could not find {input_path}")
        return False

    df = pd.read_csv(input_path)
    
    if prob_col_name not in df.columns:
        print(f"❌ Error: Column '{prob_col_name}' not found in {input_path}")
        return False

    bin_cols = [col for col in df.columns if col.startswith('Bin_')]
    
    # Multiply each bin by the probability value
    for col in bin_cols:
        df[col] = df[col] * df[prob_col_name]
        
    # Drop the original probability column since its influence is now embedded in the bins
    df.drop(columns=[prob_col_name], inplace=True)
    
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Created interaction matrix: {os.path.basename(output_path)}")
    return True

if __name__ == "__main__":
    print("🚀 Starting Interaction Matrix Generation...\n")
    
    # Create for Step 2 (Noisy / Per-Batch Data)
    print("Processing Step 2 (Macro/Noisy Data)...")
    create_interaction_matrix(STEP2_MASTER, STEP2_INTERACTION, 'Batch_Probability')
    
    print("-" * 40)
    
    # Create for Step 2s (Clean / Averaged Data)
    print("Processing Step 2s (Micro/Clean Data)...")
    create_interaction_matrix(STEP2S_MASTER, STEP2S_INTERACTION, 'Mean_Eval_Probability')
    
    print("\n🎉 All interaction matrices generated successfully!")