import pandas as pd
import os

def create_step2s_matrix():
    print("🚀 Starting Data Merge for Step 2s (Fixed-Eval Macro Integration)...")

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

    # We use the raw batch level file but we will aggregate it to get the smooth evaluation curve
    batches_path = os.path.join(ROOT_DIR, "gradient-pretiction-via-next-token-prediction", "probability_results_batch_level.csv")
    bins_path = os.path.join(ROOT_DIR, "gradient-pretiction-via-embedding_norm", "embedding_norms_empirical.csv")

    # Load Data
    try:
        print(f"Loading Batches from: {os.path.basename(batches_path)}")
        df_batches = pd.read_csv(batches_path)
        
        print(f"Loading Bins from: {os.path.basename(bins_path)}")
        df_bins = pd.read_csv(bins_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the file. {e}")
        return

    # Fix the "Model" naming mismatch
    df_batches['Model'] = df_batches['Model'].astype(str).str.replace('Model ', '')
    df_bins['Model'] = df_bins['Model'].astype(str).str.replace('Model ', '')

    # Filter ONLY the Random Bins
    print("Filtering exclusively for 'Random' Bins...")
    df_bins_random = df_bins[df_bins['Method'] == 'Random'].copy()

    # Pivot the Bins (Turn 50 rows into 50 columns)
    print("Pivoting the 50 Bins into columns...")
    df_bins_wide = df_bins_random.pivot_table(
        index=['Model', 'Step'], 
        columns='Bin_ID', 
        values='Feature_Value'
    ).reset_index()

    # Rename the new columns to look like "Bin_0", "Bin_1", etc.
    new_cols = []
    for col in df_bins_wide.columns:
        if col in ['Model', 'Step']:
            new_cols.append(col)
        else:
            new_cols.append(f'Bin_{col}')
    df_bins_wide.columns = new_cols

    # Aggregate the probabilities (This is the implementation of Yanai's fixed eval approach)
    print("Aggregating 1000 batches into a single smooth Mean Evaluation Probability...")
    df_mean_prob = df_batches.groupby(['Model', 'Step'])['Batch_Probability'].mean().reset_index()
    df_mean_prob.rename(columns={'Batch_Probability': 'Mean_Eval_Probability'}, inplace=True)

    # Merge the Aggregated Probability with the Pivoted Bins
    print("Merging features...")
    df_master = pd.merge(df_mean_prob, df_bins_wide, on=['Model', 'Step'], how='inner')

    # Sort values to make the CSV look pretty
    df_master = df_master.sort_values(by=['Model', 'Step'])

    # Save the Master Matrix
    output_path = os.path.join(CURRENT_DIR, 'step2s_master_matrix.csv')
    df_master.to_csv(output_path, index=False)
    
    print(f"\n🎉 Success! Step 2s master matrix created.")
    print(f"📊 Final Shape: {df_master.shape[0]} rows, {df_master.shape[1]} columns.")
    print(f"💾 Saved to: {output_path}")

if __name__ == "__main__":
    create_step2s_matrix()