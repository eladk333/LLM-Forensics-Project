import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. PATH & CONFIGURATION
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(CURRENT_DIR, "eda_graphs")
os.makedirs(EDA_DIR, exist_ok=True)

WIKI_FILE = os.path.join(CURRENT_DIR, "wiki_master_features.csv")
OWT_FILE = os.path.join(CURRENT_DIR, "owt_master_features.csv")

# ==========================================
# 2. RAW GRAPH GENERATION ENGINE
# ==========================================
def plot_cross_dataset_eda_raw():
    print(f"\n📊 Loading datasets for 124M RAW feature tracking...")
    
    if not os.path.exists(WIKI_FILE) or not os.path.exists(OWT_FILE):
        print("❌ Error: Master CSV files are missing from the OWT directory.")
        return

    # Load master CSV data directly without any normalization modifications
    df_wiki_raw = pd.read_csv(WIKI_FILE)
    df_owt_raw = pd.read_csv(OWT_FILE)

    # Filter out only the 124M model trajectory rows
    df_wiki_124m = df_wiki_raw[df_wiki_raw['Model'] == 'Model 124M'].copy()
    df_owt_124m = df_owt_raw[df_owt_raw['Model'] == 'Model 124M'].copy()

    # Define the 5 specific core features to track sequentially
    features_to_plot = ['embedding_norm', 'Avg_Probability', 'l1_norm', 'weight_variance', 'logit_norm']
    
    # Accurate labeling system representing native structural units
    feature_labels = {
        'embedding_norm': 'Embedding Matrix Frobenius Norm (Raw Value)',
        'Avg_Probability': 'Average Next-Token Probability (Raw Score)',
        'l1_norm': 'Embedding Matrix L1 Norm (Raw Value)',
        'weight_variance': 'Embedding Matrix Weight Variance (Raw Value)',
        'logit_norm': 'LM Head Head Frobenius Norm (Raw Value)'
    }

    print(f"🔄 Generating 5 dynamic raw cross-dataset curves in: {EDA_DIR}")

    for feature in features_to_plot:
        plt.figure(figsize=(11, 7))
        
        # Plot native Wikipedia baseline metrics
        if not df_wiki_124m.empty:
            plt.plot(df_wiki_124m['Step'], df_wiki_124m[feature], 
                     color='royalblue', marker='o', linestyle='-', linewidth=2, alpha=0.8, label='Model 124M (Wikipedia - Raw)')
            
        # Plot native OpenWebText target metrics
        if not df_owt_124m.empty:
            plt.plot(df_owt_124m['Step'], df_owt_124m[feature], 
                     color='crimson', marker='s', linestyle='--', linewidth=2, alpha=0.8, label='Model 124M (OpenWebText - Raw)')

        # Strict academic layout configuration
        plt.title(f"Cross-Dataset Raw Trajectory: {feature}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Gradient Updates (Training Steps)", fontsize=12)
        plt.ylabel(feature_labels[feature], fontsize=12)
        
        plt.legend(loc='best', frameon=True, fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Save exact plot matrices to disk
        plot_path = os.path.join(EDA_DIR, f"eda_cross_raw_{feature}_vs_steps.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" ✅ Saved: eda_cross_raw_{feature}_vs_steps.png")

    print("\n🎉 Success! All 5 raw comparison curves have been generated for your slides.")

if __name__ == "__main__":
    plot_cross_dataset_eda_raw()