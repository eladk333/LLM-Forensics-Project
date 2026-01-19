import torch
import os
import glob
import re
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION - PATHS
# ==========================================

# Base path to 'My Drive' - adjust if necessary
MY_DRIVE_BASE = r"G:/My Drive" 


PATH_MY_7M = os.path.join(MY_DRIVE_BASE, "LLM_erez_property", "MinGPT_7M_Checkpoints")
PATH_MY_30M = os.path.join(MY_DRIVE_BASE, "LLM_erez_property", "MinGPT_30M_Checkpoints")


FRIENDS_SHARED_FOLDER = os.path.join(MY_DRIVE_BASE, "LLM_erez_property", "124M") 

# output CSV file path
OUTPUT_CSV = os.path.join(MY_DRIVE_BASE, "LLM_erez_property", "gradient_norm_results.csv")

# Folders to analyze
FOLDERS_TO_ANALYZE = {
    "Model 7M": PATH_MY_7M,
    "Model 30M": PATH_MY_30M,
    "Model 124M": FRIENDS_SHARED_FOLDER
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_step_number(filename):
    """Extracts the step number from the filename (e.g., ckpt_step_500.pt)."""
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def process_folder(model_name, folder_path):
    print(f"\n🔎 Analyzing {model_name} in: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder not found: {folder_path}")
        print("   (Tip: Did you create a Shortcut to the shared folder in your 'My Drive'?)")
        return []

    # List all .pt files in the folder
    files = glob.glob(os.path.join(folder_path, "*.pt"))
    
    # Filter only files that have 'step' in their name (to exclude final_model if it doesn't have a number)
    checkpoint_files = [f for f in files if "step_" in os.path.basename(f)]
    
    folder_data = []
    print(f"   Found {len(checkpoint_files)} checkpoint files.")

    for file_path in checkpoint_files:
        step = extract_step_number(os.path.basename(file_path))
        
        try:
            # Load weights to CPU (saves GPU memory and prevents crashes)
            state_dict = torch.load(file_path, map_location='cpu')

            # Extract Embedding table
            # In minGPT, weights are located at transformer.wte.weight
            if 'transformer.wte.weight' in state_dict:
                wte = state_dict['transformer.wte.weight']
                
                # Calculate Norm (L2) of the embedding weights
                norm_val = torch.norm(wte, p=2).item()

                folder_data.append({
                    "Model": model_name,
                    "Step": step,
                    "Embedding_Norm": norm_val
                })
            else:
                # Sometimes in other models the name is different, try a fallback
                if 'wte.weight' in state_dict:
                    wte = state_dict['wte.weight']
                    norm_val = torch.norm(wte, p=2).item()
                    folder_data.append({"Model": model_name, "Step": step, "Embedding_Norm": norm_val})
                else:
                    print(f"   ⚠️ 'wte' weights not found in {os.path.basename(file_path)}")

        except Exception as e:
            print(f"   ❌ Error reading {os.path.basename(file_path)}: {e}")

    return folder_data

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    all_results = []

    for name, path in FOLDERS_TO_ANALYZE.items():
        results = process_folder(name, path)
        all_results.extend(results)

    if all_results:
        # Create DataFrame
        df = pd.DataFrame(all_results)
        # Sort by Model and Step for organization
        df = df.sort_values(by=["Model", "Step"])
        
        # Save to CSV
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Success! Data saved to: {OUTPUT_CSV}")
        print("Preview:")
        print(df.head())
    else:
        print("\n❌ No data extracted. Check your paths.")