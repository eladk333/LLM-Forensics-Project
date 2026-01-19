import torch
import os
import glob
import re
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION - LOCAL PATHS (VS Code)
# ==========================================

# Base path for local Windows execution
MY_DRIVE_BASE = r"G:/My Drive" 
PROJECT_FOLDER = "LLM_erez_property"

# Define paths to the specific model checkpoints
# Note: Ensure "124M" is a valid Shortcut inside LLM_erez_property
PATH_MY_7M = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "MinGPT_7M_Checkpoints")
PATH_MY_30M = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "MinGPT_30M_Checkpoints")
FRIENDS_SHARED_FOLDER = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "124M") 

# Output CSV file path
OUTPUT_CSV = os.path.join(MY_DRIVE_BASE, PROJECT_FOLDER, "gradient_norm_results.csv")

# Map model names to their respective folder paths
FOLDERS_TO_ANALYZE = {
    "Model 7M": PATH_MY_7M,
    "Model 30M": PATH_MY_30M,
    "Model 124M": FRIENDS_SHARED_FOLDER
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_step_number(filename):
    """
    Extracts the integer step number from the filename.
    Expected format: 'ckpt_step_500.pt' -> returns 500.
    """
    match = re.search(r'step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def get_model_step_map(files):
    """
    Creates a mapping of filename -> step number.
    Handles 'final_model_1_epoch.pt' by assigning it (max_step + interval).
    This ensures we don't lose the final data point.
    """
    step_map = {}
    steps = []
    
    # 1. First pass: Identify numbered steps
    for f in files:
        basename = os.path.basename(f)
        if "step_" in basename:
            s = extract_step_number(basename)
            if s != -1:
                step_map[f] = s
                steps.append(s)
    
    # 2. Determine interval and max step (default to 500 if unknown)
    if not steps:
        return {} 
    
    max_step = max(steps)
    steps.sort()
    interval = 500
    if len(steps) > 1:
        interval = steps[1] - steps[0] 
        if interval <= 0: interval = 500

    # 3. Second pass: Handle 'final_model'
    for f in files:
        basename = os.path.basename(f)
        if "final_model" in basename and f not in step_map:
            final_step = max_step + interval
            step_map[f] = final_step
            print(f"   ℹ️ Identified final model: {basename} -> Assigned Step {final_step}")
            
    return step_map

def process_folder(model_name, folder_path):
    """
    Iterates through .pt files in a folder, extracts embedding weights, 
    calculates L2 norm, and returns a list of data dictionaries.
    """
    print(f"\n🔎 Analyzing {model_name} in: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder not found: {folder_path}")
        return []

    # Get all .pt files
    all_files = glob.glob(os.path.join(folder_path, "*.pt"))
    
    # Map files to steps
    file_step_map = get_model_step_map(all_files)
    files_to_process = list(file_step_map.keys())
    
    folder_data = []
    print(f"   Found {len(files_to_process)} valid model files.")

    for i, file_path in enumerate(files_to_process):
        # Print progress every 10 files
        if i % 10 == 0:
            print(f"   Processing file {i}/{len(files_to_process)}...")

        step = file_step_map[file_path]
        
        try:
            # Load weights to CPU
            state_dict = torch.load(file_path, map_location='cpu')

            # Extract Embedding table & Calculate Norm (L2)
            # The core logic of the experiment is here:
            if 'transformer.wte.weight' in state_dict:
                wte = state_dict['transformer.wte.weight']
                norm_val = torch.norm(wte, p=2).item()
                folder_data.append({"Model": model_name, "Step": step, "Embedding_Norm": norm_val})
            
            elif 'wte.weight' in state_dict: 
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
        # Save to CSV
        df = pd.DataFrame(all_results)
        df = df.sort_values(by=["Model", "Step"])
        
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Success! Data saved to: {OUTPUT_CSV}")
    else:
        print("\n❌ No data extracted. Please check your folder paths.")