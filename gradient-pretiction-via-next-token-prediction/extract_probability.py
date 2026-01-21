import os
import sys
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
# UPDATE THIS PATH to your project folder on the server
BASE_PATH = "/home/erez/llm_project" 

# Path to the Training Data Cache (wiki_103_full_cache.pt)
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki_103_full_cache.pt")

# Add minGPT to python path
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

from mingpt.model import GPT
from mingpt.utils import set_seed

# Set seed for reproducibility of the shuffling
set_seed(3407)

# Model Architectures
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    # '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128}, # Commented out to save time (running on Colab)
}

# Hyperparameters for the Server Run
# Increased Batch Size slightly to leverage Server RAM, but kept safe for speed
BATCH_SIZE = 64      
BLOCK_SIZE = 128
# We limit to 200 batches. Since we shuffle the ENTIRE dataset, 
# 200 batches * 64 samples = 12,800 random blocks. This is statistically robust.
MAX_BATCHES = 200    
NUM_WORKERS = 4      # Increased workers to speed up data loading

# ==========================================
# 2. DATASET CLASS
# ==========================================
class WikiDataset(Dataset):
    """
    Custom Dataset to load the tokenized WikiText-103 training data.
    """
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from cache: {cache_path}")
            self.tokens = torch.load(cache_path)
            
            # Ensure tensor type
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            
            print(f"✅ Loaded {len(self.tokens):,} tokens into memory.")
        else:
            print(f"❌ Error: Cache not found at {cache_path}")
            # Fallback to empty list to prevent crash, will be handled in main
            self.tokens = torch.empty(0, dtype=torch.long)

    def __len__(self):
        # We expose the FULL dataset length. Control is done via MAX_BATCHES in the loop.
        if len(self.tokens) == 0:
            return 0
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Extract a chunk of length block_size + 1
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]   # Target (Next Token)
        return x, y

# ==========================================
# 3. CORE LOGIC: PROBABILITY EXTRACTION
# ==========================================
def calculate_checkpoint_stats(model, loader, device, max_batches):
    """
    Runs inference on the data loader and calculates the average probability
    assigned to the GROUND TRUTH (correct) next token.
    """
    model.eval()
    all_probs = []
    
    # Use Automatic Mixed Precision (AMP) for speed on the server
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            for i, (x, y) in enumerate(loader):
                if i >= max_batches:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # 1. Forward Pass
                logits, _ = model(x)
                
                # 2. Softmax to get probabilities (0.0 to 1.0)
                probs = F.softmax(logits, dim=-1)
                
                # 3. Gather the probability of the TRUE target
                # y shape: [B, T] -> y_expanded: [B, T, 1]
                y_expanded = y.unsqueeze(-1)
                
                # Gather along the last dimension (vocab)
                correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
                
                # 4. Store mean probability of this batch
                all_probs.append(correct_probs.mean().item())
            
    # Return Global Mean over all batches
    if len(all_probs) == 0:
        return 0.0
    return np.mean(all_probs)

def get_step_number(filepath):
    """ Extracts the step number from the checkpoint filename (e.g., ckpt_500.pt -> 500) """
    try:
        # Assuming format likes 'ckpt_500.pt' or 'run_step_500.pt'
        # Taking the last part after split by '_' and removing extension
        filename = os.path.basename(filepath)
        parts = filename.replace('.pt', '').split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return -1
    except:
        return -1

# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running Experiment on: {device}")
    print(f"📂 Base Path: {BASE_PATH}")
    
    # 1. Initialize Dataset (Loaded once for all models)
    # Note: We use shuffle=True here to sample from the ENTIRE distribution
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0:
        print("❌ Dataset is empty. Exiting.")
        return

    loader = DataLoader(dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    
    all_results = []

    # 2. Iterate over all model sizes (7M, 30M, 124M)
    for model_size, conf in CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"🔎 Processing Model Configuration: {model_size}")
        print(f"{'='*50}")
        
        # Define Checkpoint Directory
        ckpt_folder = os.path.join(BASE_PATH, f"MinGPT_Checkpoints_{model_size}")
        if not os.path.exists(ckpt_folder):
            print(f"⚠️  Folder not found: {ckpt_folder}. Skipping model.")
            continue

        # Initialize Model Architecture
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        
        model = GPT(model_config)
        model.to(device)

        # Retrieve and Sort Checkpoints
        files = glob.glob(os.path.join(ckpt_folder, "*.pt"))
        valid_ckpts = [f for f in files if get_step_number(f) >= 0]
        # Sort by step count
        sorted_ckpts = sorted(valid_ckpts, key=get_step_number)
        
        print(f"ℹ️  Found {len(sorted_ckpts)} valid checkpoints.")

        # Process each checkpoint
        for ckpt_path in sorted_ckpts:
            step = get_step_number(ckpt_path)
            
            try:
                # Load weights
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                # Run Analysis
                print(f"   ⏳ Analyzing Step {step}...", end="\r")
                avg_prob = calculate_checkpoint_stats(model, loader, device, max_batches=MAX_BATCHES)
                
                print(f"   ✅ Step {step}: Avg Confidence = {avg_prob:.5f}")
                
                all_results.append({
                    "Model": f"Model {model_size}",
                    "Step": step,
                    "Avg_Probability": avg_prob
                })
                
            except Exception as e:
                print(f"\n   ❌ Error reading checkpoint {ckpt_path}: {e}")

    # 3. Save Consolidated Results
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        output_csv = os.path.join(BASE_PATH, "probability_results_full.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 SUCCESS! Results saved to: {output_csv}")
    else:
        print("\n⚠️  No results generated.")

if __name__ == "__main__":
    main()