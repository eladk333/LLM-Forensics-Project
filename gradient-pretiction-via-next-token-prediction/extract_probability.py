import os
import sys
import torch
import glob
import pandas as pd
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

# ==========================================
# 1. SERVER CONFIGURATION
# ==========================================
# UPDATE THIS PATH to your project folder on the Linux server
BASE_PATH = "/home/erez/llm_project" 

# Path to the Training Data Cache (WikiText-103)
# Ensuring we use the training set to measure "memorization" / learning progress
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "wiki_103_full_cache.pt")

# Add minGPT to system path
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: minGPT library not found. Check the sys.path append above.")
    sys.exit(1)

set_seed(3407)

# Model Architectures
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# Hyperparameters for Inference
BATCH_SIZE = 32
BLOCK_SIZE = 128
MAX_BATCHES = 150  # Optimization: Stop after 150 batches (approx 4,800 blocks) to save time

# ==========================================
# 2. DATASET CLASS
# ==========================================
class WikiDataset(Dataset):
    """
    Custom Dataset to load pre-tokenized tensors.
    """
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from: {cache_path}")
            self.tokens = torch.load(cache_path)
            # Ensure tensor type
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens)} tokens.")
        else:
            print(f"❌ Cache not found at {cache_path}")
            self.tokens = torch.tensor([], dtype=torch.long)

    def __len__(self):
        # Optimization: We effectively limit the dataset iteration in the main loop via MAX_BATCHES,
        # but technically we expose the full length here minus the block buffer.
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        # Extract a block of context (x) and the target next token (y)
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ==========================================
# 3. STATISTICAL CALCULATION LOGIC
# ==========================================
def calculate_confidence(model, loader, device, max_batches=150):
    """
    Calculates the Average Probability assigned by the model to the TRUE next token.
    
    Logic:
    1. Forward pass to get logits.
    2. Apply Softmax to get probabilities.
    3. Gather the probability corresponding to the ground truth token (y).
    4. Compute the mean over the batch.
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():  # Optimization: Disable gradient calculation to save VRAM
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break  # Optimization: Stop early to save time
            
            x, y = x.to(device), y.to(device)
            
            # Forward Pass
            logits, _ = model(x)
            
            # Softmax: Convert logits to probabilities (0 to 1)
            probs = F.softmax(logits, dim=-1)
            
            # Gather: Extract the probability of the specific ground truth token
            # y shape: [Batch, Time] -> Expanded: [Batch, Time, 1]
            y_expanded = y.unsqueeze(-1)
            correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
            
            # Append batch mean
            all_probs.append(correct_probs.mean().item())
            
    if len(all_probs) == 0:
        return 0.0
        
    return np.mean(all_probs)

def get_step_from_filename(filepath):
    """ Extracts the step number from filenames like 'ckpt_500.pt' """
    try:
        # Assuming format ends with: ..._500.pt
        return int(filepath.split('_')[-1].split('.')[0])
    except:
        return -1

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running Experiment on: {device}")
    
    # Load Dataset (Training Data)
    dataset = WikiDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0:
        print("❌ Dataset is empty. Exiting.")
        return

    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=2)
    
    all_results = []

    # Iterate over all model sizes
    for model_size, conf in CONFIGS.items():
        print(f"\n{'='*40}")
        print(f"🔎 Processing Model: {model_size}")
        print(f"{'='*40}")
        
        # Define Checkpoint Path
        ckpt_folder = os.path.join(BASE_PATH, f"MinGPT_Checkpoints_{model_size}")
        
        if not os.path.exists(ckpt_folder):
            print(f"⚠️ Folder not found: {ckpt_folder}. Skipping.")
            continue

        # Initialize Model Structure
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        
        model = GPT(model_config)
        model.to(device)

        # Find and Sort Checkpoints
        files = glob.glob(os.path.join(ckpt_folder, "*.pt"))
        valid_ckpts = [f for f in files if get_step_from_filename(f) >= 0]
        sorted_ckpts = sorted(valid_ckpts, key=get_step_from_filename)
        
        print(f"   Found {len(sorted_ckpts)} valid checkpoints.")

        # Process each checkpoint
        for ckpt_path in sorted_ckpts:
            step = get_step_from_filename(ckpt_path)
            
            try:
                # Load Weights
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                # Run Calculation
                avg_prob = calculate_confidence(model, loader, device, max_batches=MAX_BATCHES)
                
                print(f"   -> Step {step}: Avg Confidence = {avg_prob:.5f}")
                
                all_results.append({
                    "Model": f"Model {model_size}",
                    "Step": step,
                    "Avg_Probability": avg_prob
                })
                
            except Exception as e:
                print(f"   ❌ Error processing step {step}: {e}")

    # Save Final Results
    df = pd.DataFrame(all_results)
    output_csv = os.path.join(BASE_PATH, "probability_growth_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n✅ SUCCESS! All results saved to: {output_csv}")

if __name__ == "__main__":
    main()