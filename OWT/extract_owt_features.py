import os
import sys
import torch
import glob
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Since this script is inside 'OWT', the base path of the project is one directory up
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths based on your tree structure
DATA_CACHE_PATH = os.path.join(BASE_PATH, "data", "datasets", "owt_117M_full_cache.pt")
CHECKPOINT_BASE_DIR = os.path.join(BASE_PATH, "data", "models")
OUTPUT_CSV = os.path.join(BASE_PATH, "OWT", "owt_extracted_features.csv")

# Add minGPT to sys.path so we can import it
sys.path.append(os.path.join(BASE_PATH, 'minGPT'))

try:
    from mingpt.model import GPT
    from mingpt.utils import set_seed
except ImportError:
    print("❌ Error: minGPT not found.")
    sys.exit(1)

set_seed(3407)

# Configuration matching your runs
CONFIGS = {
    '7M_owt':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128}, 
    '30M_owt':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '124M_owt': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
}

BATCH_SIZE = 64      
BLOCK_SIZE = 128
MAX_BATCHES = 1000  # Number of fixed batches to test (2000 batches * 64 * 128 = ~16M tokens)
NUM_WORKERS = 4      

# ==========================================
# 2. DATASET
# ==========================================
class OWTDataset(Dataset):
    def __init__(self, cache_path, block_size=128):
        self.block_size = block_size
        if os.path.exists(cache_path):
            print(f"🚀 Loading tokens from: {cache_path}")
            self.tokens = torch.load(cache_path)
            if not isinstance(self.tokens, torch.Tensor):
                self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            print(f"✅ Loaded {len(self.tokens):,} tokens.")
        else:
            print(f"❌ Error: Cache not found at {cache_path}")
            self.tokens = torch.empty(0, dtype=torch.long)

    def __len__(self):
        if len(self.tokens) == 0: return 0
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def get_fixed_batches(loader, max_batches):
    print(f"🔒 Freezing {max_batches} batches for testing...")
    fixed_data = []
    iterator = iter(loader)
    try:
        for _ in range(max_batches):
            fixed_data.append(next(iterator))
    except StopIteration:
        pass
    print(f"✅ Locked {len(fixed_data)} batches in RAM.")
    return fixed_data

def calculate_next_token_prob(model, fixed_batches, device):
    """
    Calculates the average next token probability across the frozen batches.
    Returns a single scalar value.
    """
    model.eval()
    total_prob = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            for x, y in fixed_batches:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                probs = F.softmax(logits, dim=-1)
                
                # Gather the probability assigned to the CORRECT target token
                y_expanded = y.unsqueeze(-1)
                correct_probs = probs.gather(-1, y_expanded).squeeze(-1)
                
                # Sum the probabilities and count the tokens
                total_prob += correct_probs.sum().item()
                total_tokens += correct_probs.numel()
                
    if total_tokens == 0: return 0.0
    return total_prob / total_tokens

def calculate_embedding_norm(model):
    """
    Calculates the Frobenius norm of the token embedding weight matrix.
    Returns a single scalar value.
    """
    # In minGPT, the token embeddings are usually at transformer.wte.weight
    try:
        weight_matrix = model.transformer.wte.weight
    except AttributeError:
        # Fallback if structure is slightly different
        weight_matrix = model.tok_emb.weight 
        
    frob_norm = torch.norm(weight_matrix, p='fro').item()
    return frob_norm

def get_all_checkpoints_smart(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.pt"))
    numbered = []
    final_file = None
    
    for f in all_files:
        fname = os.path.basename(f)
        
        if "final_model" in fname or "full_epoch" in fname:
            final_file = f
            continue
            
        parts = fname.replace('.pt', '').split('_')
        for p in parts:
            if p.isdigit():
                if p == '1' and 'epoch' in fname: continue
                numbered.append((int(p), f))
                break
    
    numbered.sort(key=lambda x: x[0])
    
    if final_file:
        if numbered:
            max_step = numbered[-1][0]
            simulated_step = max_step + 500
        else:
            simulated_step = 500
        numbered.append((simulated_step, final_file))
        
    return numbered

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Running Extraction on: {device}")
    
    dataset = OWTDataset(DATA_CACHE_PATH, block_size=BLOCK_SIZE)
    if len(dataset) == 0: return

    loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    fixed_batches = get_fixed_batches(loader, MAX_BATCHES)
    
    all_results = []

    # Iterate over the three OWT models (7M, 30M, 124M)
    for folder_name, conf in CONFIGS.items():
        # Clean model name for the CSV (e.g., '7M_owt' -> '7M')
        model_size = folder_name.split('_')[0] 
        print(f"\n{'='*40}\n🔎 Processing: {model_size} (from {folder_name})\n{'='*40}")
        
        ckpt_folder = os.path.join(CHECKPOINT_BASE_DIR, folder_name)
        
        if not os.path.exists(ckpt_folder):
            print(f"⚠️  Folder not found at: {ckpt_folder}")
            continue

        # Initialize the model architecture
        model_config = GPT.get_default_config()
        model_config.model_type = None
        model_config.n_layer = conf['n_layer']
        model_config.n_head = conf['n_head']
        model_config.n_embd = conf['n_embd']
        model_config.vocab_size = 50257
        model_config.block_size = BLOCK_SIZE
        model = GPT(model_config).to(device)

        valid_ckpts = get_all_checkpoints_smart(ckpt_folder)
        print(f"ℹ️  Found {len(valid_ckpts)} checkpoints to analyze.")

        for step, ckpt_path in valid_ckpts:
            try:
                # Load weights
                state_dict = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(state_dict)
                
                print(f"  ⏳ Analyzing Step {step}...", end="\r")
                
                # Extract the 2 features (1 scalar per feature)
                emb_norm = calculate_embedding_norm(model)
                avg_prob = calculate_next_token_prob(model, fixed_batches, device)
                
                # Save as a single row
                all_results.append({
                    "Model": f"{model_size}", 
                    "Step": step, 
                    "Embedding_Norm": emb_norm,
                    "Next_Token_Prob": avg_prob
                })
                
                print(f"  ✅ Step {step}: Norm={emb_norm:.2f} | Prob={avg_prob:.4f}")
                
            except Exception as e:
                print(f"\n  ❌ Error processing step {step}: {e}")

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n🎉 Extraction Complete! Data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()