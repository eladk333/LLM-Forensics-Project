import os
import sys
import math
import csv
import time
import argparse
import tempfile
import json
import subprocess
# Dynamic path: Adds the current folder's 'minGPT' subdirectory to python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'minGPT'))

import torch
import glob
import re
import datasets as datasets_lib
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
import mingpt.model

# --- HUGGINGFACE AUTH CHECK ---
from huggingface_hub import HfApi
try:
    HfApi().whoami()
    print("HuggingFace: authenticated")
except Exception:
    print("WARNING: Not logged in to HuggingFace. Run: huggingface-cli login")

# --- CUSTOM BLOCK (fixed: ModuleDict for proper c_proj init) ---
# Custom architecture: uses pre-norm (LayerNorm before attention/MLP) following GPT-2/GPT-3 convention.
# ModuleDict ensures c_proj is registered for minGPT's special init (0.02/sqrt(2*n_layer) scaling).
class CustomBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = mingpt.model.CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.ModuleDict(dict(
            c_fc    = torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            act     = mingpt.model.NewGELU(),
            c_proj  = torch.nn.Linear(4 * config.n_embd, config.n_embd),
            dropout = torch.nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

mingpt.model.Block = CustomBlock

# --- MODEL CONFIGS (scaling-law aligned) ---
# Param formula: 2*V*E + B*E + L*(12*E^2 + 13*E) + 2*E  (V=50257, B=1024)
#
# Architecture configs are CUSTOM to this project (not from any single paper).
# We target approximate parameter counts and pick n_layer/n_head/n_embd to hit them,
# with head_dim=64 (GPU-optimal, matching GPT-3).
#
# Hyperparameters sourced from:
#   - Token budget: Hoffmann et al. 2022 (Chinchilla), Table A3 / Approach 3 — 20x params
#   - LR, betas, weight decay, grad clip, warmup, min_lr=peak/10, dropout=0.0:
#     Brown et al. 2020 (GPT-3), Table 2.1
#   - Cosine schedule shape: Loshchilov & Hutter 2017 (SGDR / cosine annealing)
#   - AdamW optimizer: Loshchilov & Hutter 2019 (decoupled weight decay)
#   - block_size=1024: Radford et al. 2019 (GPT-2)
#
# Batch size is kept CONSTANT across scales (effective_batch=512 seqs = ~0.5M tokens/step)
# as a deliberate research choice: isolating the effect of model capacity from optimization dynamics.
# GPT-3 Table 2.1 varies batch size with scale; we document this deviation.
MODEL_CONFIGS = {
    '100M': {
        'n_layer': 7, 'n_head': 10, 'n_embd': 640,
        'max_tokens': 2_000_000_000,     # Chinchilla 20x (Hoffmann et al. 2022, Table A3)
        'peak_lr': 6e-4,                 # GPT-3 Table 2.1 (~125M anchor)
        'min_lr': 6e-5,                  # GPT-3: min_lr = peak_lr / 10
        'micro_batch': 32,
        'grad_accum': 16,                # effective batch = 512 seqs (~0.5M tokens/step)
        'eval_interval': 500,
        'approx_params': '99.5M',
    },
    '200M': {
        'n_layer': 12, 'n_head': 14, 'n_embd': 896,
        'max_tokens': 4_000_000_000,     # Chinchilla 20x
        'peak_lr': 4.5e-4,              # log-interpolated between GPT-3 125M and 350M anchors
        'min_lr': 4.5e-5,
        'micro_batch': 16,
        'grad_accum': 32,
        'eval_interval': 1000,
        'approx_params': '206.7M',
    },
    '300M': {
        'n_layer': 16, 'n_head': 16, 'n_embd': 1024,
        'max_tokens': 6_000_000_000,     # Chinchilla 20x
        'peak_lr': 3.2e-4,              # log-interpolated between GPT-3 125M and 350M anchors
        'min_lr': 3.2e-5,
        'micro_batch': 16,
        'grad_accum': 32,
        'eval_interval': 1500,
        'approx_params': '305.5M',
    },
    '400M': {
        'n_layer': 24, 'n_head': 16, 'n_embd': 1024,
        'max_tokens': 8_000_000_000,     # Chinchilla 20x
        'peak_lr': 3e-4,                # GPT-3 Table 2.1 (~350M anchor)
        'min_lr': 3e-5,
        'micro_batch': 8,
        'grad_accum': 64,               # effective batch = 512 seqs
        'eval_interval': 2000,
        'approx_params': '406.3M',
    },
    '500M': {
        'n_layer': 24, 'n_head': 16, 'n_embd': 1152,
        'max_tokens': 10_000_000_000,    # Chinchilla 20x
        # Rounded to GPT-3 760M anchor; true interpolation gives ~2.75e-4 (C3)
        'peak_lr': 2.5e-4,              # GPT-3 Table 2.1 (~760M anchor, rounded)
        'min_lr': 2.5e-5,
        'micro_batch': 8,
        'grad_accum': 64,
        'eval_interval': 2000,
        'approx_params': '499.5M',
    },
}

# --- CLI ARGS ---
parser = argparse.ArgumentParser(description='Train GPT model (100M-500M)')
parser.add_argument('--model-size', choices=list(MODEL_CONFIGS.keys()), default='100M',
                    help='Model size config to use')
parser.add_argument('--batch-size', type=int, default=None,
                    help='Override micro batch size (default: per-config)')
parser.add_argument('--grad-accum', type=int, default=None,
                    help='Override gradient accumulation steps')
parser.add_argument('--max-tokens', type=int, default=None,
                    help='Override token budget')
parser.add_argument('--checkpoint-dir', type=str, default=None,
                    help='Override checkpoint directory')
parser.add_argument('--resume-from', type=str, default=None,
                    help='Explicit checkpoint path to resume from')
parser.add_argument('--eval-interval', type=int, default=None,
                    help='Override eval interval (optimizer steps)')
args = parser.parse_args()

# --- RESOLVE CONFIGURATION ---
cfg = MODEL_CONFIGS[args.model_size]

N_LAYER = cfg['n_layer']
N_HEAD  = cfg['n_head']
N_EMBD  = cfg['n_embd']
BLOCK_SIZE = 1024                       # Radford et al. 2019 (GPT-2)

BATCH_SIZE = args.batch_size if args.batch_size is not None else cfg['micro_batch']
GRAD_ACCUM_STEPS = args.grad_accum if args.grad_accum is not None else cfg['grad_accum']
MAX_TOKENS = args.max_tokens if args.max_tokens is not None else cfg['max_tokens']
PEAK_LR = cfg['peak_lr']
MIN_LR  = cfg['min_lr']
# GPT-3 (Brown et al. 2020, Table 2.1): 375M tokens warmup.
# At ~524K tokens/step (512 seqs * 1024 tokens) this gives 715 steps.
# NOTE (C2): Warmup asymmetry — 715 steps = 18.7% of 100M total steps but only 3.75% for 500M.
# We keep this value (GPT-3 standard) and accept the asymmetry.
WARMUP_STEPS = 715
WEIGHT_DECAY = 0.1                      # GPT-3 Table 2.1
ADAM_BETAS = (0.9, 0.95)                # GPT-3 Table 2.1
GRAD_CLIP = 1.0                         # GPT-3 Table 2.1
NUM_EPOCHS = 2                          # 500M needs ~1.2 epochs; token budget stops training

EVAL_INTERVAL = args.eval_interval if args.eval_interval is not None else cfg['eval_interval']
# C1: Fixed token count for eval to ensure comparable results across model scales.
# ~3.2M tokens regardless of batch size. Dynamically compute number of eval batches.
EVAL_TOKENS = 3_200_000

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = args.checkpoint_dir or os.path.join(
    PROJECT_ROOT, 'data', 'models', f'{args.model_size}_Context{BLOCK_SIZE}'
)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

set_seed(3407)

# --- AUTO-DETECT AMP DTYPE (V100 vs A100) ---
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    AMP_DTYPE = torch.bfloat16
    USE_SCALER = False
    print(f"AMP: bfloat16 (Ampere+ GPU detected)")
else:
    AMP_DTYPE = torch.float16
    USE_SCALER = True
    print(f"AMP: float16 with GradScaler (pre-Ampere GPU)")

# --- TRAINING DATASET ---
class LargeTextDataset(Dataset):
    def __init__(self, split='train', block_size=1024):
        self.block_size = block_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        print(f"Loading OpenWebText ({split})...")
        self.dataset = load_dataset("openwebtext", split=split)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.block_size + 1,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].squeeze(0)
        x, y = tokens[:-1], tokens[1:]
        # Mark padding positions in targets with -1 (ignored by minGPT's cross_entropy ignore_index=-1)
        pad_mask = (y == self.pad_token_id)
        y = y.masked_fill(pad_mask, -1)
        return x, y

# --- VALIDATION DATASET (concatenated chunks for clean perplexity) ---
# Used for both C4 validation (OLMo-style) and WikiText-103 (Merity et al. 2017).
# Perplexity on C4 validation (common practice) and WikiText-103 (standard benchmark).
class ValidationDataset(Dataset):
    def __init__(self, block_size=1024, max_tokens=10_000_000,
                 dataset_name="allenai/c4", dataset_config="en",
                 dataset_split="validation", streaming=True,
                 text_key="text", filter_blank=False, label=None):
        self.block_size = block_size
        self.label = label or dataset_name
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        print(f"Loading {self.label} validation split ({'streaming' if streaming else 'full'})...")
        if streaming:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split, streaming=True)
        else:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split)

        # Concatenate text into a single token stream, then chunk
        all_tokens = []
        for example in ds:
            text = example[text_key]
            if filter_blank and (not text or text.strip() == ''):
                continue
            toks = tokenizer.encode(text)
            all_tokens.extend(toks)
            if len(all_tokens) >= max_tokens:
                break

        n_chunks = len(all_tokens) // (block_size + 1)
        all_tokens = all_tokens[:n_chunks * (block_size + 1)]
        self.chunks = torch.tensor(all_tokens, dtype=torch.long).reshape(n_chunks, block_size + 1)
        print(f"  {self.label} val: {n_chunks} chunks of {block_size} tokens "
              f"({len(all_tokens)/1e6:.1f}M tokens total)")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return chunk[:-1], chunk[1:]

# --- EVALUATION FUNCTION ---
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """Evaluate model on a validation loader.

    Args:
        max_batches: if None, uses all batches in the loader.
    """
    model.eval()
    total_loss = 0.0
    count = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            logits, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    model.train()
    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return avg_loss, perplexity

# --- HELPER: ATOMIC CHECKPOINT SAVE ---
def safe_save(state_dict, path):
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

# --- HELPER: FIND LATEST CHECKPOINT ---
def get_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint by opt_step number.

    A2: Fallback uses checkpoint_epoch_*.pt (not *.pt) to avoid loading
    best_model.pt or other non-resumable files.
    """
    list_of_files = glob.glob(os.path.join(ckpt_dir, 'ckpt_opt_*.pt'))
    if not list_of_files:
        # A2: Fall back to epoch checkpoints only (not arbitrary .pt files)
        list_of_files = glob.glob(os.path.join(ckpt_dir, 'checkpoint_epoch_*.pt'))
    if not list_of_files:
        return None
    # Parse opt_step from filename for robust ordering
    def extract_step(path):
        m = re.search(r'ckpt_opt_(\d+)\.pt', path)
        return int(m.group(1)) if m else -1
    ckpt_with_steps = [(f, extract_step(f)) for f in list_of_files]
    named = [(f, s) for f, s in ckpt_with_steps if s >= 0]
    if named:
        return max(named, key=lambda x: x[1])[0]
    return max(list_of_files, key=os.path.getmtime)

# All checkpoints kept for research (A8) — norm extraction at each training step.
# No cleanup_old_checkpoints function.

# --- HELPER: TRUNCATE CSV ON RESUME (A4) ---
def truncate_csv_to_step(csv_path, max_opt_step):
    """Remove CSV rows with opt_step > max_opt_step to avoid duplicate/stale entries on resume."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) <= 1:
        return  # only header or empty
    header = rows[0]
    data_rows = rows[1:]
    kept = []
    for row in data_rows:
        try:
            step_val = int(row[0])
            if step_val <= max_opt_step:
                kept.append(row)
        except (ValueError, IndexError):
            kept.append(row)  # keep malformed rows
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)
    removed = len(data_rows) - len(kept)
    if removed > 0:
        print(f"   CSV truncated: removed {removed} rows with opt_step > {max_opt_step}")

# --- HELPER: SEEDED TRAIN LOADER (A1) ---
def make_train_loader(dataset, batch_size, epoch, base_seed=3407):
    """Create a DataLoader with a deterministic per-epoch shuffle.

    Uses a seeded RandomSampler so that resuming mid-epoch replays the exact
    same batch order. persistent_workers=False because HF dataset is in memory.
    """
    g = torch.Generator()
    g.manual_seed(base_seed + epoch)
    sampler = RandomSampler(dataset, generator=g)
    return DataLoader(
        dataset, sampler=sampler, pin_memory=True, drop_last=True,
        batch_size=batch_size, num_workers=8, persistent_workers=False
    )

# --- HELPER: GIT INFO ---
def get_git_info():
    """Get current git commit hash and dirty status."""
    info = {'git_commit': 'unknown', 'git_dirty': False}
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5
        )
        if result.returncode == 0:
            info['git_commit'] = result.stdout.strip()
        result = subprocess.run(
            ['git', 'diff', '--quiet'],
            capture_output=True, cwd=PROJECT_ROOT, timeout=5
        )
        info['git_dirty'] = (result.returncode != 0)
    except Exception:
        pass
    return info

# --- HELPER: HYPERPARAMETER PROVENANCE TABLE (B6) ---
def print_hyperparameter_provenance(model_size, cfg, total_opt_steps):
    """Print a table mapping each hyperparameter to its paper source."""
    print(f"\n{'='*72}")
    print(f"  Hyperparameter Provenance — {model_size} Model")
    print(f"{'='*72}")
    print(f"  {'Parameter':<28} {'Value':<20} {'Source'}")
    print(f"  {'-'*28} {'-'*20} {'-'*40}")
    rows = [
        ("Token budget", f"{cfg['max_tokens']/1e9:.0f}B (20x params)", "Hoffmann et al. 2022 (Chinchilla) Table A3"),
        ("Peak LR", f"{cfg['peak_lr']:.1e}", "Brown et al. 2020 (GPT-3) Table 2.1"),
        ("Min LR", f"{cfg['min_lr']:.1e} (peak/10)", "Brown et al. 2020 (GPT-3) Table 2.1"),
        ("Warmup steps", f"{WARMUP_STEPS}", "Brown et al. 2020 (GPT-3) 375M tok warmup"),
        ("LR schedule", "Cosine decay", "Loshchilov & Hutter 2017 (SGDR)"),
        ("Optimizer", "AdamW", "Loshchilov & Hutter 2019"),
        ("Adam betas", f"{ADAM_BETAS}", "Brown et al. 2020 (GPT-3) Table 2.1"),
        ("Weight decay", f"{WEIGHT_DECAY}", "Brown et al. 2020 (GPT-3) Table 2.1"),
        ("Grad clip", f"{GRAD_CLIP}", "Brown et al. 2020 (GPT-3) Table 2.1"),
        ("Dropout", "0.0", "Brown et al. 2020 (GPT-3) Section 2.1"),
        ("block_size", f"{BLOCK_SIZE}", "Radford et al. 2019 (GPT-2)"),
        ("Effective batch", f"{cfg['micro_batch'] * cfg['grad_accum']} seqs", "Constant across scales (research choice)"),
        ("Total opt steps", f"~{total_opt_steps:,}", "Derived from token budget / batch size"),
    ]
    for param, value, source in rows:
        print(f"  {param:<28} {value:<20} {source}")
    print(f"{'='*72}\n")


if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"  GPT Training — {args.model_size} model ({cfg['approx_params']})")
    print(f"{'='*60}")
    print(f"Config: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}, block={BLOCK_SIZE}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")

    # C1: Compute eval batches dynamically from fixed token budget
    eval_batches = EVAL_TOKENS // (BATCH_SIZE * BLOCK_SIZE)
    print(f"Eval: {EVAL_TOKENS/1e6:.1f}M tokens = {eval_batches} batches (batch_size={BATCH_SIZE})")

    # B7: Enhanced config.json with full reproducibility info
    git_info = get_git_info()
    config_path = os.path.join(CHECKPOINT_DIR, 'config.json')
    config_snapshot = {
        'model_size': args.model_size,
        'n_layer': N_LAYER, 'n_head': N_HEAD, 'n_embd': N_EMBD,
        'block_size': BLOCK_SIZE, 'vocab_size': 50257,
        'dropout': 0.0,                    # GPT-3 Section 2.1: zero dropout
        'batch_size': BATCH_SIZE, 'grad_accum': GRAD_ACCUM_STEPS,
        'effective_batch': BATCH_SIZE * GRAD_ACCUM_STEPS,
        'max_tokens': MAX_TOKENS, 'peak_lr': PEAK_LR, 'min_lr': MIN_LR,
        'warmup_steps': WARMUP_STEPS, 'weight_decay': WEIGHT_DECAY,
        'adam_betas': list(ADAM_BETAS), 'grad_clip': GRAD_CLIP,
        'eval_interval': EVAL_INTERVAL,
        'eval_tokens': EVAL_TOKENS,
        'seed': 3407, 'amp_dtype': str(AMP_DTYPE),
        'torch_version': torch.__version__,
        'torch_cuda_version': getattr(torch.version, 'cuda', 'N/A'),
        'cudnn_version': str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else 'N/A',
        'python_version': sys.version,
        'transformers_version': __import__('transformers').__version__,
        'datasets_version': datasets_lib.__version__,
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'git_commit': git_info['git_commit'],
        'git_dirty': git_info['git_dirty'],
        'train_dataset': 'openwebtext',
        'val_dataset': 'allenai/c4 (en, validation) + wikitext-103-raw-v1 (validation)',
    }
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config_snapshot, f, indent=2)
    else:
        print(f"   config.json already exists (preserving original hardware info)")

    # 1. Setup Training Data
    train_dataset = LargeTextDataset('train', BLOCK_SIZE)
    # DataLoader is created per-epoch via make_train_loader() (A1)

    # 2. Setup Validation Data
    # C4 validation (OLMo-style)
    val_dataset = ValidationDataset(BLOCK_SIZE, label="C4")
    val_loader = DataLoader(
        val_dataset, shuffle=False, pin_memory=True,
        batch_size=BATCH_SIZE, num_workers=4
    )

    # B9: WikiText-103 validation (Merity et al. 2017) for paper comparability
    wt103_val_dataset = ValidationDataset(
        BLOCK_SIZE, max_tokens=10_000_000,
        dataset_name="wikitext", dataset_config="wikitext-103-raw-v1",
        dataset_split="validation", streaming=False,
        text_key="text", filter_blank=True, label="WikiText-103"
    )
    wt103_val_loader = DataLoader(
        wt103_val_dataset, shuffle=False, pin_memory=True,
        batch_size=BATCH_SIZE, num_workers=4
    )

    # 3. Setup Model
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head = N_HEAD
    model_config.n_embd = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = BLOCK_SIZE
    # A0: Disable dropout — GPT-3 (Brown et al. 2020, Section 2.1) uses zero dropout.
    # MUST be set BEFORE GPT() constructor, which creates Dropout modules in __init__.
    # Research confound: dropout randomly zeros embedding dims; frequent tokens average out
    # noise better than rare tokens, creating artificial frequency-norm correlation.
    model_config.embd_pdrop = 0.0
    model_config.resid_pdrop = 0.0
    model_config.attn_pdrop = 0.0
    model = GPT(model_config)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb_params = sum(p.numel() for p in [model.transformer.wte.weight,
                                            model.transformer.wpe.weight])
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Embedding params: {n_emb_params:,} ({100*n_emb_params/n_params:.1f}%)")
    print(f"  Transformer params: {n_params - n_emb_params:,} ({100*(n_params-n_emb_params)/n_params:.1f}%)")

    # Update config snapshot with param counts (B7)
    config_snapshot['param_count_total'] = n_params
    config_snapshot['param_count_embedding'] = n_emb_params
    config_snapshot['embedding_fraction'] = round(n_emb_params / n_params, 4)

    # 4. RESUME LOGIC
    latest_ckpt_path = args.resume_from or get_latest_checkpoint(CHECKPOINT_DIR)
    start_epoch = 0
    start_step = -1
    start_opt_step = 0
    tokens_seen = 0
    BEST_LOSS = float('inf')
    BEST_VAL_LOSS = float('inf')
    eval_history = []
    checkpoint_data = None
    is_legacy_ckpt = False

    if latest_ckpt_path and os.path.exists(latest_ckpt_path):
        print(f"Found checkpoint: {latest_ckpt_path}")
        # A7: Wrap checkpoint load in try/except, fallback to second-latest
        try:
            checkpoint_data = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"   ERROR loading checkpoint: {e}")
            print(f"   Attempting to find second-latest checkpoint...")
            # Find all checkpoints and try the second-latest
            all_ckpts = sorted(
                glob.glob(os.path.join(CHECKPOINT_DIR, 'ckpt_opt_*.pt')),
                key=lambda p: int(re.search(r'ckpt_opt_(\d+)\.pt', p).group(1))
                if re.search(r'ckpt_opt_(\d+)\.pt', p) else -1
            )
            fallback = None
            for c in reversed(all_ckpts):
                if c != latest_ckpt_path:
                    fallback = c
                    break
            if fallback:
                print(f"   Trying fallback: {fallback}")
                try:
                    checkpoint_data = torch.load(fallback, map_location='cpu', weights_only=False)
                    latest_ckpt_path = fallback
                except Exception as e2:
                    print(f"   Fallback also failed: {e2}. Starting from scratch.")
                    checkpoint_data = None
            else:
                print(f"   No fallback checkpoint found. Starting from scratch.")
                checkpoint_data = None

        if checkpoint_data is not None and 'model_state_dict' in checkpoint_data:
            # A3: Validate checkpoint config matches current config
            ckpt_model_size = checkpoint_data.get('model_size', None)
            ckpt_config = checkpoint_data.get('config', {})
            config_mismatch = False
            if ckpt_model_size and ckpt_model_size != args.model_size:
                print(f"   ERROR: Checkpoint model_size '{ckpt_model_size}' != current '{args.model_size}'")
                config_mismatch = True
            for key in ['n_layer', 'n_head', 'n_embd']:
                ckpt_val = ckpt_config.get(key, None)
                current_val = {'n_layer': N_LAYER, 'n_head': N_HEAD, 'n_embd': N_EMBD}[key]
                if ckpt_val is not None and ckpt_val != current_val:
                    print(f"   ERROR: Checkpoint {key}={ckpt_val} != current {key}={current_val}")
                    config_mismatch = True
            if config_mismatch:
                print(f"   ABORTING RESUME — config mismatch. Delete old checkpoints or use correct --model-size.")
                sys.exit(1)

            state_dict = checkpoint_data['model_state_dict']
            unwrapped_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(unwrapped_state_dict)
            start_epoch = checkpoint_data.get('epoch', 0)
            start_step = checkpoint_data.get('step', -1)
            start_opt_step = checkpoint_data.get('opt_step', 0)
            tokens_seen = checkpoint_data.get('tokens_seen', 0)
            BEST_LOSS = checkpoint_data.get('best_loss', float('inf'))
            BEST_VAL_LOSS = checkpoint_data.get('best_val_loss', float('inf'))
            eval_history = checkpoint_data.get('eval_history', [])

            # A6: Round start_step down to grad accum boundary
            if start_step >= 0:
                aligned_step = (start_step // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS - 1
                if aligned_step != start_step:
                    print(f"   Aligning start_step {start_step} -> {aligned_step} (grad_accum boundary)")
                    start_step = aligned_step

            print(f"   Resuming from Epoch {start_epoch+1}, Step {start_step+1}, OptStep {start_opt_step}")
            print(f"   Best Loss: {BEST_LOSS:.4f}, Best Val Loss: {BEST_VAL_LOSS:.4f}")
            print(f"   Tokens seen: {tokens_seen/1e9:.2f}B")
        elif checkpoint_data is not None:
            # Legacy checkpoint (just model weights)
            unwrapped_state_dict = {k.replace("module.", ""): v for k, v in checkpoint_data.items()}
            model.load_state_dict(unwrapped_state_dict)
            is_legacy_ckpt = True
            match = re.search(r'epoch_(\d+)_step_(\d+)', latest_ckpt_path)
            if match:
                start_epoch = int(match.group(1)) - 1
                start_step = int(match.group(2))
            print(f"   [Legacy Checkpoint] Loaded weights. Epoch {start_epoch+1}, Step {start_step}.")
    else:
        print(f"   Starting training from SCRATCH for {args.model_size} model.")

    # 5. SINGLE GPU & AMP SETUP
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model.to(device)
    model.train()

    # 6. Configure Optimizer
    # AdamW: Loshchilov & Hutter 2019 (decoupled weight decay regularization)
    train_cfg = Trainer.get_default_config()
    train_cfg.learning_rate = PEAK_LR
    train_cfg.weight_decay = WEIGHT_DECAY
    train_cfg.betas = ADAM_BETAS
    optimizer = model.configure_optimizers(train_cfg)

    # 7. LR Schedule: Linear warmup + Cosine decay to min_lr
    # Cosine shape: Loshchilov & Hutter 2017 (SGDR)
    # Warmup + min_lr = peak/10: Brown et al. 2020 (GPT-3) Table 2.1
    # Compute from token budget (not dataset size) so cosine decays properly
    TOTAL_OPT_STEPS = MAX_TOKENS // (BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE)
    config_snapshot['total_opt_steps_planned'] = TOTAL_OPT_STEPS

    # B6: Print hyperparameter provenance table
    print_hyperparameter_provenance(args.model_size, cfg, TOTAL_OPT_STEPS)

    print(f"   Total optimizer steps: ~{TOTAL_OPT_STEPS:,}")
    print(f"   LR: {PEAK_LR} -> {MIN_LR} (cosine, {WARMUP_STEPS} warmup steps)")
    print(f"   Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS} seqs = "
          f"{BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE / 1e6:.1f}M tokens/step")
    print(f"   Token budget: {MAX_TOKENS/1e9:.1f}B | Eval every {EVAL_INTERVAL} opt steps")

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = min(1.0, (step - WARMUP_STEPS) / max(1, TOTAL_OPT_STEPS - WARMUP_STEPS))
        min_ratio = MIN_LR / PEAK_LR
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 8. AMP Scaler (only for float16 / pre-Ampere GPUs)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_SCALER)

    # 9. Load Optimizer/Scheduler/Scaler states if resuming
    if checkpoint_data is not None and not is_legacy_ckpt:
        if 'optimizer_state_dict' in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            print("   Optimizer state loaded.")
        if 'scheduler_state_dict' in checkpoint_data:
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            print("   LR Scheduler state loaded.")
        if 'scaler_state_dict' in checkpoint_data and USE_SCALER:
            scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
            print("   AMP Scaler state loaded.")

    del checkpoint_data
    torch.cuda.empty_cache()

    # 10. CSV LOGGER
    LOG_PATH = os.path.join(CHECKPOINT_DIR, 'training_log.csv')

    # A4: Truncate CSV to avoid stale/duplicate rows on resume
    if start_opt_step > 0:
        truncate_csv_to_step(LOG_PATH, start_opt_step)

    # B2: Extended CSV columns
    CSV_HEADER = [
        'opt_step', 'epoch', 'train_loss', 'val_loss', 'perplexity',
        'wt103_val_loss', 'wt103_perplexity',
        'lr', 'tokens_seen',
        'grad_norm', 'grad_clipped', 'tokens_per_sec', 'gpu_mem_gb', 'wall_time_sec'
    ]
    log_exists = os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 0
    log_file = open(LOG_PATH, 'a', newline='')
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(CSV_HEADER)

    # 11. TRAINING LOOP
    opt_step = start_opt_step
    stop_training = False
    start_time = time.time()
    last_valid_loss = 0.0        # B8: Track last valid training loss for end-of-epoch eval
    last_grad_norm = 0.0         # B1: Track gradient norm for logging
    last_grad_clipped = False

    print(f"\nStarting Training: Epoch {start_epoch+1} to {NUM_EPOCHS}")
    print(f"Max tokens: {MAX_TOKENS/1e9:.1f}B | OpenWebText: ~8B tokens")

    for epoch in range(start_epoch, NUM_EPOCHS):
        if stop_training:
            break
        print(f"\n=== EPOCH {epoch+1}/{NUM_EPOCHS} ===")

        # A1: Create seeded DataLoader per epoch for deterministic shuffle
        train_loader = make_train_loader(train_dataset, BATCH_SIZE, epoch)

        accum_loss = 0.0
        model.zero_grad(set_to_none=True)  # clean gradient state at epoch start
        step_start_time = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            # Fast-forward dataloader if resuming mid-epoch
            # A5: Guard — only fast-forward if start_step >= 0
            if epoch == start_epoch and start_step >= 0 and batch_idx <= start_step:
                # A9: Timing around fast-forward
                if batch_idx == 0:
                    ff_start = time.time()
                if batch_idx % 500 == 0:
                    print(f"Fast-forwarding... Skipping batch {batch_idx}/{start_step}", end="\r")
                if batch_idx == start_step:
                    ff_elapsed = time.time() - ff_start
                    print(f"\nFast-forward complete: skipped {start_step+1} batches in {ff_elapsed:.1f}s")
                continue

            x, y = x.to(device), y.to(device)

            # Forward + backward (accumulate gradients)
            with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                logits, loss = model(x, y)
                loss_scaled = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item() / GRAD_ACCUM_STEPS

            # Count actual non-padding tokens
            tokens_seen += (x != train_dataset.pad_token_id).sum().item()

            # Optimizer step every GRAD_ACCUM_STEPS micro-batches
            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                # B1: Capture gradient norm from clip_grad_norm_ return value
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                grad_norm_val = grad_norm.item()
                grad_clipped = grad_norm_val > GRAD_CLIP
                last_grad_norm = grad_norm_val
                last_grad_clipped = grad_clipped

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad(set_to_none=True)

                opt_step += 1

                # NaN detection
                if not math.isfinite(accum_loss):
                    print(f"\nWARNING: Loss is {accum_loss} at opt_step {opt_step}. Stopping.")
                    stop_training = True
                    break

                if accum_loss < BEST_LOSS:
                    BEST_LOSS = accum_loss

                # B8: Track last valid training loss
                last_valid_loss = accum_loss

                # B2/B3: Compute metrics for logging
                step_elapsed = time.time() - step_start_time
                toks_this_step = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
                tokens_per_sec = toks_this_step / max(step_elapsed, 1e-6)
                gpu_mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                wall_time_sec = time.time() - start_time

                # Log and print every 20 optimizer steps
                if opt_step % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    # B3: Enhanced status line
                    print(f"OptStep {opt_step} | Loss: {accum_loss:.4f} | "
                          f"LR: {current_lr:.2e} | GradNorm: {grad_norm_val:.2f} | "
                          f"{tokens_per_sec/1e3:.0f}K tok/s | {gpu_mem_gb:.1f}GB | "
                          f"{tokens_seen/1e9:.2f}B tok | "
                          f"{wall_time_sec/3600:.1f}h", end="\r")

                    # B2: Only write train-only row if NOT an eval step
                    # (eval steps get a combined row below)
                    if opt_step % EVAL_INTERVAL != 0:
                        log_writer.writerow([
                            opt_step, epoch + 1, f"{accum_loss:.6f}", "", "",
                            "", "",  # wt103 columns
                            f"{current_lr:.2e}", tokens_seen,
                            f"{grad_norm_val:.4f}", int(grad_clipped),
                            f"{tokens_per_sec:.0f}", f"{gpu_mem_gb:.2f}",
                            f"{wall_time_sec:.1f}"
                        ])
                        log_file.flush()

                # Evaluate + Checkpoint every EVAL_INTERVAL optimizer steps
                if opt_step % EVAL_INTERVAL == 0 and opt_step > 0:
                    # C4 validation
                    val_loss, val_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
                    # B9: WikiText-103 validation
                    wt103_val_loss, wt103_val_ppl = evaluate(model, wt103_val_loader, device, max_batches=eval_batches)

                    print(f"\n[Eval @ OptStep {opt_step}] "
                          f"C4 Val Loss: {val_loss:.4f} | C4 PPL: {val_ppl:.2f} | "
                          f"WT103 Val Loss: {wt103_val_loss:.4f} | WT103 PPL: {wt103_val_ppl:.2f} | "
                          f"Tokens: {tokens_seen/1e9:.2f}B")

                    current_lr = scheduler.get_last_lr()[0]
                    eval_history.append({
                        'opt_step': opt_step, 'epoch': epoch + 1,
                        'val_loss': val_loss, 'val_perplexity': val_ppl,
                        'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
                        'train_loss': accum_loss, 'lr': current_lr,
                        'tokens_seen': tokens_seen
                    })

                    # B2: Single combined row with both train and eval metrics
                    log_writer.writerow([
                        opt_step, epoch + 1, f"{accum_loss:.6f}",
                        f"{val_loss:.6f}", f"{val_ppl:.2f}",
                        f"{wt103_val_loss:.6f}", f"{wt103_val_ppl:.2f}",
                        f"{current_lr:.2e}", tokens_seen,
                        f"{grad_norm_val:.4f}", int(grad_clipped),
                        f"{tokens_per_sec:.0f}", f"{gpu_mem_gb:.2f}",
                        f"{wall_time_sec:.1f}"
                    ])
                    log_file.flush()

                    # Save best model
                    if val_loss < BEST_VAL_LOSS:
                        BEST_VAL_LOSS = val_loss
                        best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
                        safe_save({
                            'opt_step': opt_step, 'epoch': epoch, 'step': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'best_val_loss': BEST_VAL_LOSS,
                            'val_loss': val_loss, 'val_perplexity': val_ppl,
                            'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
                            'tokens_seen': tokens_seen,
                            'model_size': args.model_size,
                            'config': config_snapshot,
                        }, best_path)
                        print(f"   New best val loss! Saved best_model.pt")

                    # Regular checkpoint (atomic)
                    safe_path = os.path.join(CHECKPOINT_DIR, f'ckpt_opt_{opt_step}.pt')
                    safe_save({
                        'epoch': epoch, 'step': batch_idx, 'opt_step': opt_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_loss': BEST_LOSS, 'best_val_loss': BEST_VAL_LOSS,
                        'val_loss': val_loss, 'val_perplexity': val_ppl,
                        'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
                        'eval_history': eval_history, 'tokens_seen': tokens_seen,
                        'model_size': args.model_size,
                        'config': config_snapshot,
                    }, safe_path)

                    # All checkpoints kept for research (A8)

                accum_loss = 0.0
                step_start_time = time.time()

                # Check token budget
                if tokens_seen >= MAX_TOKENS:
                    print(f"\nReached {tokens_seen/1e9:.2f}B tokens (budget: {MAX_TOKENS/1e9:.1f}B). Stopping.")
                    stop_training = True
                    break

        # End-of-epoch evaluation (always)
        # C4 validation
        val_loss, val_ppl = evaluate(model, val_loader, device, max_batches=eval_batches)
        # B9: WikiText-103 validation
        wt103_val_loss, wt103_val_ppl = evaluate(model, wt103_val_loader, device, max_batches=eval_batches)

        print(f"\n[Eval @ End of Epoch {epoch+1}] "
              f"C4 Val Loss: {val_loss:.4f} | C4 PPL: {val_ppl:.2f} | "
              f"WT103 Val Loss: {wt103_val_loss:.4f} | WT103 PPL: {wt103_val_ppl:.2f}")

        # B8: Use last_valid_loss instead of accum_loss (which is 0.0 after reset)
        eval_history.append({
            'opt_step': opt_step, 'epoch': epoch + 1,
            'val_loss': val_loss, 'val_perplexity': val_ppl,
            'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
            'train_loss': last_valid_loss, 'lr': scheduler.get_last_lr()[0],
            'tokens_seen': tokens_seen
        })

        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
            safe_save({
                'opt_step': opt_step, 'epoch': epoch, 'step': batch_idx,
                'model_state_dict': model.state_dict(),
                'best_val_loss': BEST_VAL_LOSS,
                'val_loss': val_loss, 'val_perplexity': val_ppl,
                'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
                'tokens_seen': tokens_seen,
                'model_size': args.model_size,
                'config': config_snapshot,
            }, best_path)
            print(f"   New best val loss! Saved best_model.pt")

        # A5: Save epoch checkpoint with epoch+1 so resume starts at next epoch
        final_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
        safe_save({
            'epoch': epoch + 1, 'step': -1, 'opt_step': opt_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': BEST_LOSS, 'best_val_loss': BEST_VAL_LOSS,
            'val_loss': val_loss, 'val_perplexity': val_ppl,
            'wt103_val_loss': wt103_val_loss, 'wt103_val_perplexity': wt103_val_ppl,
            'eval_history': eval_history, 'tokens_seen': tokens_seen,
            'model_size': args.model_size,
            'config': config_snapshot,
        }, final_path)
        print(f"\nSAVED: {final_path}")

    # Cleanup
    log_file.close()
    elapsed = time.time() - start_time

    # B4: End-of-training summary
    summary_path = os.path.join(CHECKPOINT_DIR, 'training_summary.txt')
    summary_lines = [
        f"{'='*60}",
        f"  Training Summary — {args.model_size} Model ({cfg['approx_params']})",
        f"{'='*60}",
        f"",
        f"Architecture:",
        f"  Layers: {N_LAYER}, Heads: {N_HEAD}, Embed dim: {N_EMBD}",
        f"  Block size: {BLOCK_SIZE}, Vocab: 50257",
        f"  Dropout: 0.0 (GPT-3 Section 2.1)",
        f"  Total params: {n_params:,} ({n_params/1e6:.2f}M)",
        f"  Embedding params: {n_emb_params:,} ({100*n_emb_params/n_params:.1f}%)",
        f"",
        f"Hyperparameter Sources:",
        f"  Token budget: Hoffmann et al. 2022 (Chinchilla) Table A3, 20x params",
        f"  LR/betas/WD/clip/warmup/dropout: Brown et al. 2020 (GPT-3) Table 2.1",
        f"  Cosine schedule: Loshchilov & Hutter 2017 (SGDR)",
        f"  AdamW: Loshchilov & Hutter 2019",
        f"  block_size=1024: Radford et al. 2019 (GPT-2)",
        f"",
        f"Training:",
        f"  Dataset: OpenWebText (~8-9B tokens)",
        f"  Token budget: {MAX_TOKENS/1e9:.1f}B",
        f"  Tokens seen: {tokens_seen/1e9:.2f}B",
        f"  Total time: {elapsed/3600:.1f}h",
        f"  Final opt step: {opt_step}",
        f"  Peak LR: {PEAK_LR}, Min LR: {MIN_LR}",
        f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS} seqs",
        f"",
        f"Loss Trajectory:",
    ]
    # Add eval history entries
    for entry in eval_history[-10:]:  # last 10 eval points
        line = (f"  OptStep {entry['opt_step']:>6d} | "
                f"Train: {entry.get('train_loss', 0):.4f} | "
                f"C4 Val: {entry['val_loss']:.4f} | PPL: {entry['val_perplexity']:.2f}")
        if 'wt103_val_loss' in entry:
            line += f" | WT103: {entry['wt103_val_loss']:.4f} | PPL: {entry['wt103_val_perplexity']:.2f}"
        summary_lines.append(line)
    summary_lines.extend([
        f"",
        f"Health Checks:",
        f"  Best val loss: {BEST_VAL_LOSS:.4f}",
        f"  Best train loss: {BEST_LOSS:.4f}",
        f"  Final grad norm: {last_grad_norm:.4f} (clipped: {last_grad_clipped})",
        f"  NaN/Inf detected: {'YES — training stopped early' if stop_training and not math.isfinite(last_valid_loss) else 'No'}",
        f"",
        f"Artifacts:",
        f"  Checkpoints: {CHECKPOINT_DIR}",
        f"  Training log: {LOG_PATH}",
        f"  Config: {config_path}",
        f"  Best model: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}",
        f"",
        f"References:",
        f"  Brown et al. 2020 — Language Models are Few-Shot Learners (GPT-3)",
        f"  Hoffmann et al. 2022 — Training Compute-Optimal Large Language Models (Chinchilla)",
        f"  Loshchilov & Hutter 2017 — SGDR: Stochastic Gradient Descent with Warm Restarts",
        f"  Loshchilov & Hutter 2019 — Decoupled Weight Decay Regularization (AdamW)",
        f"  Radford et al. 2019 — Language Models are Unsupervised Multitask Learners (GPT-2)",
        f"  Merity et al. 2017 — Pointer Sentinel Mixture Models (WikiText-103)",
        f"{'='*60}",
    ])
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f"\n{'='*60}")
    print(f"Training complete — {args.model_size} model")
    print(f"Total time: {elapsed/3600:.1f}h")
    print(f"Tokens seen: {tokens_seen/1e9:.2f}B | Best val loss: {BEST_VAL_LOSS:.4f}")
    print(f"Log: {LOG_PATH}")
    print(f"Summary: {summary_path}")
    print(f"{'='*60}")
