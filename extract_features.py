import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import mingpt.model
from mingpt.model import GPT
from transformers import GPT2Tokenizer

class MultiGPUBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = mingpt.model.CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            mingpt.model.NewGELU(),
            torch.nn.Linear(4 * config.n_embd, config.n_embd),
            torch.nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Keep a reference to the original Block before any patching
ORIGINAL_BLOCK = mingpt.model.Block

# Model config
CONFIGS = {
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128,  'block_size': 128,  'folder': 'MinGPT_Checkpoints_7M',   'ckpt': 'final_model_1_epoch.pt',      'use_multi_gpu_block': False},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384,  'block_size': 128,  'folder': 'MinGPT_Checkpoints_30M',  'ckpt': 'final_model_1_epoch.pt',      'use_multi_gpu_block': False},
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768,  'block_size': 128,  'folder': 'MinGPT_Checkpoints_124M', 'ckpt': 'final_model_1_epoch.pt',      'use_multi_gpu_block': False},
    '500M': {'n_layer': 24, 'n_head': 16, 'n_embd': 1280, 'block_size': 1024, 'folder': 'MinGPT_Checkpoints_500M', 'ckpt': 'checkpoint_epoch_1.pt',       'use_multi_gpu_block': True},
}

BASE_FOLDER = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\models'


def load_model(size):
    conf = CONFIGS[size]

    # Patch Block only for 500M (trained with MultiGPUBlock), restore for others
    if conf['use_multi_gpu_block']:
        mingpt.model.Block = MultiGPUBlock
    else:
        mingpt.model.Block = ORIGINAL_BLOCK

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = conf['n_layer']
    model_config.n_head  = conf['n_head']
    model_config.n_embd  = conf['n_embd']
    model_config.vocab_size = 50257
    model_config.block_size = conf['block_size']

    model = GPT(model_config)

    model_path = os.path.join(BASE_FOLDER, conf['folder'])
    ckpt_path  = os.path.join(model_path, conf['ckpt'])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format (500M)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded dict checkpoint (epoch={checkpoint.get('epoch','?')}, "
              f"step={checkpoint.get('step','?')}, "
              f"best_loss={checkpoint.get('best_loss', float('nan')):.4f})")
    else:
        # Legacy raw state dict (7M, 30M, 124M)
        model.load_state_dict(checkpoint)
        print(f"  Loaded raw state dict.")

    model.eval()
    return model, tokenizer, model_path


def get_weight_stats(model):
    embedding_matrix = model.transformer.wte.weight.detach().numpy()
    l2_norm  = np.linalg.norm(embedding_matrix, axis=1)
    variance = np.var(embedding_matrix, axis=1)
    mean     = np.mean(embedding_matrix, axis=1)
    l1_norm  = np.linalg.norm(embedding_matrix, ord=1, axis=1)
    global_mean_vector = np.mean(embedding_matrix, axis=0)
    dist_to_center = np.linalg.norm(embedding_matrix - global_mean_vector, axis=1)
    return l2_norm, variance, mean, l1_norm, dist_to_center


def get_advanced_stats(model):
    embedding_matrix = model.transformer.wte.weight.detach().numpy()
    skew_val = skew(embedding_matrix, axis=1)
    kurt_val = kurtosis(embedding_matrix, axis=1)
    return skew_val, kurt_val


def get_logit_norms(model):
    weights = model.lm_head.weight.detach().numpy()
    return np.linalg.norm(weights, axis=1)


def extract_features_for(size):
    print(f"\n{'='*40}")
    print(f"Processing {size} model...")
    model, tokenizer, model_path = load_model(size)

    vocab_size    = 50257
    token_strings = [tokenizer.decode([i]) for i in range(vocab_size)]
    token_lengths = [len(s.strip()) if len(s.strip()) > 0 else 0 for s in token_strings]
    is_upper      = [1 if (s.strip() and s.strip()[0].isupper()) else 0 for s in token_strings]

    df = pd.DataFrame({
        'token_id':  np.arange(vocab_size),
        'token_str': token_strings,
        'token_len': token_lengths,
        'is_upper':  is_upper
    })

    print("  Extracting weight statistics...")
    l2, var, mean, l1, dist = get_weight_stats(model)
    df['embedding_norm']  = l2
    df['weight_variance'] = var
    df['weight_mean']     = mean
    df['l1_norm']         = l1
    df['dist_to_center']  = dist

    print("  Extracting advanced stats...")
    skew_val, kurt_val = get_advanced_stats(model)
    df['weight_skew']     = skew_val
    df['weight_kurtosis'] = kurt_val

    print("  Extracting logit norms...")
    df['logit_norm'] = get_logit_norms(model)

    output_file = os.path.join(model_path, 'model_features.csv')
    df.to_csv(output_file, index=False)
    print(f"  ✅ Saved: {output_file}")


if __name__ == "__main__":
    for size in ['7M', '30M', '124M', '500M']:
        try:
            extract_features_for(size)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping {size}: {e}")

    print("\nDone.")