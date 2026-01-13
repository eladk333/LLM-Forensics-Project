import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import pandas as pd
import numpy as np
from mingpt.model import GPT
from transformers import GPT2Tokenizer

# Model config
MODEL_SIZE = '124M'
CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

# Paths
BASE_FOLDER = r'C:\Users\elad.k.int\LLM-Forensics-Project\data\models'
MODEL_PATH = os.path.join(BASE_FOLDER, f'MinGPT_Checkpoints_{MODEL_SIZE}')
OUTPUT_FILE = os.path.join(MODEL_PATH, 'final_frequency_dataset.csv')


def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    conf = CONFIGS[MODEL_SIZE]
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = conf['n_layer']
    model_config.n_head = conf['n_head']
    model_config.n_embd = conf['n_embd']
    model_config.vocab_size = 50257
    model_config.block_size = 128

    model = GPT(model_config)

    # Load weights
    ckpt_path = os.path.join(MODEL_PATH, 'final_model_1_epoch.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model not found at {ckpt_path}")
    
    
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval() # Switch the model from training model to evaluation/inference mode
    
    return model, tokenizer

def get_embedding_norms(model):
    # Get the embedding matrix and transfer it to an array
    embedding_matrix = model.transformer.wte.weight.detach().numpy()
    # Calculate norms
    norms = np.linalg.norm(embedding_matrix, axis=1)
    return norms


def get_logit_norms(model):
    weights = model.lm_head.weight.detach().numpy()
    return np.linalg.norm(weights, axis=1)


def extract_features():
    model, tokenizer = load_model()


    # Add the strings to the output
    vocab_size = 50257
    token_strings = [tokenizer.decode([i]) for i in range(vocab_size)]

    # Dataframe for the output
    df = pd.DataFrame({
        'token_id': np.arange(vocab_size),
        'token_str': token_strings
    })
    
    # Add all the features
    df['embedding_norm'] = get_embedding_norms(model)

    df['logit_norm'] = get_logit_norms(model)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved feautre file here: {OUTPUT_FILE}")


if __name__ == "__main__":
    extract_features()

