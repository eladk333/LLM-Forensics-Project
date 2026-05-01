"""
extract_features_gptj_olmo.py
─────────────────────────────
Extracts the same weight-space features used in the minGPT pipeline
(embedding_norm, logit_norm, weight_variance, weight_mean, l1_norm,
dist_to_center, weight_skew, weight_kurtosis) from GPT-J-6B and OLMo-7B.

Output: one model_features.csv per model, saved to OUTPUT_DIR.
        The schema matches your existing model_features.csv files exactly,
        so the cross-model evaluation pipeline can consume them directly.

Requirements:
    pip install transformers torch scipy pandas numpy accelerate
    # OLMo also needs:
    pip install ai2-olmo
"""

import os
import numpy as np
import pandas as pd
import torch
from scipy.stats import skew, kurtosis
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# CONFIG  –  edit output dir as needed
# ─────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "models")

MODELS = {
    "gptj_6B": {
        "hf_name":    "EleutherAI/gpt-j-6b",
        "out_folder": "gptj_6B",
        # HuggingFace attribute paths inside the model object
        "emb_path":   lambda m: m.transformer.wte.weight,      # (vocab, n_embd)
        "lm_path":    lambda m: m.lm_head.weight,              # (vocab, n_embd)
    },
    "olmo_7B": {
        "hf_name":    "allenai/OLMo-7B",
        "out_folder": "olmo_7B",
        # OLMo uses model.model.transformer.wte and model.lm_head
        "emb_path":   lambda m: m.model.transformer.wte.weight,
        "lm_path":    lambda m: m.lm_head.weight,
    },
}


# ─────────────────────────────────────────────
# FEATURE EXTRACTION  (mirrors your minGPT code)
# ─────────────────────────────────────────────

def get_weight_stats(embedding_matrix: np.ndarray) -> dict:
    """
    Mirrors get_weight_stats() + get_advanced_stats() from your
    minGPT feature extractor.

    embedding_matrix : (vocab_size, n_embd) float32 numpy array
    Returns a dict of per-token arrays, each of length vocab_size.
    """
    global_mean_vector = embedding_matrix.mean(axis=0)          # (n_embd,)

    return {
        "embedding_norm":  np.linalg.norm(embedding_matrix, axis=1),
        "weight_variance": np.var(embedding_matrix, axis=1),
        "weight_mean":     np.mean(embedding_matrix, axis=1),
        "l1_norm":         np.linalg.norm(embedding_matrix, ord=1, axis=1),
        "dist_to_center":  np.linalg.norm(
                               embedding_matrix - global_mean_vector, axis=1
                           ),
        "weight_skew":     skew(embedding_matrix, axis=1),
        "weight_kurtosis": kurtosis(embedding_matrix, axis=1),
    }


def get_logit_norms(lm_weight: np.ndarray) -> np.ndarray:
    """Mirrors get_logit_norms() from your minGPT extractor."""
    return np.linalg.norm(lm_weight, axis=1)


# ─────────────────────────────────────────────
# PER-MODEL PIPELINE
# ─────────────────────────────────────────────

def extract_features(model_key: str, cfg: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  Model : {model_key}  ({cfg['hf_name']})")
    print(f"{'='*50}")

    out_folder = os.path.join(OUTPUT_DIR, cfg["out_folder"])
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, "model_features.csv")

    # ── Load tokenizer ────────────────────────────────────────────────────
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hf_name"])
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {vocab_size:,}")

    # Decode every token the same way your minGPT script does
    token_strings = [tokenizer.decode([i]) for i in range(vocab_size)]
    token_lengths = [len(s.strip()) if s.strip() else 0 for s in token_strings]
    is_upper      = [
        1 if (s.strip() and s.strip()[0].isupper()) else 0
        for s in token_strings
    ]

    # ── Load model (CPU, bfloat16 to save RAM; cast to float32 for numpy) ─
    print("  Loading model weights (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["hf_name"],
        torch_dtype=torch.float32,   # float32 so numpy ops are exact
        low_cpu_mem_usage=True,      # stream weights rather than double-buffer
        device_map="cpu",            # stay on CPU — we only need the weights
    )
    model.eval()

    # ── Extract embedding matrix ──────────────────────────────────────────
    print("  Extracting embedding matrix...")
    emb_weight = cfg["emb_path"](model).detach().float().numpy()  # (V, D)
    assert emb_weight.shape[0] == vocab_size, (
        f"Embedding rows ({emb_weight.shape[0]}) != vocab_size ({vocab_size}). "
        "Check emb_path for this model."
    )

    # ── Extract LM-head matrix ────────────────────────────────────────────
    print("  Extracting lm_head matrix...")
    lm_weight = cfg["lm_path"](model).detach().float().numpy()    # (V, D)

    # Some models tie lm_head to the embedding; that's fine — we still use it.
    if lm_weight.shape[0] != vocab_size:
        # Transposed head (rare): (D, V) → (V, D)
        lm_weight = lm_weight.T
    assert lm_weight.shape[0] == vocab_size, (
        f"LM-head rows ({lm_weight.shape[0]}) != vocab_size ({vocab_size})."
    )

    # ── Compute features ─────────────────────────────────────────────────
    print("  Computing weight statistics...")
    stats = get_weight_stats(emb_weight)

    print("  Computing logit norms...")
    logit_norms = get_logit_norms(lm_weight)

    # ── Assemble DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame({
        "token_id":  np.arange(vocab_size),
        "token_str": token_strings,
        "token_len": token_lengths,
        "is_upper":  is_upper,
        **stats,
        "logit_norm": logit_norms,
    })

    df.to_csv(out_path, index=False)
    print(f"\n  ✅  Saved {len(df):,} rows → {out_path}")
    print(f"      Columns: {list(df.columns)}")

    # Free GPU/CPU memory before next model
    del model, emb_weight, lm_weight
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for key, cfg in MODELS.items():
        try:
            extract_features(key, cfg)
        except Exception as e:
            print(f"\n  ❌  {key} failed: {e}")
            raise   # re-raise so you see the full traceback

    print("\n\nAll done.")
    print("Next step: add 'gptj_6B' and 'olmo_7B' entries to MODEL_FOLDER_MAP")
    print("and their frequency CSVs to FREQ_FILE_MAP in your evaluation script.")