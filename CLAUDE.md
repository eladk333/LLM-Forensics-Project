# LLM-Forensics-Project

## Overview
Research project training GPT models via Karpathy's minGPT on OpenWebText, then analyzing the relationship between token frequency and model internal representations (embedding/logit norms).

## Server Rules (MANDATORY)

- **Compute server**: `ssh tsruyag@dsinlp01`
- **Single GPU only** — never use DataParallel or multi-GPU
- **Before ANY GPU job**: run `nvidia-smi` to find a free GPU
- **Always prefix**: `CUDA_VISIBLE_DEVICES=X python train.py` (where X is the free GPU)
- **Always use tmux** for training — never launch long jobs outside tmux
- **HuggingFace auth required**: run `huggingface-cli login` or set `HF_TOKEN` env var before accessing datasets

## Running Training

```bash
# 1. Connect to compute server
ssh tsruyag@dsinlp01

# 2. Start (or reattach) tmux session
tmux new -s training
# or
tmux attach -t training

# 3. Check which GPU is free
nvidia-smi

# 4. Run on a free GPU (e.g., GPU 3)
CUDA_VISIBLE_DEVICES=3 python train.py
```

## Project Structure

| File | Purpose |
|---|---|
| `train.py` | Main training script (100M-500M models, OpenWebText, AMP + eval + checkpointing) |
| `token_frequency.py` | Count token frequencies from cached WikiText-103 dataset |
| `extract_features.py` | Extract embedding/logit norms from a trained checkpoint |
| `extract_embedded_norm` | Older version of feature extraction (no .py extension) |
| `analyze_results.py` | Regression analysis: predict log-frequency from embedding norm |
| `plot_training.py` | Plot training/val loss, perplexity, and LR curves from logs + checkpoints |
| `minGPT/` | Git submodule (Karpathy's minGPT) — do NOT modify files inside |
| `requirements.txt` | torch, transformers, datasets |

### Pipeline Flow
1. Train a GPT model (`train.py`)
2. Count token frequencies from a corpus (`token_frequency.py`)
3. Extract embedding/logit norms from the trained model (`extract_features.py`)
4. Analyze — predict token frequency from norms using regression/MLP (`analyze_results.py`)

## Model Configuration (train.py) — 100M Target

| Parameter | Value | Reasoning |
|---|---|---|
| Layers | 7 | Balanced depth for 100M budget |
| Heads | 10 | head_dim=64 (GPU-optimal) |
| Embedding dim | 640 | Closest config to 100M |
| Context (block_size) | 1024 | Standard GPT-2 context |
| Vocab size | 50257 (GPT-2 tokenizer) | |
| Approx params | **99.45M** | Verified against minGPT formula |
| Dropout | 0.0 | GPT-3 standard (Brown et al. 2020, Section 2.1) |
| Batch size | 32 micro, 512 effective (16x grad accum) | ~0.5M tokens/step |
| Checkpoints | `data/models/100M_Context1024/` | |

### Training Recipe (Chinchilla-informed)
| Setting | Value |
|---|---|
| Training tokens | 2B (Chinchilla 20x; Hoffmann et al. 2022, Table A3) |
| Peak LR | 6e-4 (GPT-3 Table 2.1 for 125M) |
| Min LR | 6e-5 (10% of peak) |
| Schedule | Cosine decay with 715-step warmup (GPT-3: 375M tokens) |
| Weight decay | 0.1 |
| Adam betas | (0.9, 0.95) |
| Grad clip | 1.0 |
| Eval | C4 + WikiText-103 validation, every 500 optimizer steps (100M) |

Checkpoints save every EVAL_INTERVAL optimizer steps (500 for 100M) + end of each epoch. Full state: model, optimizer, scaler, scheduler, eval metrics.

### Scaling Law Notes
- Chinchilla optimal for 100M = ~2B tokens (20x ratio)
- OpenWebText has ~8-9B tokens → train <1 epoch (no data repetition)
- Training time: ~3-7 hours on A100, ~10-20 hours on V100
- 64.7% of params are in embeddings (unavoidable with 50K vocab at this scale)

## Known Issues / TODOs

1. **Hardcoded Colab paths** — `token_frequency.py`, `extract_features.py`, `analyze_results.py`, and `extract_embedded_norm` all reference `/content/drive/MyDrive/llm`. Must be updated to use `PROJECT_ROOT`-relative paths (see `train.py` for the pattern).
2. **minGPT submodule uninitialized** — Run `git submodule update --init` after cloning.
3. **`extract_embedded_norm` missing .py extension** — Should be renamed to `extract_embedded_norm.py`.

## Development Workflow

- Use **plan mode** for non-trivial changes before implementing
- Use **agent teams** for deep research tasks (e.g., scaling law analysis, architecture decisions)
- When modifying `train.py`, always **preserve checkpoint resume compatibility** (new checkpoints must load old ones)
- Always check `nvidia-smi` output before suggesting any GPU commands
- Multiple model size configs exist: 7M, 30M, 124M (in feature scripts) and 100M-500M (in train.py)
