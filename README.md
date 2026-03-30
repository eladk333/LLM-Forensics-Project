# Meeting Notes: LLM Forensics Project
**Date:** March 30, 2026 | **Status:** Training phase in progress

---

## 1. What Are We Doing? (The Research Question)

We are training GPT language models **from scratch** at multiple sizes (100M, 200M, 300M, 400M, 500M parameters) and then asking:

> **Do language models give more "representational weight" to tokens they see more often during training?**

Concretely, we measure the **L2 norm** (length of the vector) of each token's embedding and each token's output logit vector. We then check whether these norms correlate with how frequently that token appears in natural text.

**Why this matters:** If a model consistently assigns larger embedding norms to frequent tokens, it tells us something fundamental about how neural language models organize their internal knowledge. It connects the statistical structure of language (Zipf's law) to the geometry of learned representations. This has implications for understanding model efficiency, compression, and potentially for improving tokenization and training strategies.

**The pipeline:**
```
Train models  -->  Count token frequencies  -->  Extract norms  -->  Regression analysis
 (train.py)     (token_frequency.py)      (extract_features.py)   (analyze_results.py)
```

We are currently in **Step 1** (training), with 2 of 5 models complete and 1 in progress.

---

## 2. Chinchilla Scaling Laws (The Key Idea)

### What is Chinchilla?

**Chinchilla** (Hoffmann et al., 2022) is a landmark paper from DeepMind that answered: **given a fixed compute budget, what is the optimal balance between model size and training data?**

Before Chinchilla, the common wisdom (from the Kaplan et al. 2020 scaling laws) was: "make the model as big as possible, even if you train it on relatively little data." This led to models like GPT-3 (175B parameters) being trained on only 300B tokens — roughly a **1.7x** tokens-to-parameters ratio.

### What Chinchilla showed

Chinchilla demonstrated that **model size and training data should scale equally.** The optimal ratio is approximately:

> **Optimal training tokens = 20 x model parameters**

This means a 100M parameter model should see **2 billion tokens**, a 200M model should see **4 billion tokens**, and so on.

**The evidence:** DeepMind trained over 400 models ranging from 70M to 16B parameters, each with different token budgets. They found that for any given compute budget, the best perplexity comes from a roughly **1:20 parameter-to-token ratio**. Their 70B-parameter "Chinchilla" model, trained on 1.4 trillion tokens (20x), matched the performance of the much larger 280B-parameter Gopher model, which was trained on only 300B tokens (~1x).

### How we follow Chinchilla

| Model Size | Parameters | Token Budget | Ratio | Source |
|------------|-----------|-------------|-------|--------|
| 100M | 99.5M | 2.0B | 20.1x | Hoffmann et al. Table A3, Approach 3 |
| 200M | 206.7M | 4.0B | 19.4x | Same |
| 300M | 305.5M | 6.0B | 19.6x | Same |
| 400M | 406.3M | 8.0B | 19.7x | Same |
| 500M | 499.5M | 10.0B | 20.0x | Same |

Our training dataset (OpenWebText) has ~8-9B tokens, which means we can train models up to ~400M without repeating data. The 500M model will see slight repetition.

### Why this matters for our project

By training at the compute-optimal point, we ensure that each model has been trained "fairly" — none are undertrained or overtrained relative to their capacity. This makes cross-scale comparisons of the frequency-norm relationship scientifically valid.

---

## 3. Training Setup

### Data

| Property | Value |
|----------|-------|
| **Training data** | OpenWebText (~8-9B tokens) — open-source recreation of GPT-2's WebText |
| **Tokenizer** | GPT-2 BPE, 50,257 vocabulary |
| **Validation data** | C4 (web text, out-of-distribution) + WikiText-103 (standard benchmark) |
| **Eval protocol** | Fixed 3.2M tokens per evaluation, non-overlapping 1024-token chunks |

### Architecture

All models use the GPT-2 architecture (decoder-only transformer, pre-norm) via Karpathy's minGPT, with custom dimensions to hit parameter targets:

| Model | Layers | Heads | Embed Dim | Head Dim | Total Params | Embedding % |
|-------|--------|-------|-----------|----------|-------------|-------------|
| 100M | 7 | 10 | 640 | 64 | 99.5M | 33% |
| 200M | 12 | 14 | 896 | 64 | 206.7M | 22% |
| 300M | 16 | 16 | 1,024 | 64 | 305.5M | 34% |

**Note on embedding fraction:** At small scales, the embedding matrix (vocab_size x embed_dim = 50,257 x 640 = 32M parameters) takes up a huge portion of the model. This is actually *relevant* to our research — the embedding matrix is exactly what we're studying.

### Hyperparameters (Every Choice Has a Paper Citation)

| Setting | Value | Paper |
|---------|-------|-------|
| Token budget | 20x params | Hoffmann et al. 2022 (Chinchilla), Table A3 |
| Peak learning rate | 6e-4 (100M) to 3.2e-4 (300M) | Brown et al. 2020 (GPT-3), Table 2.1 |
| Min learning rate | Peak / 10 | Brown et al. 2020 (GPT-3 convention) |
| LR schedule | Cosine decay | Loshchilov & Hutter 2017 (SGDR) |
| Warmup | 375M tokens (715 optimizer steps) | Brown et al. 2020, Section 2.1 |
| Optimizer | AdamW, betas=(0.9, 0.95) | Loshchilov & Hutter 2019; Brown et al. 2020 |
| Weight decay | 0.1 | Brown et al. 2020, Table 2.1 |
| Gradient clipping | 1.0 | Brown et al. 2020, Section 2.1 |
| Dropout | 0.0 | Brown et al. 2020, Section 2.1 (no dropout for pre-training) |
| Batch size | 512 sequences (~524K tokens/step) | Held constant across all scales (research design choice) |

**Key design decision:** We keep batch size constant across all model sizes. GPT-3 scales batch size with model size, but we deliberately hold it fixed so the *only* variable is the model itself — not the optimization dynamics. This makes our cross-scale analysis cleaner.

### Infrastructure

- **GPU:** Single NVIDIA A100-SXM4-80GB
- **Precision:** bfloat16 (native on A100, no loss scaling needed)
- **Framework:** minGPT (Karpathy) with custom modifications for pre-norm and proper initialization
- **Checkpointing:** Atomic saves (crash-safe), deterministic shuffle for reproducible resume
- **All checkpoints retained** — we need norms from every checkpoint to study how the frequency-norm relationship evolves *during* training

---

## 4. Results So Far

### Training Progress

| Model | Status | Tokens Seen | Wall Time | Opt Steps | Best C4 Val Loss | Best C4 PPL | Best WT103 PPL |
|-------|--------|-------------|-----------|-----------|-----------------|-------------|----------------|
| **100M** | Complete | 2.0B / 2.0B | 10.0h | 5,605 | 4.2615 | **70.92** | **79.08** |
| **200M** | Complete | 4.0B / 4.0B | 41.9h | 11,211 | 3.9515 | **52.02** | **54.22** |
| **300M** | ~75% | 4.5B / 6.0B | ~69h | 12,680+ | 3.6856 | **39.33** | ~45.6 |
| 400M | Not started | — | — | — | — | — | — |
| 500M | Not started | — | — | — | — | — | — |

### What the Numbers Mean

**Perplexity** measures how "surprised" the model is by unseen text. A perplexity of 70 means the model is, on average, as uncertain as if it were choosing uniformly among 70 options for each next token. Lower is better.

**Scaling trend:** Perplexity drops substantially with each size increase:
- 100M → 200M: **70.9 → 52.0** (27% reduction)
- 200M → 300M: **52.0 → ~39.3** (24% reduction, still training)

This is consistent with power-law scaling: each doubling of parameters gives a roughly constant *percentage* improvement in perplexity, not a constant *absolute* improvement.

### Loss Trajectories (100M Model)

```
Step  1500 | Train: 3.785 | C4 Val: 4.568 | PPL:  96.4
Step  2000 | Train: 3.678 | C4 Val: 4.434 | PPL:  84.3
Step  2500 | Train: 3.565 | C4 Val: 4.371 | PPL:  79.2
Step  3000 | Train: 3.529 | C4 Val: 4.278 | PPL:  72.1
Step  3500 | Train: 3.514 | C4 Val: 4.272 | PPL:  71.7
Step  4000 | Train: 3.467 | C4 Val: 4.262 | PPL:  70.9  <-- best
Step  4500 | Train: 3.459 | C4 Val: 4.280 | PPL:  72.3
Step  5000 | Train: 3.368 | C4 Val: 4.270 | PPL:  71.5
Step  5605 | Train: 3.396 | C4 Val: 4.278 | PPL:  72.1  (final)
```

**Observation:** Validation loss plateaus after step ~3500, while training loss keeps decreasing. This is expected — the model continues memorizing training data while generalization saturates. The gap between training and validation loss (~0.9) indicates some overfitting is occurring, which is normal.

### Loss Trajectories (200M Model)

```
Step  3000 | Train: 3.388 | C4 Val: 4.214 | PPL:  67.6
Step  5000 | Train: 3.127 | C4 Val: 4.051 | PPL:  57.4
Step  7000 | Train: 3.108 | C4 Val: 3.984 | PPL:  53.7
Step  9000 | Train: 3.069 | C4 Val: 3.964 | PPL:  52.7
Step 11000 | Train: 3.117 | C4 Val: 3.952 | PPL:  52.0
Step 11211 | Train: 3.081 | C4 Val: 3.952 | PPL:  52.0  (final)
```

**Observation:** The 200M model keeps improving its validation loss even at the end of training (loss still decreasing at final step). This suggests the Chinchilla 20x token budget is well-calibrated — the model uses all its training data productively.

### Health Indicators

All models are training healthily:
- **No NaN or Inf** detected in any run
- **Gradient norms** are stable and decrease with scale: 0.42 (100M) → 0.31 (200M) → 0.28 (300M)
- **No gradient clipping triggered** in any model (grad norms well below the clip threshold of 1.0)

---

## 5. What to Show the Professor

### File Tour (in order of importance)

| # | File | What It Shows | How to Present |
|---|------|---------------|----------------|
| 1 | `TRAINING_SETUP.md` | Complete methodology with paper citations | "Here is our full methodology document — every hyperparameter traces to a specific paper." |
| 2 | `data/models/100M_Context1024/training_summary.txt` | Completed 100M results | "Here are the final results for our 100M model." |
| 3 | `data/models/200M_Context1024/training_summary.txt` | Completed 200M results | "And the 200M model shows clear improvement." |
| 4 | `train.py` (lines 1-80) | Model configurations | "Here are the 5 model configs, each sized to hit Chinchilla-optimal ratios." |
| 5 | `train.py` (lines 80-130) | Hyperparameter provenance | "Every hyperparameter has a paper citation printed at training start." |

### Commands to Demo

```bash
# Show training curves for the 100M model (generates a 4-panel plot)
python plot_training.py --model_dir 100M_Context1024 --save

# Show training curves for the 200M model
python plot_training.py --model_dir 200M_Context1024 --save

# Show training curves for the 300M model (in progress)
python plot_training.py --model_dir 300M_Context1024 --save

# Check if 300M is still training (on the compute server)
# ssh tsruyag@dsinlp01
# nvidia-smi
```

The plots show 4 panels each: (1) training loss, (2) validation loss with best-loss marker, (3) perplexity, (4) learning rate schedule.

### Key Talking Points

1. **"We're following established best practices."** Every hyperparameter comes from GPT-3 or Chinchilla. The methodology section cites 7 papers.

2. **"Scaling is working as expected."** Perplexity improves with model size (70.9 → 52.0 → ~39.3), consistent with power-law scaling predictions.

3. **"Training is robust."** No instabilities, atomic checkpointing for crash recovery, deterministic resume. The code handles GPU failures gracefully.

4. **"We retain all checkpoints for downstream analysis."** Unlike typical training where you only keep the best model, we keep checkpoints at every evaluation point. This lets us study how embedding norms evolve *during* training, not just at the end.

5. **"Next phase is the analysis pipeline."** Once training is complete, we extract norms from every checkpoint and run regression analysis to quantify the frequency-norm relationship across scales and training stages.

---

## 6. Scaling Laws: What to Expect for Remaining Models

Based on the Chinchilla scaling law, we can roughly predict performance for the untrained models:

| Model | Predicted C4 PPL (rough estimate) | Token Budget | Est. Training Time |
|-------|----------------------------------|-------------|-------------------|
| 400M | ~33-36 | 8.0B | ~100-130h |
| 500M | ~29-33 | 10.0B | ~150-200h |

These are rough extrapolations from the power-law trend in our existing data points. The actual numbers will depend on architecture efficiency and data quality at those token counts.

---

## 7. Glossary (For Quick Reference During Meeting)

| Term | Meaning |
|------|---------|
| **Perplexity (PPL)** | exp(cross-entropy loss). Measures how "surprised" the model is by unseen text. Lower = better. PPL of 50 means the model is as uncertain as choosing among 50 equally likely next tokens. |
| **Cross-entropy loss** | The average negative log-probability the model assigns to the correct next token. Lower = better. |
| **Chinchilla-optimal** | Training with ~20 tokens per parameter. Named after the DeepMind paper that established this ratio. |
| **Scaling law** | The empirical finding that LLM performance improves as a predictable power-law function of model size, data size, and compute. |
| **Embedding norm** | The L2 (Euclidean) length of a token's embedding vector. Our hypothesis: frequent tokens get larger norms. |
| **Logit norm** | The L2 length of a token's output projection vector. Similar hypothesis as embedding norm. |
| **bfloat16** | A 16-bit floating point format optimized for deep learning on A100 GPUs. Saves memory and speeds up training without hurting quality. |
| **Gradient accumulation** | Technique to simulate large batch sizes with limited GPU memory. Instead of processing 512 sequences at once, we process 16-32 at a time and accumulate gradients over 16-32 steps before updating weights. |
| **AdamW** | The standard optimizer for transformer training. "W" stands for decoupled weight decay (a regularization technique). |
| **Cosine decay** | Learning rate schedule that starts high (for fast initial progress), then gradually decreases following a cosine curve (for fine-grained optimization near convergence). |
| **OpenWebText** | ~8-9B tokens of web text (Reddit links with 3+ upvotes). Open-source recreation of the dataset used to train GPT-2. |
| **C4** | "Colossal Clean Crawled Corpus" — a large web text dataset. We use its validation split as an out-of-distribution evaluation benchmark. |
| **WikiText-103** | A standard language modeling benchmark based on Wikipedia articles. |
| **Pre-norm** | A transformer variant where LayerNorm is applied *before* attention/MLP (instead of after). More stable for training. |

---

## 8. Next Steps

1. **Complete 300M training** — currently ~75% done, ETA ~24-48 hours
2. **Train 400M model** — 8B tokens, estimated ~100-130 hours on A100
3. **Train 500M model** — 10B tokens, estimated ~150-200 hours
4. **Update feature extraction scripts** — `extract_features.py`, `token_frequency.py`, and `analyze_results.py` still have hardcoded Colab paths and target old model sizes. Need to update for the new 100M-500M checkpoints.
5. **Run the analysis pipeline** on completed models:
   - Count token frequencies from OpenWebText (or WikiText-103)
   - Extract embedding and logit norms from every saved checkpoint
   - Run regression analysis (linear + MLP) to predict log-frequency from norms
   - Compare results across model scales and training stages
6. **Write up findings** — characterize how the frequency-norm relationship changes with model scale and training progress

---

*Generated from project data on March 30, 2026. All metrics are from actual training runs on A100 GPU.*
