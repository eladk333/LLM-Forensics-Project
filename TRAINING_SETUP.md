# Training Methodology: Token Frequency and Model Internal Representations

## 1. Research Goal

This project investigates how **token frequency** in natural language corpora relates to the **internal representations** learned by GPT-style language models. Specifically, we train GPT models from scratch at five scales (100M to 500M parameters), then measure the L2 norm of each token's embedding vector and output logit vector. By correlating these norms with corpus frequency, we aim to characterize how model scale affects the frequency--representation relationship.

The core question: do language models allocate representational capacity (measured by vector norm) in proportion to how often a token appears in training data, and how does this allocation change as models grow?

---

## 2. Training Data

| Property | Value |
|----------|-------|
| Dataset | OpenWebText |
| Source | Open-source recreation of OpenAI's WebText (Reddit links with 3+ karma) |
| Size | ~8--9 billion tokens |
| Tokenizer | GPT-2 BPE (50,257 vocabulary) |
| Access | HuggingFace Datasets (`openwebtext`) |

**Why OpenWebText.** Three properties make it well-suited to this study:

1. **Sufficient scale.** At ~8--9B tokens, it supports Chinchilla-optimal training (20 tokens per parameter) for models up to 500M without significant data repetition.
2. **Well-understood corpus.** OpenWebText is widely used in the LLM research community, making our results comparable to prior work on GPT-2-scale models.
3. **Natural Zipfian distribution.** As a broad web corpus, token frequencies follow the power-law distribution characteristic of natural language, which is essential for studying the frequency--norm relationship across the full frequency spectrum.

---

## 3. Model Configurations

All models share the GPT-2 BPE tokenizer (50,257 tokens) and a context length of 1,024 tokens. Architecture dimensions (layers, heads, embedding size) are custom-sized to hit target parameter counts; they do not correspond to standard model tables from published work.

| Model | Layers | Heads | Embed Dim | Head Dim | Approx Params | Embedding % | Token Budget | Peak LR | Eff. Batch (seqs) |
|-------|--------|-------|-----------|----------|---------------|-------------|-------------|---------|-------------------|
| 100M  | 7      | 10    | 640       | 64       | 99.5M         | ~65%        | 2B          | 6.0e-4  | 512               |
| 200M  | 12     | 14    | 896       | 64       | 206.7M        | ~44%        | 4B          | 4.5e-4  | 512               |
| 300M  | 16     | 16    | 1,024     | 64       | 305.5M        | ~34%        | 6B          | 3.2e-4  | 512               |
| 400M  | 24     | 16    | 1,024     | 64       | 406.3M        | ~26%        | 8B          | 3.0e-4  | 512               |
| 500M  | 24     | 16    | 1,152     | 72       | 499.5M        | ~23%        | 10B         | 2.5e-4  | 512               |

The high embedding fraction at small scales (65% for the 100M model) is an inherent consequence of using a 50K vocabulary with narrow hidden dimensions. This is unavoidable at these scales and is itself relevant to our research question: the embedding matrix dominates the parameter budget, so understanding how its norms relate to frequency is especially important.

---

## 4. Scaling Law Compliance

Every hyperparameter choice traces to a specific published result. The table below provides full provenance.

| Hyperparameter | Value | Source | Citation Detail |
|----------------|-------|--------|-----------------|
| Token budget (20x params) | 2B--10B | Hoffmann et al., 2022 | Table A3, Approach 3 (compute-optimal ratio) |
| Peak learning rate | 6.0e-4 to 2.5e-4 | Brown et al., 2020 | Table 2.1 (LR by model size) |
| Weight decay | 0.1 | Brown et al., 2020 | Table 2.1 |
| Adam betas | (0.9, 0.95) | Brown et al., 2020 | Table 2.1 |
| Gradient clipping | 1.0 | Brown et al., 2020 | Section 2.1 |
| Warmup | 375M tokens (715 optimizer steps) | Brown et al., 2020 | Section 2.1 (fixed token warmup) |
| LR schedule shape | Cosine decay | Loshchilov and Hutter, 2017 | SGDR cosine annealing |
| Min LR floor | 10% of peak | Brown et al., 2020 | GPT-3 convention; also used in Chinchilla and LLaMA |
| Dropout | 0.0 | Brown et al., 2020 | Section 2.1 (no dropout for pre-training) |
| Context length | 1,024 | Radford et al., 2019 | GPT-2 default context window |
| Optimizer | AdamW | Loshchilov and Hutter, 2019 | Decoupled weight decay regularization |

**Important caveats:**

- **Architecture configurations (L/H/E) are custom.** The layer counts, head counts, and embedding dimensions were chosen to hit specific parameter targets. They do not come from standard model tables in GPT-2 or GPT-3.
- **Constant effective batch size (512 sequences) is a deliberate research design choice.** GPT-3 scales batch size with model size; we hold it fixed so that the only variable across model sizes is the model itself, not the optimization dynamics.
- **500M peak LR (2.5e-4) is rounded.** GPT-3 Table 2.1 gives 2.5e-4 for the 760M model. True log-interpolation between the 350M anchor (3.0e-4) and the 760M anchor (2.5e-4) would give approximately 2.75e-4 for a 500M model. We use the 760M value directly as a conservative choice.
- **Warmup fraction varies with model size.** The 375M-token warmup is 18.7% of the 100M model's 2B training budget but only 3.75% of the 500M model's 10B budget. This matches GPT-3's approach of using a fixed token warmup across all sizes, but it means smaller models spend proportionally more time in warmup.

---

## 5. Evaluation Methodology

We evaluate on two held-out corpora to measure both in-domain generalization and transfer to a standard benchmark.

| Evaluation Set | Source | Purpose |
|----------------|--------|---------|
| C4 validation | AllenAI C4 English validation split | Out-of-distribution web text; standard in modern LLM practice (OLMo, Pythia) |
| WikiText-103 validation | Merity et al., 2017 | Standard language modeling benchmark; enables comparison with published results |

**Evaluation protocol:**

- **Concatenated token chunks.** Validation text is concatenated into a single token stream and chunked into non-overlapping blocks of 1,024 tokens. There is no padding, and every token contributes to the loss. This matches the training format and avoids padding-induced bias.
- **Fixed evaluation size.** We evaluate on a fixed token count (~3.2M tokens, corresponding to approximately 3,100 chunks of 1,024 tokens) across all model sizes. This ensures perplexity measurements are directly comparable.
- **Evaluation frequency.** Models are evaluated at regular optimizer-step intervals (500 steps for 100M, scaling up to 2,000 steps for 400M--500M), yielding approximately 7--10 evaluation points per training run. An additional evaluation is performed at the end of each epoch.

**Reported metric:** Validation perplexity, computed as exp(mean cross-entropy loss) over the evaluation chunks.

---

## 6. Training Infrastructure

| Component | Detail |
|-----------|--------|
| Hardware | Single NVIDIA A100-SXM4-80GB GPU |
| Mixed precision | bfloat16 (native on Ampere+; no loss scaling needed) |
| Framework | Karpathy's minGPT with custom Block for correct c_proj initialization |
| Gradient accumulation | 16--64 micro-batches per optimizer step (model-size dependent) |
| Effective batch | 512 sequences = ~524K tokens per optimizer step (all model sizes) |
| Checkpoint saving | Atomic writes via POSIX os.replace to prevent corruption on interruption |
| Data ordering | Seeded shuffle (seed=3407) for reproducible resume after checkpoint reload |
| Data loading | 8 persistent workers, pin memory, memory-mapped Arrow dataset |

The script auto-detects GPU architecture: bfloat16 on Ampere and newer (CUDA capability >= 8.0), float16 with GradScaler on Volta (V100). All results reported in this study use bfloat16 on A100.

---

## 7. Results Structure

Each model size produces a self-contained checkpoint directory under `data/models/{SIZE}_Context1024/`. The contents are:

| File | Description |
|------|-------------|
| `config.json` | Complete training configuration snapshot (architecture, hyperparameters, hardware, software versions). Written once on the first run and never overwritten. |
| `training_log.csv` | Per-step training log with columns: optimizer step, epoch, training loss, validation loss, perplexity, learning rate, tokens seen. |
| `training_summary.txt` | Human-readable summary of the completed training run. |
| `best_model.pt` | Model weights from the optimizer step with the lowest validation loss. Contains model state dict, evaluation metrics, and configuration. |
| `ckpt_opt_{step}.pt` | Full training checkpoints saved at each evaluation interval. Contains model weights, optimizer state, scheduler state, AMP scaler state, and all training counters. All checkpoints are retained (not pruned) because the downstream analysis extracts embedding norms at each checkpoint to study how the frequency--norm relationship evolves during training. |
| `stdout.log` | Console output captured during training (when run under output redirection). |

---

## 8. Analysis Pipeline (Downstream)

The training phase produces model checkpoints. The analysis pipeline then proceeds through three additional stages:

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Train models | `train.py` | OpenWebText | Checkpoints with model weights at each evaluation step |
| 2. Count token frequencies | `token_frequency.py` | WikiText-103 (reference corpus) | Per-token frequency counts across the full 50,257-token vocabulary |
| 3. Extract norms | `extract_features.py` | Trained checkpoints | Per-token embedding norms (L2) and logit norms from each checkpoint |
| 4. Regression analysis | `analyze_results.py` | Frequency counts + norms | Regression predicting log-frequency from embedding/logit norms (linear and MLP) |

By running stages 3 and 4 on every saved checkpoint across all five model sizes, we can characterize how the frequency--norm relationship (a) differs across model scales and (b) evolves over the course of training.

---

## 9. Key References

- **Brown, T. B., Mann, B., Ryder, N., Subbiah, M., et al.** (2020). Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, 33. [GPT-3 hyperparameters: Table 2.1, Section 2.1]

- **Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., et al.** (2022). Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems*, 35. [Chinchilla scaling laws: Table A3, Approach 3]

- **Loshchilov, I. and Hutter, F.** (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *International Conference on Learning Representations*. [Cosine annealing schedule]

- **Loshchilov, I. and Hutter, F.** (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*. [AdamW optimizer]

- **Merity, S., Xiong, C., Bradbury, J., and Socher, R.** (2017). Pointer Sentinel Mixture Models. *International Conference on Learning Representations*. [WikiText-103 benchmark]

- **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I.** (2019). Language Models are Unsupervised Multitask Learners. *OpenAI Technical Report*. [GPT-2 architecture and tokenizer]

- **Karpathy, A.** (2021). minGPT. GitHub repository. [Training framework]
