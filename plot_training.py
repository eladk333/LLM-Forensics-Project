"""
Plot training curves from training_log.csv and checkpoint eval history.
Generates 4 plots:
  1. Training loss over optimizer steps
  2. Validation loss over optimizer steps
  3. Validation perplexity over optimizer steps
  4. Learning rate schedule

Usage:
    python plot_training.py                          # default: 100M_Context1024
    python plot_training.py --model_dir 500M_Context1024
    python plot_training.py --save                   # save to PNG instead of showing
"""

import os
import sys
import argparse
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_csv_log(model_dir):
    """Load training_log.csv into a DataFrame."""
    csv_path = os.path.join(model_dir, 'training_log.csv')
    if not os.path.exists(csv_path):
        print(f"No training log found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    # Deduplicate rows from resumed training runs
    if 'opt_step' in df.columns:
        df = df.drop_duplicates(subset='opt_step', keep='last')
    # Convert numeric columns (they may be stored as strings)
    for col in ['opt_step', 'epoch', 'tokens_seen']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['train_loss', 'val_loss', 'perplexity', 'lr']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded {len(df)} log entries from {csv_path}")
    return df


def load_checkpoint_eval_history(model_dir):
    """Load eval_history from the latest checkpoint."""
    ckpt_files = glob.glob(os.path.join(model_dir, '*.pt'))
    if not ckpt_files:
        return None

    latest = max(ckpt_files, key=os.path.getmtime)
    print(f"Loading eval history from: {os.path.basename(latest)}")

    ckpt = torch.load(latest, map_location='cpu')
    history = ckpt.get('eval_history', [])
    if not history:
        return None

    df = pd.DataFrame(history)
    print(f"  {len(df)} evaluation points found")
    return df


def plot_training_curves(df_log, df_eval, model_dir, save=False):
    """Generate training visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    model_name = os.path.basename(model_dir)

    # --- Plot 1: Training Loss ---
    ax = axes[0, 0]
    if df_log is not None:
        train_rows = df_log[df_log['train_loss'].notna()]
        if not train_rows.empty:
            # Use tokens_seen for x-axis if available, else opt_step
            if 'tokens_seen' in train_rows.columns and train_rows['tokens_seen'].notna().any():
                x = train_rows['tokens_seen'] / 1e9
                ax.set_xlabel('Tokens (Billions)')
            else:
                x = train_rows['opt_step']
                ax.set_xlabel('Optimizer Step')

            ax.plot(x, train_rows['train_loss'], alpha=0.3, color='blue', linewidth=0.5)
            # Smoothed line (rolling average)
            window = max(1, len(train_rows) // 100)
            smoothed = train_rows['train_loss'].rolling(window=window, min_periods=1).mean()
            ax.plot(x, smoothed, color='blue', linewidth=2, label=f'Train Loss (smoothed, w={window})')
            ax.legend()
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Validation Loss ---
    ax = axes[0, 1]
    has_val = False
    if df_eval is not None and 'val_loss' in df_eval.columns:
        if 'tokens_seen' in df_eval.columns:
            x = df_eval['tokens_seen'] / 1e9
            ax.set_xlabel('Tokens (Billions)')
        else:
            x = df_eval['opt_step']
            ax.set_xlabel('Optimizer Step')

        ax.plot(x, df_eval['val_loss'], 'o-', color='red', linewidth=2, markersize=4, label='Val Loss')
        best_idx = df_eval['val_loss'].idxmin()
        ax.axhline(y=df_eval.loc[best_idx, 'val_loss'], color='green', linestyle='--',
                    alpha=0.5, label=f"Best: {df_eval.loc[best_idx, 'val_loss']:.4f}")
        ax.legend()
        has_val = True
    elif df_log is not None:
        val_rows = df_log[df_log['val_loss'].notna()]
        if not val_rows.empty:
            if 'tokens_seen' in val_rows.columns and val_rows['tokens_seen'].notna().any():
                x = val_rows['tokens_seen'] / 1e9
                ax.set_xlabel('Tokens (Billions)')
            else:
                x = val_rows['opt_step']
                ax.set_xlabel('Optimizer Step')
            ax.plot(x, val_rows['val_loss'], 'o-', color='red', linewidth=2, markersize=4, label='Val Loss')
            ax.legend()
            has_val = True

    if not has_val:
        ax.text(0.5, 0.5, 'No validation data yet', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Perplexity ---
    ax = axes[1, 0]
    has_ppl = False
    if df_eval is not None and 'val_perplexity' in df_eval.columns:
        if 'tokens_seen' in df_eval.columns:
            x = df_eval['tokens_seen'] / 1e9
            ax.set_xlabel('Tokens (Billions)')
        else:
            x = df_eval['opt_step']
            ax.set_xlabel('Optimizer Step')
        ax.plot(x, df_eval['val_perplexity'], 's-', color='purple', linewidth=2, markersize=4, label='Perplexity')
        ax.legend()
        has_ppl = True
    elif df_log is not None and 'perplexity' in df_log.columns:
        ppl_rows = df_log[df_log['perplexity'].notna()]
        if not ppl_rows.empty:
            if 'tokens_seen' in ppl_rows.columns:
                x = ppl_rows['tokens_seen'] / 1e9
                ax.set_xlabel('Tokens (Billions)')
            else:
                x = ppl_rows['opt_step']
                ax.set_xlabel('Optimizer Step')
            ax.plot(x, ppl_rows['perplexity'], 's-', color='purple', linewidth=2, markersize=4, label='Perplexity')
            ax.legend()
            has_ppl = True

    if not has_ppl:
        ax.text(0.5, 0.5, 'No perplexity data yet', ha='center', va='center', transform=ax.transAxes)
    ax.set_ylabel('Perplexity')
    ax.set_title('Validation Perplexity')
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Learning Rate ---
    ax = axes[1, 1]
    if df_log is not None and 'lr' in df_log.columns:
        lr_rows = df_log[df_log['lr'].notna()]
        if not lr_rows.empty:
            ax.plot(lr_rows['opt_step'], lr_rows['lr'], color='orange', linewidth=2)
    ax.set_xlabel('Optimizer Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('LR Schedule')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Training Curves: {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save:
        out_path = os.path.join(model_dir, 'training_curves.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()


def plot_checkpoint_comparison(model_dir, save=False):
    """Plot val loss across all saved checkpoints (from their metadata)."""
    ckpt_files = sorted(glob.glob(os.path.join(model_dir, '*.pt')))
    if not ckpt_files:
        print("No checkpoints found.")
        return

    data = []
    for f in ckpt_files:
        ckpt = torch.load(f, map_location='cpu')
        if isinstance(ckpt, dict) and 'val_loss' in ckpt:
            data.append({
                'file': os.path.basename(f),
                'opt_step': ckpt.get('opt_step', 0),
                'val_loss': ckpt.get('val_loss', None),
                'val_perplexity': ckpt.get('val_perplexity', None),
                'tokens_seen': ckpt.get('tokens_seen', 0),
            })

    if not data:
        print("No checkpoints with eval data found.")
        return

    df = pd.DataFrame(data).sort_values('opt_step')
    print(f"Found {len(df)} checkpoints with eval data")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_name = os.path.basename(model_dir)

    ax1.plot(df['tokens_seen'] / 1e9, df['val_loss'], 'D-', color='red', markersize=6)
    ax1.set_xlabel('Tokens (Billions)')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Val Loss per Checkpoint')
    ax1.grid(True, alpha=0.3)

    if df['val_perplexity'].notna().any():
        ax2.plot(df['tokens_seen'] / 1e9, df['val_perplexity'], 'D-', color='purple', markersize=6)
        ax2.set_xlabel('Tokens (Billions)')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity per Checkpoint')
        ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Checkpoint Comparison: {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        out_path = os.path.join(model_dir, 'checkpoint_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--model_dir', default='100M_Context1024',
                        help='Model subdirectory under data/models/ (default: 100M_Context1024)')
    parser.add_argument('--save', action='store_true',
                        help='Save plots as PNG instead of displaying')
    parser.add_argument('--checkpoints', action='store_true',
                        help='Also plot checkpoint-by-checkpoint comparison')
    args = parser.parse_args()

    model_dir = os.path.join(PROJECT_ROOT, 'data', 'models', args.model_dir)
    if not os.path.exists(model_dir):
        print(f"Directory not found: {model_dir}")
        print("Available models:")
        models_root = os.path.join(PROJECT_ROOT, 'data', 'models')
        if os.path.exists(models_root):
            for d in os.listdir(models_root):
                if os.path.isdir(os.path.join(models_root, d)):
                    print(f"  {d}")
        sys.exit(1)

    # Load data from both sources
    df_log = load_csv_log(model_dir)
    df_eval = load_checkpoint_eval_history(model_dir)

    if df_log is None and df_eval is None:
        print("No training data found. Train the model first.")
        sys.exit(1)

    # Plot training curves
    plot_training_curves(df_log, df_eval, model_dir, save=args.save)

    # Optionally plot checkpoint comparison
    if args.checkpoints:
        plot_checkpoint_comparison(model_dir, save=args.save)
