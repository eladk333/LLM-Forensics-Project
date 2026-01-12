import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'minGPT'))
import torch
import math
import glob
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed


# We set const seed so the training will be deterministic
set_seed(3407)

# Model settings
N_LAYER = 12   # Increased from 6
N_HEAD  = 12   # Increased from 6
N_EMBD  = 768  # Increased from 384
BATCH_SIZE = 32 # Same for all models


# N_LAYER = 6   # Layers
# N_HEAD  = 6   # Attention heads
# N_EMBD  = 384  # Embedding Dimension




# # Model settings
# N_LAYER = 4   # Layers
# N_HEAD  = 4   # Heads
# N_EMBD  = 128  # Embedding Dimensio

# Paths
CHECKPOINT_FOLDER_PATH = os.path.join(os.getcwd(), 'data', 'models', '124M') # Path for the model checkpoints
DATA_CACHE_PATH = os.path.join(os.getcwd(), 'data', 'datasets') # Path for the dataset
os.makedirs(CHECKPOINT_FOLDER_PATH, exist_ok=True) # If the model folder doesn't exist it creates it

# The dataset class
class WikiDataset(Dataset):
    def __init__(self, split='train', block_size=128):
        self.block_size = block_size # Context length of how many tokens the model can look at the predict the next one
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        cache_folder = DATA_CACHE_PATH
        os.makedirs(cache_folder, exist_ok=True)
        cache_path = os.path.join(cache_folder, 'wiki_103_full_cache.pt') # Just create the full path for the dataset

        # If we already tokenized and saved the dataset, just load it
        if os.path.exists(cache_path):
            print(f"Found cached data, loading from {cache_path}...")
            self.tokens = torch.load(cache_path) # Load dataset from path
            print(f"Loaded {len(self.tokens)} tokens.")
        # If not we tokenize and it and save it for next time
        else:
            print(f"Cache not found. Loading WikiText-103 ({split})...")
            dataset = load_dataset("wikitext", "wikitext-103-v1", split=split) # Downloads Wikitext-103 to local disk

            print("Tokenizing data...")
            cleaned_articles = [] # Empty list to later hold our cleaned atricles

            # Loops through all the articles in the dataset
            for x in dataset:
                article_text = x['text'] # Get the text of the current article

                # If the article not empty add to the list
                if len(article_text) > 0:
                    cleaned_articles.append(article_text)
            text_data = "\n".join(cleaned_articles) # Glues all the articles into 1 string with /n separting them
            self.tokens = self.tokenizer.encode(text_data) # Tokenize the data
            torch.save(self.tokens, cache_path)

        print(f"Total tokens in dataset: {len(self.tokens)}")

    # Number of sequences we need for the model to see the entire dataset (1 epoch)
    def __len__(self):
        return len(self.tokens) // self.block_size

    # Get's a block of the data accoriding to the index
    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        # If the remaining tokens are less than block_size we shift the start index to return full sized sequnance (block_size size)
        if start_idx + self.block_size + 1 > len(self.tokens):
            start_idx = len(self.tokens) - self.block_size - 1

        chunk = self.tokens[start_idx : start_idx + self.block_size + 1] # Takes a slice of the token from the start index until block_size + 1 for the target token

        dix = torch.tensor(chunk, dtype=torch.long) # Converts the list to the type our NN can work with
        x = dix[:-1] # The input of the model
        y = dix[1:] # The expected output of the model
        return x, y

# To track the lowest loss achieved during training, used to see if the model is still learning or got stuck
BEST_LOSS = float('inf')


if __name__ == '__main__':
    print("Loading dataset")
    train_dataset = WikiDataset('train', block_size=128) # Initlize the dataset class

    # Calculation of total epoch size
    total_samples = len(train_dataset) # How many blocks we have of the data
    iters_for_one_epoch = total_samples // BATCH_SIZE # How many steps (iterations) we need for 1 epoch

    # Checking for existing checkpoints
    checkpoint_files = glob.glob(os.path.join(CHECKPOINT_FOLDER_PATH, 'ckpt_step_*.pt'))

    start_iter = 0
    latest_ckpt_path = None

    if checkpoint_files:
        # Find file with max step number
        latest_ckpt_path = max(checkpoint_files, key=lambda x: int(re.search(r'ckpt_step_(\d+).pt', x).group(1)))
        start_iter = int(re.search(r'ckpt_step_(\d+).pt', latest_ckpt_path).group(1))

        print(f"Found checkpoint: {latest_ckpt_path}")
        print(f"Resuming from GLOBAL step: {start_iter}")
    else:
        print("No checkpoints found. Starting from 0.")

    # Calculate how many steps are actually left to run
    iters_remaining = iters_for_one_epoch - start_iter

    # Setup of th model config
    model_config = GPT.get_default_config()
    model_config.model_type = None
    model_config.n_layer = N_LAYER
    model_config.n_head  = N_HEAD
    model_config.n_embd  = N_EMBD
    model_config.vocab_size = 50257
    model_config.block_size = 128

    print(f"Model setup: L={N_LAYER}, H={N_HEAD}, E={N_EMBD}")
    model = GPT(model_config)

    # If resuming load weights
    if latest_ckpt_path:
        print(f"Loading weights into model")
        model.load_state_dict(torch.load(latest_ckpt_path, map_location='cpu'))

    # Training config
    train_config = Trainer.get_default_config() # Defualt config for trainer
    train_config.learning_rate = 0.0006
    train_config.max_iters = iters_for_one_epoch
    train_config.batch_size = BATCH_SIZE
    train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(
    train_dataset,
    shuffle=False,          # So we will read all the data even if we resume training
    pin_memory=True,        # Allocates special RAM that speeds up the transfer of data from CPU to GPU.
    batch_size=BATCH_SIZE,
    num_workers=0           # Tells Python to load the data in the main process (easier to debug) rather than using background threads.
)


    # # Initialize the optimizer (the part that updates the weigths in minGPT)
    optimizer = model.configure_optimizers(train_config)

    # Moves the model to the gpu
    model.to(train_config.device)

    # Puts the model in training mode
    model.train()

    print(f"Starting the training (Step {start_iter} to {iters_for_one_epoch})...")

    # The training loop
    # We loop through the 'train_loader' which gives us batches of data (x, y) one by one
    # enumerate() gives us a counter (batch_idx) so we know which step we are on.
    for batch_idx, (x, y) in enumerate(train_loader):

        # For when we resume training
        if batch_idx <= start_iter and start_iter > 0:
            continue

        # To stop exactly when we reach the end of 1 epoch.
        if batch_idx >= iters_for_one_epoch:
            break

        # Moves data to the gpu
        x = x.to(train_config.device)
        y = y.to(train_config.device)

        # Foward pass, we feed the input x to the model
        logits, loss = model(x, y)

        # Resert the gradient so only the current step will be taken into effect
        model.zero_grad(set_to_none=True)

        # Calculate the gradients (backpropagation)
        loss.backward()

        # Clips the size of the gradient to be 1 to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the weights.
        optimizer.step()

        if loss.item() < BEST_LOSS:
                BEST_LOSS = loss.item()

        # Print every 10 steps
        if batch_idx % 10 == 0:

            percent = (batch_idx / iters_for_one_epoch) * 100
            print(f"step {batch_idx}/{iters_for_one_epoch} ({percent:.1f}%): loss {loss.item():.4f}, best_loss {BEST_LOSS:.4f}")


        # Saves a checkpoint every 500 steps
        if batch_idx % 500 == 0:
            ckpt_path = os.path.join(CHECKPOINT_FOLDER_PATH, f'ckpt_step_{batch_idx}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Saves the final model
    final_path = os.path.join(CHECKPOINT_FOLDER_PATH, 'final_model_1_epoch.pt')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")