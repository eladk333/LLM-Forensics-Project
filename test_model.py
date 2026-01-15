import os
import sys
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import GPT2Tokenizer
import threading

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'minGPT'))

try:
    from mingpt.model import GPT
except ImportError:
    print("❌ Error: Could not import 'mingpt'. Make sure the 'minGPT' folder is in the same directory.")
    sys.exit(1)

# --- Configuration ---
BASE_MODELS_FOLDER = r'G:\My Drive\llm\data\models'

MODEL_CONFIGS = {
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

class GPTPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT Model Player - Debug Mode")
        self.root.geometry("1000x700")
        
        # State
        self.model = None
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.current_model_name = None
        self.is_loading = False
        self.suppress_newlines = tk.BooleanVar(value=True)
        
        # Variables to hold current predictions
        self.current_top_token_id = None
        self.current_top_token_str = None
        
        # Layout
        self._setup_ui()
        
        # Load default
        self.model_selector.set('124M')
        self.load_model_thread('124M')

    def _setup_ui(self):
        # 1. Top Control Bar
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Model Selector
        ttk.Label(control_frame, text="Model Size:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_selector = ttk.Combobox(control_frame, values=list(MODEL_CONFIGS.keys()), state="readonly", width=8)
        self.model_selector.pack(side=tk.LEFT)
        self.model_selector.bind("<<ComboboxSelected>>", lambda e: self.load_model_thread(self.model_selector.get()))
        
        # Suppress \n
        ttk.Checkbutton(control_frame, text="Ban Newlines", variable=self.suppress_newlines, command=self.run_inference).pack(side=tk.LEFT, padx=15)
        
        # Ban Specific ID Input
        ttk.Label(control_frame, text="Ban ID:").pack(side=tk.LEFT, padx=(15, 5))
        self.ban_id_entry = ttk.Entry(control_frame, width=8)
        self.ban_id_entry.pack(side=tk.LEFT)
        self.ban_id_entry.bind("<Return>", lambda e: self.run_inference()) 
        ttk.Button(control_frame, text="Apply", command=self.run_inference, width=6).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Idle", foreground="gray")
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # 2. Main Content Area
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # --- Left Side: Input ---
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)
        
        ttk.Label(left_frame, text="Input Text:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0,5))
        
        self.text_area = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.bind("<KeyRelease>", self.on_text_change)
        
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_text).pack(side=tk.LEFT)
        
        # CHANGED BUTTON HERE
        self.add_btn = ttk.Button(btn_frame, text="Add Top Prediction", command=self.add_top_prediction)
        self.add_btn.pack(side=tk.RIGHT)

        # --- Right Side: Predictions ---
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        ttk.Label(right_frame, text="Top Predictions (Double-click to add):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0,5))
        
        cols = ('rank', 'id', 'token', 'prob')
        self.tree = ttk.Treeview(right_frame, columns=cols, show='headings', selectmode='browse')
        
        self.tree.heading('rank', text='#')
        self.tree.heading('id', text='ID')
        self.tree.heading('token', text='Token')
        self.tree.heading('prob', text='Prob')
        
        self.tree.column('rank', width=30, anchor='center')
        self.tree.column('id', width=50, anchor='center')
        self.tree.column('token', width=100, anchor='w')
        self.tree.column('prob', width=60, anchor='e')
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<Double-1>', self.on_item_double_click)

    def load_model_thread(self, size):
        if self.is_loading: return
        threading.Thread(target=self.load_model, args=(size,), daemon=True).start()

    def load_model(self, size):
        self.is_loading = True
        self.root.after(0, lambda: self.status_label.config(text=f"Loading {size}...", foreground="orange"))
        
        try:
            conf = MODEL_CONFIGS[size]
            model_config = GPT.get_default_config()
            model_config.model_type = None
            model_config.n_layer = conf['n_layer']
            model_config.n_head = conf['n_head']
            model_config.n_embd = conf['n_embd']
            model_config.vocab_size = 50257
            model_config.block_size = 128
            
            model = GPT(model_config)
            ckpt_path = os.path.join(BASE_MODELS_FOLDER, f'MinGPT_Checkpoints_{size}', 'final_model_1_epoch.pt')
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing: {ckpt_path}")

            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            model.eval()
            self.model = model
            self.root.after(0, lambda: self.status_label.config(text=f"Active: {size}", foreground="green"))
            self.root.after(0, self.run_inference)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}", foreground="red"))
            print(e)
        finally:
            self.is_loading = False

    def on_text_change(self, event=None):
        if event and event.keysym in ['Return', 'space', 'BackSpace']:
            self.run_inference()

    def clear_text(self):
        self.text_area.delete('1.0', tk.END)
        self.tree.delete(*self.tree.get_children())
        self.current_top_token_str = None

    def add_top_prediction(self):
        """NEW: Adds the currently calculated top token to the text area"""
        # Run inference first to ensure we have the latest top token
        self.run_inference()
        
        if self.current_top_token_str:
            self.text_area.insert(tk.END, self.current_top_token_str)
            self.text_area.see(tk.END)
            # Run inference AGAIN to predict the word after this one
            self.run_inference()

    def run_inference(self):
        if not self.model: return
        text = self.text_area.get('1.0', tk.END).rstrip('\n')
        if not text: return

        idx = self.tokenizer.encode(text)
        if len(idx) == 0: return

        x = torch.tensor([idx], dtype=torch.long)
        if x.size(1) > 128: x = x[:, -128:]

        with torch.no_grad():
            logits, _ = self.model(x)
            last_token_logits = logits[0, -1, :]

            # --- BANNING LOGIC ---
            if self.suppress_newlines.get():
                last_token_logits[198] = -float('inf') # \n
                last_token_logits[628] = -float('inf') # \n\n
            
            # Ban user-specific ID
            user_ban = self.ban_id_entry.get().strip()
            if user_ban.isdigit():
                ban_id = int(user_ban)
                if 0 <= ban_id < 50257:
                    last_token_logits[ban_id] = -float('inf')

            probs = F.softmax(last_token_logits, dim=0)
            top_probs, top_indices = torch.topk(probs, 30)

        # Update tracking variables for the button
        self.current_top_token_id = top_indices[0].item()
        self.current_top_token_str = self.tokenizer.decode([self.current_top_token_id])

        self.update_tree(top_probs, top_indices)

    def update_tree(self, probs, indices):
        self.tree.delete(*self.tree.get_children())
        for rank, (prob, idx) in enumerate(zip(probs, indices)):
            token_id = idx.item()
            token_str = self.tokenizer.decode([token_id])
            prob_pct = f"{prob.item() * 100:.2f}%"
            display_token = repr(token_str).strip("'")
            
            self.tree.insert('', 'end', values=(rank+1, token_id, display_token, prob_pct), tags=(str(token_id),))

    def on_item_double_click(self, event):
        # FIX: Check if selection exists before accessing [0]
        selection = self.tree.selection()
        if not selection:
            return 

        item_id = selection[0]
        token_id = int(self.tree.item(item_id, 'tags')[0])
        raw_token = self.tokenizer.decode([token_id])
        self.text_area.insert(tk.END, raw_token)
        self.text_area.see(tk.END)
        self.run_inference()

if __name__ == "__main__":
    root = tk.Tk()
    app = GPTPlayerApp(root)
    root.mainloop()