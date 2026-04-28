import os
import sys
import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import GPT2Tokenizer
import threading
# --- Prompts for Testing ---
prompt_capitals = """Beijing is the capital of China.
Ottawa is the capital of Canada.
Cairo is the capital of Egypt.
Tokyo is the capital of Japan.
Brasilia is the capital of Brazil.
Madrid is the capital of"""

prompt_currency = """The currency of China is the Yuan.
The currency of India is the Rupee.
The currency of the USA is the Dollar.
The currency of Japan is the Yen.
The currency of the UK is the Pound.
The currency of France is the"""

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'minGPT'))

try:
    from mingpt.model import GPT
    import mingpt.model # Required for the patch
except ImportError:
    print("❌ Error: Could not import 'mingpt'. Make sure the 'minGPT' folder is in the same directory.")
    sys.exit(1)

# --- ARCHITECTURE PATCH (Required to match train.py) ---
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

# Apply the patch so GPT initializes with the correct layer names
mingpt.model.Block = MultiGPUBlock


# --- Configuration ---
BASE_MODELS_FOLDER = os.path.join(current_dir, 'data', 'models', 'MinGPT_Checkpoints_500M')

MODEL_CONFIGS = {
    '500M': {'n_layer': 24, 'n_head': 16, 'n_embd': 1280}, 
    '124M': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
    '30M':  {'n_layer': 6,  'n_head': 6,  'n_embd': 384},
    '7M':   {'n_layer': 4,  'n_head': 4,  'n_embd': 128},
}

class GPTPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GPT Model Player - Smart Filter Edition")
        self.root.geometry("1100x800")
        
        # --- State ---
        self.model = None
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.is_loading = False
        self._after_id = None
        
        # UI Variables
        self.suppress_newlines = tk.BooleanVar(value=True)
        self.smart_punctuation = tk.BooleanVar(value=True) # NEW: Auto-fix punctuation
        self.temperature_var = tk.DoubleVar(value=0.8) 
        self.top_k_var = tk.IntVar(value=40) 
        
        self.banned_ids = set() # Fixed: Must be a set, not a dict
        
        # To track the last generated token for quick banning
        self.last_gen_id = None
        self.last_gen_token = ""

        # --- Layout ---
        self._setup_ui()
        
        # Load default
        self.text_area.insert("1.0", "The history of the")
        self.model_selector.set('500M')          # Changed from '124M'
        self.load_model_thread('500M')

    def _setup_ui(self):
        # 1. Top Control Bar
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Model Selector
        ttk.Label(control_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_selector = ttk.Combobox(control_frame, values=list(MODEL_CONFIGS.keys()), state="readonly", width=6)
        self.model_selector.pack(side=tk.LEFT)
        self.model_selector.bind("<<ComboboxSelected>>", lambda e: self.load_model_thread(self.model_selector.get()))
        
        # Temp & Top-K
        ttk.Label(control_frame, text="Temp:").pack(side=tk.LEFT, padx=(15, 5))
        self.temp_spin = ttk.Spinbox(control_frame, from_=0.0, to=3.0, increment=0.1, textvariable=self.temperature_var, width=5, command=self.trigger_inference)
        self.temp_spin.pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Top-K:").pack(side=tk.LEFT, padx=(15, 5))
        self.topk_spin = ttk.Spinbox(control_frame, from_=1, to=100, increment=5, textvariable=self.top_k_var, width=5, command=self.trigger_inference)
        self.topk_spin.pack(side=tk.LEFT)
        
        # Toggles
        ttk.Checkbutton(control_frame, text="Ban Newlines", variable=self.suppress_newlines, command=self.trigger_inference).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(control_frame, text="Smart Punctuation", variable=self.smart_punctuation, command=self.trigger_inference).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Idle", foreground="gray")
        self.status_label.pack(side=tk.RIGHT, padx=20)

        # 2. Main Content Area
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left Side: Input
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)
        
        ttk.Label(left_frame, text="Input Text:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0,5))
        self.text_area = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 12))
        self.text_area.pack(fill=tk.BOTH, expand=True)
        self.text_area.bind("<KeyRelease>", self.on_text_change)
        
        # Buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_text).pack(side=tk.LEFT)
        self.add_btn = ttk.Button(btn_frame, text="Generate Next Token", command=self.add_sampled_prediction)
        self.add_btn.pack(side=tk.RIGHT)
        self.root.bind("<Tab>", lambda e: self.add_sampled_prediction()) 

        # Right Side: Predictions
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        ttk.Label(right_frame, text="Next Token Probabilities:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0,5))
        cols = ('rank', 'id', 'token', 'prob')
        self.tree = ttk.Treeview(right_frame, columns=cols, show='headings', selectmode='browse')
        for col in cols: self.tree.heading(col, text=col.capitalize())
        self.tree.column('rank', width=30, anchor='center'); self.tree.column('id', width=50, anchor='center')
        self.tree.column('token', width=100, anchor='w'); self.tree.column('prob', width=60, anchor='e')
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<Double-1>', self.on_item_double_click)

        # 3. Bottom Quick Ban Panel (NEW)
        ban_frame = ttk.LabelFrame(self.root, text="Quick Ban Tools", padding=10)
        ban_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.last_token_label = ttk.Label(ban_frame, text="Last Generated: None", font=("Consolas", 10))
        self.last_token_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(ban_frame, text="BAN Last Token ID", command=self.ban_last_token).pack(side=tk.LEFT, padx=10)
        
        ttk.Label(ban_frame, text="Manual Ban ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.ban_id_entry = ttk.Entry(ban_frame, width=6)
        self.ban_id_entry.pack(side=tk.LEFT)
        self.ban_id_entry.bind("<Return>", lambda e: self.trigger_inference())
        ttk.Button(ban_frame, text="Add", command=self.add_manual_ban, width=6).pack(side=tk.LEFT, padx=5)
        
        self.banned_list_label = ttk.Label(ban_frame, text=f"Banned Count: {len(self.banned_ids)}", foreground="red")
        self.banned_list_label.pack(side=tk.RIGHT, padx=10)

    # --- Logic ---

    def ban_last_token(self):
        if self.last_gen_id is not None:
            self.banned_ids.add(self.last_gen_id)
            print(f"Banned ID: {self.last_gen_id}")
            self.update_ban_label()
            self.trigger_inference()

    def add_manual_ban(self):
        txt = self.ban_id_entry.get().strip()
        if txt.isdigit():
            self.banned_ids.add(int(txt))
            self.ban_id_entry.delete(0, tk.END)
            self.update_ban_label()
            self.trigger_inference()

    def update_ban_label(self):
        self.banned_list_label.config(text=f"Banned IDs: {len(self.banned_ids)}")

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
            model_config.n_layer = conf['n_layer']; model_config.n_head = conf['n_head']; model_config.n_embd = conf['n_embd']
            # 1. Dynamic block size based on the model
            model_config.vocab_size = 50257
            model_config.block_size = 1024 if size == '500M' else 128 
            
            model = GPT(model_config)
            
            # 2. Custom path handling for the 500M model's naming convention
            
            ckpt_candidates = [
               # os.path.join(BASE_MODELS_FOLDER, 'ckpt_epoch_1_step_100000.pt'),
                os.path.join(BASE_MODELS_FOLDER, 'checkpoint_epoch_1.pt'),
            ]

            checkpoint = None
            last_error = None
            for ckpt_path in ckpt_candidates:
                if not os.path.exists(ckpt_path):
                    continue
                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                    break
                except Exception as e:
                    last_error = e

            if checkpoint is None:
                raise FileNotFoundError(f"Unable to load any checkpoint. Last error: {last_error}")
            
            # Extract weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Safely strip 'module.' prefix if the model was saved via DataParallel
            unwrapped_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key[7:] if key.startswith('module.') else key
                unwrapped_state_dict[clean_key] = value
                
            model.load_state_dict(unwrapped_state_dict)
            model.eval()
            self.model = model
            self.run_terminal_tests()
            self.root.after(0, lambda: self.status_label.config(text=f"Active: {size}", foreground="green"))
            self.root.after(0, self.trigger_inference)
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {e}", foreground="red"))
            print(f"Error: {e}")
        finally:
            self.is_loading = False

    def on_text_change(self, event=None):
        if event and event.keysym in ['Return', 'space', 'BackSpace']: self.trigger_inference(); return
        if self._after_id: self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(300, self.trigger_inference)

    def trigger_inference(self): self.run_inference()

    def clear_text(self):
        self.text_area.delete('1.0', tk.END)
        self.tree.delete(*self.tree.get_children())

    def add_sampled_prediction(self):
       # self.run_inference()
        if self.current_chosen_str:
            self.text_area.insert(tk.END, self.current_chosen_str)
            self.text_area.see(tk.END)
            
            # UPDATE LAST TOKEN INFO
            self.last_gen_id = self.current_chosen_id
            self.last_gen_token = self.current_chosen_str
            safe_display = repr(self.last_gen_token).strip("'")
            self.last_token_label.config(text=f"Last Generated: ID {self.last_gen_id} | '{safe_display}'")
            
            self.root.after(10, self.trigger_inference)

    def run_inference(self):
        if not self.model: return
        text = self.text_area.get('1.0', tk.END).rstrip('\n') 
        if not text: return
        try: idx = self.tokenizer.encode(text)
        except: return 
        if len(idx) == 0: return

        x = torch.tensor([idx], dtype=torch.long)
        
        # Determine max context based on current model
        max_context = 1024 if self.model_selector.get() == '500M' else 128
        
        if x.size(1) > max_context: x = x[:, -max_context:]

        with torch.no_grad():
            logits, _ = self.model(x)
            last_token_logits = logits[0, -1, :]

            # --- 1. BANNING LOGIC ---
            for bid in self.banned_ids:
                last_token_logits[bid] = -float('inf')

            if self.suppress_newlines.get():
                last_token_logits[198] = -float('inf')
                last_token_logits[628] = -float('inf')

            # --- 2. SMART PUNCTUATION (NEW) ---
            # If the text ends with a quote, BAN the period
            if self.smart_punctuation.get():
                # Check the last character of the input text
                stripped_text = text.rstrip()
                if stripped_text.endswith('"') or stripped_text.endswith(' "'):
                    # Ban '.' (13) and ' .' (764)
                    last_token_logits[13] = -float('inf')
                    last_token_logits[764] = -float('inf')

            # --- 3. REPETITION PENALTY ---
            repetition_penalty = 1.2 
            for token_id in set(x[0].tolist()):
                if last_token_logits[token_id] > 0: last_token_logits[token_id] /= repetition_penalty
                else: last_token_logits[token_id] *= repetition_penalty

            # --- 4. SAMPLING ---
            try: temp = float(self.temperature_var.get()); top_k = int(self.top_k_var.get())
            except: temp = 0.8; top_k = 40

            is_greedy = temp < 0.05
            if is_greedy:
                scaled_logits = last_token_logits
            else:
                scaled_logits = last_token_logits / temp
                v, _ = torch.topk(scaled_logits, top_k)
                scaled_logits[scaled_logits < v[[-1]]] = -float('Inf')

            probs = F.softmax(scaled_logits, dim=0)
            
            if is_greedy:
                _, top_idx = torch.max(probs, dim=0)
                self.current_chosen_id = top_idx.item()
            else:
                sample_idx = torch.multinomial(probs, num_samples=1)
                self.current_chosen_id = sample_idx.item()

            self.current_chosen_str = self.tokenizer.decode([self.current_chosen_id])
            top_probs, top_indices = torch.topk(probs, 30)

        self.update_tree(top_probs, top_indices)

    def update_tree(self, probs, indices):
        self.tree.delete(*self.tree.get_children())
        for rank, (prob, idx) in enumerate(zip(probs, indices)):
            token_id = idx.item()
            token_str = self.tokenizer.decode([token_id])
            prob_pct = f"{prob.item() * 100:.2f}%"
            display_token = repr(token_str).strip("'")
            tags = ['chosen'] if token_id == self.current_chosen_id else []
            self.tree.insert('', 'end', values=(rank+1, token_id, display_token, prob_pct), tags=tags)
        self.tree.tag_configure('chosen', background='#e1f5fe')

    def on_item_double_click(self, event):
        selection = self.tree.selection()
        if not selection: return
        vals = self.tree.item(selection[0], 'values')
        self.text_area.insert(tk.END, self.tokenizer.decode([int(vals[1])]))
        self.text_area.see(tk.END)
        self.trigger_inference()

    def run_terminal_tests(self):
        if not self.model: return
        
        print("\n" + "="*50)
        print(f"🧠 Running Factual Recall Tests (Top-5 & 5-Shot) for {self.model_selector.get()}...")
        print("="*50)
        
        # Upgraded to strict 5-shot prompts based on standard evaluation practices
        test_suite = [
            {
                "name": "Capital - Spain",
                "prompt": "Beijing is the capital of China.\nOttawa is the capital of Canada.\nCairo is the capital of Egypt.\nTokyo is the capital of Japan.\nBrasilia is the capital of Brazil.\nMadrid is the capital of",
                "expected": [" Spain", "Spain"]
            },
            {
                "name": "Currency - France",
                "prompt": "The currency of China is the Yuan.\nThe currency of India is the Rupee.\nThe currency of the USA is the Dollar.\nThe currency of Japan is the Yen.\nThe currency of the UK is the Pound.\nThe currency of France is the",
                "expected": [" Euro", "Euro", " Franc"]
            },
            {
                "name": "Company HQ - Google",
                "prompt": "Microsoft is headquartered in Redmond.\nApple is headquartered in Cupertino.\nAmazon is headquartered in Seattle.\nMeta is headquartered in Menlo Park.\nNetflix is headquartered in Los Gatos.\nGoogle is headquartered in",
                "expected": [" Mountain", "Mountain View", " California", " Mountain View,"]
            },
            {
                "name": "Person Instrument - Miles Davis",
                "prompt": "Jimi Hendrix plays the guitar.\nElton John plays the piano.\nLouis Armstrong plays the trumpet.\nYo-Yo Ma plays the cello.\nStevie Wonder plays the keyboard.\nMiles Davis plays the",
                "expected": [" trumpet", "trumpet", " Trumpet", " horn"]
            },
            {
                "name": "Country Language - Brazil",
                "prompt": "People in France speak French.\nPeople in Germany speak German.\nPeople in Japan speak Japanese.\nPeople in Italy speak Italian.\nPeople in Spain speak Spanish.\nPeople in Brazil speak",
                "expected": [" Portuguese", "Portuguese"]
            },
            {
                "name": "Company CEO - Apple",
                "prompt": "Satya Nadella is the CEO of Microsoft.\nMark Zuckerberg is the CEO of Meta.\nSundar Pichai is the CEO of Google.\nAndy Jassy is the CEO of Amazon.\nElon Musk is the CEO of Tesla.\nTim Cook is the CEO of",
                "expected": [" Apple", "Apple", " Apple Inc"]
            }
            
        ]
        
        self.model.eval()
        correct_count = 0
        total_tests = len(test_suite)
        TOP_K_CHECK = 10 # How deep to look for the correct answer

        with torch.no_grad():
            for test in test_suite:
                try:
                    name = test["name"]
                    text = test["prompt"]
                    expected_answers = test["expected"]
                    
                    idx = self.tokenizer.encode(text)
                    x = torch.tensor([idx], dtype=torch.long)
                    
                    max_context = 1024 if self.model_selector.get() == '500M' else 128
                    if x.size(1) > max_context: x = x[:, -max_context:]
                    
                    logits, _ = self.model(x)
                    next_token_logits = logits[0, -1, :]
                    
                    # --- TOP K LOGIC ---
                    top_logits, top_indices = torch.topk(next_token_logits, TOP_K_CHECK)
                    top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                    
                    last_line = text.split('\n')[-1]
                    
                    # Evaluate correctness across the top K tokens
                    is_correct = False
                    matched_token = ""
                    for expected in expected_answers:
                        for token in top_tokens:
                            if expected.lower() in token.lower():
                                is_correct = True
                                matched_token = token
                                break
                        if is_correct:
                            break
                    
                    if is_correct:
                        correct_count += 1
                        status = "✅ PASS"
                        print(f"[{name}] {status}")
                        print(f"Prompt:   '{last_line}'")
                        print(f"Match:    '{matched_token}' (found in Top-{TOP_K_CHECK})")
                    else:
                        status = "❌ FAIL"
                        print(f"[{name}] {status}")
                        print(f"Prompt:   '{last_line}'")
                        print(f"Top-{TOP_K_CHECK}:  {top_tokens}")
                        print(f"Expected: {expected_answers}")
                    print("-" * 50)
                except Exception as e:
                    print(f"Failed to run test {name}: {e}")

        accuracy = (correct_count / total_tests) * 100 if total_tests > 0 else 0.0
        print(f"\n📊 FINAL SCORE (Top-{TOP_K_CHECK} Accuracy): {correct_count}/{total_tests} ({accuracy:.1f}%)")
        print("="*50 + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = GPTPlayerApp(root)
    root.mainloop()