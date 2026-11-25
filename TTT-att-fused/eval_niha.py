import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
TORCHDYNAMO_VERBOSE=1

import json
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Import your custom TTT wrapper
from model.model_wrapper import Qwen3TTTWrapper

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
CHECKPOINT_PATH = "./checkpoints/ttt_final.pt"
DATA_PATH = "/data/fineweb_edu/mini_fineweb_edu_lc/fineweb_edu_subset.jsonl"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test grid
CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
DEPTH_PERCENTS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Fact configuration
NEEDLE = " <crazy fact> Donald like to eat Chicken. </crazy fact> "
QUESTION = "\nWhat does Donald like to eat? Answer in one word:"
EXPECTED_ANSWER = "Chicken"

# ─── Data Preparation ─────────────────────────────────────────────────────────

def load_master_haystack(tokenizer, required_tokens):
    """Loads a continuous chunk of real text from FineWeb once to use as the base haystack."""
    print(f"Loading {required_tokens} tokens from FineWeb...")
    haystack_tokens = []
    
    with open(DATA_PATH, 'r') as f:
        for line in f:
            text = json.loads(line)['text']
            # Tokenize without special tokens so we don't inject random BOS/EOS tags
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            haystack_tokens.extend(tokens)
            
            if len(haystack_tokens) >= required_tokens:
                break
                
    return haystack_tokens[:required_tokens]

def build_niah_prompt(tokenizer, master_haystack, target_length, depth_percent):
    """Slices the master haystack and injects the needle at the exact required depth."""
    needle_tokens = tokenizer(NEEDLE, add_special_tokens=False).input_ids
    question_tokens = tokenizer(QUESTION, add_special_tokens=False).input_ids
    
    # Calculate how much background text we actually need for this length
    required_haystack_len = target_length - len(needle_tokens) - len(question_tokens)
    current_haystack = master_haystack[:required_haystack_len]
    
    # Insert needle at specified depth
    insert_idx = int(depth_percent * len(current_haystack))
    final_tokens = current_haystack[:insert_idx] + needle_tokens + current_haystack[insert_idx:] + question_tokens
    
    return torch.tensor([final_tokens]).to(DEVICE)

# ─── Inference Generation ─────────────────────────────────────────────────────

def greedy_generate(model, input_ids, max_new_tokens=5):
    """Concise greedy generation loop optimized for cache-less TTT forward passes."""
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    return generated

# ─── Main Evaluation Loop ─────────────────────────────────────────────────────

def run_niah():
    print("Loading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # ─── REPLICATE INITIALIZATION FROM MAIN.PY ───
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    d_head = getattr(config, "head_dim", config.hidden_size // n_heads)
    layer_types = getattr(config, "layer_types", ["full_attention"] * n_layers)
    N_PSEUDO = 256  # Must match N_PSEUDO from main.py

    print("Initializing properly sized dummy pseudo-tokens...")
    # Create zeros with the correct shape; the checkpoint will overwrite these with trained weights
    init_WK_all = [
        torch.zeros(n_heads, N_PSEUDO, d_head, dtype=torch.bfloat16, device=DEVICE) 
        if layer_types[i] == "full_attention" else torch.empty(0, dtype=torch.bfloat16, device=DEVICE)
        for i in range(n_layers)
    ]
    init_WV_all = [
        torch.zeros(n_heads, N_PSEUDO, d_head, dtype=torch.bfloat16, device=DEVICE) 
        if layer_types[i] == "full_attention" else torch.empty(0, dtype=torch.bfloat16, device=DEVICE)
        for i in range(n_layers)
    ]

    print("Initializing TTT model wrapper...")
    model = Qwen3TTTWrapper(
        model_name=MODEL_NAME, 
        init_WK_all=init_WK_all, 
        init_WV_all=init_WV_all, 
        chunk_size=512, 
        history_size=512, 
        lora_mlp=True, 
        lora_qkv=True, 
        lora_rank=64, 
        dependent_lr=True
    ).to(DEVICE).to(torch.bfloat16)
    
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    # strict=False is still good here in case some layers (like base Qwen) aren't in the checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True), strict=False)
    model.eval()

    # Pre-load the maximum amount of FineWeb text we will ever need
    max_required_tokens = max(CONTEXT_LENGTHS)
    master_haystack = load_master_haystack(tokenizer, max_required_tokens)

    results = np.zeros((len(DEPTH_PERCENTS), len(CONTEXT_LENGTHS)))
    total_iters = len(CONTEXT_LENGTHS) * len(DEPTH_PERCENTS)
    
    print("\nStarting Needle In A Haystack Evaluation...")
    with tqdm(total=total_iters) as pbar:
        for c_idx, length in enumerate(CONTEXT_LENGTHS):
            for d_idx, depth in enumerate(DEPTH_PERCENTS):
                
                # 1. Build prompt
                input_ids = build_niah_prompt(tokenizer, master_haystack, length, depth)
                
                # 2. Generate answer
                output_tokens = greedy_generate(model, input_ids, max_new_tokens=5)
                generated_text = tokenizer.decode(output_tokens, skip_special_tokens=True).lower()
                
                # 3. Grade it
                is_correct = EXPECTED_ANSWER.lower() in generated_text
                results[d_idx, c_idx] = 1 if is_correct else 0
                
                pbar.set_postfix({
                    "Len": length, "Depth": f"{depth*100}%", 
                    "Match": is_correct, "Out": repr(generated_text.strip())
                })
                pbar.update(1)

    # ─── Plotting the Heatmap ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        results, 
        xticklabels=CONTEXT_LENGTHS, 
        yticklabels=[f"{int(d*100)}%" for d in DEPTH_PERCENTS],
        cmap="RdYlGn", cbar=False, annot=True, fmt=".0f", linewidths=.5
    )
    
    plt.title("Needle In A Haystack (NIAH) - TTT Model Accuracy", pad=20, fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Depth of Needle in Document", fontsize=12)
    
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/niah_fineweb_heatmap.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"\nEvaluation complete! FineWeb-backed heatmap saved to {save_path}")

if __name__ == "__main__":
    run_niah()