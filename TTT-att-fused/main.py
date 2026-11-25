import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import tqdm
import json
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from model.model_wrapper import Qwen3TTTWrapper
from generate import dumb_generate

# ── config ────────────────────────────────────────────────────────────────────
SEQ_LEN = 2048
REPEATED_CONTEXT = True
CHUNK_SIZE = 512
HISTORY_SIZE = 512
N_PSEUDO = 256
MAX_DOCS = 500     
TRAIN_SPLIT = 0.9       
EPOCHS = 1              
BATCH_SIZE = 4          
GRAD_ACCUM_STEPS = 1    
LEARNING_RATE = 1e-4
RANDOM_INIT_KV = True

MODEL_NAME = "Qwen/Qwen3-8B"
DATA_PATH = "/data/fineweb_edu/mini_fineweb_edu_lc/fineweb_edu_subset.jsonl"
PSEUDO_TOKENS_PATH = "./KV_dataset/trained_KV/pseudo_tokens.pt"
SAVE_DIR = "./checkpoints"
DTYPE = torch.bfloat16

# ── data ──────────────────────────────────────────────────────────────────────
def load_fineweb(path, tokenizer, seq_len, max_docs=100):
    chunks = []
    with open(path, 'r') as f:
        for line in f:
            if len(chunks) >= max_docs:
                break
            doc = json.loads(line)
            tokens = tokenizer(
                doc['text'],
                return_tensors='pt',
                truncation=False,
            )['input_ids'][0]
            for i in range(0, len(tokens) - seq_len, seq_len):
                chunks.append(tokens[i:i + seq_len])
                if len(chunks) >= max_docs:
                    break
    return chunks

# ── plotting ──────────────────────────────────────────────────────────────────
def smooth(x, window=100):
    return np.convolve(x, np.ones(window) / window, mode='valid')

def plot_val_ppl(val_loss_per_pos, chunk_size, n_pseudo, save_path, epoch):
    raw_ppl = val_loss_per_pos.exp().numpy()
    
    window = 100
    smoothed_ppl = smooth(raw_ppl, window=window)
    positions = np.arange(len(smoothed_ppl)) + window // 2 

    first_chunk_end = min(chunk_size, len(raw_ppl))
    baseline_ppl = np.mean(raw_ppl[50:first_chunk_end])

    delta_ppl = smoothed_ppl - baseline_ppl

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    ax1.plot(positions, smoothed_ppl, label=f'Fine-tuned TTT PPL', color='steelblue')
    ax1.axhline(y=baseline_ppl, color='gray', linestyle=':', label=f'Baseline (Avg First Chunk: {baseline_ppl:.2f})')

    ax1.axvline(x=chunk_size, color='orange', linestyle='--', label=f'First TTT update ({chunk_size})')
    ax1.axvline(x=n_pseudo, color='red', linestyle='--', label=f'Pseudo context size ({n_pseudo})')

    y_max = baseline_ppl * 1.3
    y_min = max(0, np.min(smoothed_ppl) * 0.9)
    ax1.set_ylim(y_min, y_max)

    ax1.set_ylabel('Perplexity (Lower is Better)')
    ax1.set_title(f'Validation Per-Position PPL — Epoch {epoch} (Zoomed Y-Axis)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(positions, delta_ppl, label='Delta from First Chunk', color='purple')
    ax2.axhline(y=0, color='black', linestyle='-')
    
    ax2.fill_between(positions, 0, delta_ppl, where=(delta_ppl <= 0), facecolor='green', alpha=0.3, label='Improvement (Memory Helping)')
    ax2.fill_between(positions, 0, delta_ppl, where=(delta_ppl > 0), facecolor='red', alpha=0.3, label='Degradation (Memory Hurting)')

    ax2.axvline(x=chunk_size, color='orange', linestyle='--')
    ax2.axvline(x=n_pseudo, color='red', linestyle='--')

    ax2.set_xlabel('Position in sequence')
    ax2.set_ylabel('Δ Perplexity')
    ax2.set_title('Delta PPL (Change relative to pre-update baseline)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() 
    print(f"Plot saved to {save_path}")

# ── validation function ───────────────────────────────────────────────────────
def evaluate_per_position(model, val_chunks, device):
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        pbar = tqdm.tqdm(val_chunks, desc="Validating")
        for tokens in pbar:
            input_ids = tokens.unsqueeze(0).to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            shift_logits = logits[0, :-1, :].float()
            shift_labels = input_ids[0, 1:]
            
            per_token_loss = F.cross_entropy(
                shift_logits, 
                shift_labels, 
                reduction='none'
            )
            all_losses.append(per_token_loss.cpu())
            
            seq_loss = per_token_loss.mean().item()
            pbar.set_postfix({'ppl': f"{np.exp(seq_loss):.2f}"})
            
    all_losses = torch.stack(all_losses, dim=0) 
    mean_per_position_loss = all_losses.mean(dim=0)
    
    overall_loss = mean_per_position_loss.mean().item()
    overall_ppl = np.exp(overall_loss)
    
    return mean_per_position_loss, overall_loss, overall_ppl

# ── main training loop ────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    print("Loading data...")
    all_chunks = load_fineweb(DATA_PATH, tokenizer, seq_len=SEQ_LEN, max_docs=MAX_DOCS)
    
    split_idx = int(len(all_chunks) * TRAIN_SPLIT)
    train_chunks = all_chunks[:split_idx]
    val_chunks = all_chunks[split_idx:]
    print(f"Loaded {len(all_chunks)} total chunks.")
    print(f"  -> Training on {len(train_chunks)} chunks")
    print(f"  -> Validating on {len(val_chunks)} chunks")

    print("\nLoading pseudo tokens...")
    try:
        pseudo_tokens = torch.load(PSEUDO_TOKENS_PATH, weights_only=False)
        init_WK_all = pseudo_tokens['pseudo_context_K'].to(dtype=DTYPE, device=device) * 0.01 
        init_WV_all = pseudo_tokens['pseudo_context_V'].to(dtype=DTYPE, device=device) * 0.01 
    except FileNotFoundError:
        print("Pseudo tokens file not found. Generating random ones.")
        RANDOM_INIT_KV = True

    if RANDOM_INIT_KV:
        print("Using random initialization for pseudo tokens.")
        
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        d_head = getattr(config, "head_dim", config.hidden_size // n_heads)
        layer_types = getattr(config, "layer_types", ["full_attention"] * n_layers)
        
        init_WK_all = [
            torch.randn(n_heads, N_PSEUDO, d_head, dtype=DTYPE, device=device) * 0.01 if layer_types[i] == "full_attention" else torch.empty(0, dtype=DTYPE, device=device)
            for i in range(n_layers)
        ]
        init_WV_all = [
            torch.randn(n_heads, N_PSEUDO, d_head, dtype=DTYPE, device=device) * 0.01 if layer_types[i] == "full_attention" else torch.empty(0, dtype=DTYPE, device=device)
            for i in range(n_layers)
        ]

    print("Loading TTT wrapped model...")
    ttt_model = Qwen3TTTWrapper(
        model_name=MODEL_NAME,
        init_WK_all=init_WK_all,
        init_WV_all=init_WV_all,
        chunk_size=CHUNK_SIZE,
        history_size=HISTORY_SIZE,
        lora_mlp=False,
        lora_qkv=True,
        lora_rank=64,
        dependent_lr=False, 
        lr_base=5e-2,
        pseudo_token_mask=0.0,
    )
    
    ttt_model = ttt_model.to(dtype=DTYPE, device=device)
    ttt_model.model.enable_input_require_grads()
    ttt_model.model.gradient_checkpointing_enable()

    trainable_params = [p for p in ttt_model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"\nTotal trainable parameters: {total_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    print("\nStarting Training...")
    global_step = 0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*40}\n--- Epoch {epoch+1}/{EPOCHS} ---\n{'='*40}")
        
        ttt_model.train()
        optimizer.zero_grad()
        
        indices = torch.randperm(len(train_chunks))
        pbar = tqdm.tqdm(indices, desc=f"Training")
        
        for step, idx in enumerate(pbar):
            input_ids = train_chunks[idx].unsqueeze(0).to(device)
            if REPEATED_CONTEXT:
                input_ids = torch.cat([input_ids, input_ids], dim=1)

            outputs = ttt_model(input_ids)
            logits = outputs.logits

            shift_logits = logits[0, :-1, :].float().contiguous().view(-1, config.vocab_size)
            shift_labels = input_ids[0, 1:].contiguous().view(-1)
            
            loss = F.cross_entropy(shift_logits, shift_labels)
            
            loss_scaled = loss / GRAD_ACCUM_STEPS
            loss_scaled.backward()

            ppl = torch.exp(loss.detach()).item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ppl': f"{ppl:.2f}"})

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    ckpt_path = os.path.join(SAVE_DIR, f"ttt_step_{global_step}.pt")
                    state_dict = {name: param.detach().cpu() for name, param in ttt_model.named_parameters() if param.requires_grad}
                    torch.save(state_dict, ckpt_path)

        print(f"\n--- Running Validation for Epoch {epoch+1} ---")
        val_loss_per_pos, val_loss_overall, val_ppl_overall = evaluate_per_position(ttt_model, val_chunks, device)
        print(f"\n>>> Epoch {epoch+1} Results: Val Loss = {val_loss_overall:.4f} | Val PPL = {val_ppl_overall:.2f}")

        print("\n>>> Per-Chunk PPL Delta Summary:")
        raw_ppl = val_loss_per_pos.exp().numpy()
        baseline_avg = np.mean(raw_ppl[50:CHUNK_SIZE])
        
        for i in range(0, SEQ_LEN, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, SEQ_LEN)
            start = i if i > 0 else 50 
            chunk_avg = np.mean(raw_ppl[start:end])
            delta = chunk_avg - baseline_avg
            indicator = "📉 IMPROVED" if delta < 0 else "📈 DEGRADED"
            if i == 0: indicator = "--- (Baseline)"
            print(f"Chunk {i//CHUNK_SIZE + 1:02d} ({i:04d}-{end:04d}): PPL = {chunk_avg:6.2f} | Delta = {delta:+6.2f} | {indicator}")
        
        plot_path = os.path.join(SAVE_DIR, f"val_ppl_epoch_{epoch+1}.png")
        plot_val_ppl(val_loss_per_pos, CHUNK_SIZE, N_PSEUDO, plot_path, epoch+1)
    
    # generate outputs with the final model
    print("\nGenerating sample outputs with the final TTT model...")
    sample_prompts = [
        "Once upon a time in a land far away,",
        "The future of artificial intelligence is",
        "In the midst of the bustling city,",
    ]
    for prompt in sample_prompts:
        generated_text = dumb_generate(ttt_model, tokenizer, prompt, max_length=200)
        print(f"\nPrompt: {prompt}\nGenerated: {generated_text}")

    print("\nTraining complete. Saving final weights...")
    final_path = os.path.join(SAVE_DIR, "ttt_final.pt")
    final_state_dict = {name: param.detach().cpu() for name, param in ttt_model.named_parameters() if param.requires_grad}
    torch.save(final_state_dict, final_path)
    print(f"Saved to {final_path}")

    print("\n" + "="*40)
    print("--- Running Base Model Baseline (No TTT Wrapper) ---")
    print("="*40)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model = base_model.to(dtype=DTYPE, device=device)
    base_model.eval()

    base_all_losses = []

    with torch.no_grad():
        pbar = tqdm.tqdm(val_chunks, desc="Base model validation")
        for tokens in pbar:
            input_ids = tokens.unsqueeze(0).to(device)

            outputs = base_model(input_ids, use_cache=False)
            logits = outputs.logits

            shift_logits = logits[0, :-1, :].float()
            shift_labels = input_ids[0, 1:]

            per_token_loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
            base_all_losses.append(per_token_loss.cpu())

            seq_loss = per_token_loss.mean().item()
            pbar.set_postfix({'ppl': f"{np.exp(seq_loss):.2f}"})

    del base_model
    torch.cuda.empty_cache()

    base_all_losses = torch.stack(base_all_losses, dim=0)
    base_mean_per_position_loss = base_all_losses.mean(dim=0)
    base_overall_loss = base_mean_per_position_loss.mean().item()
    base_overall_ppl = np.exp(base_overall_loss)

    print(f"\n>>> Base Model: Val Loss = {base_overall_loss:.4f} | Val PPL = {base_overall_ppl:.2f}")
    print(f">>> TTT Model:  Val Loss = {val_loss_overall:.4f} | Val PPL = {val_ppl_overall:.2f}")
    print(f">>> Delta:      {val_ppl_overall - base_overall_ppl:+.2f} PPL  ({'worse' if val_ppl_overall > base_overall_ppl else 'better'} than base)")

    raw_ppl_ttt = val_loss_per_pos.exp().numpy()
    raw_ppl_base = base_mean_per_position_loss.exp().numpy()
    window = 100

    smoothed_ttt = smooth(raw_ppl_ttt, window=window)
    smoothed_base = smooth(raw_ppl_base, window=window)
    positions = np.arange(len(smoothed_ttt)) + window // 2

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(positions, smoothed_ttt, label='TTT Model', color='steelblue')
    ax.plot(positions, smoothed_base, label='Base Model (No TTT)', color='darkorange', linestyle='--')
    ax.axvline(x=CHUNK_SIZE, color='green', linestyle=':', label=f'First TTT update ({CHUNK_SIZE})')

    ax.set_xlabel('Position in sequence')
    ax.set_ylabel('Perplexity')
    ax.set_title('TTT vs Base Model — Per-Position Perplexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    compare_path = os.path.join(SAVE_DIR, f"ttt_vs_base_epoch_{EPOCHS}.png")
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {compare_path}")

    print("\n" + "="*40)
    print("Final Hyperparameters:")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Sequence Length: {SEQ_LEN}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"History Size: {HISTORY_SIZE}")
    print(f"Number of Pseudo Tokens: {N_PSEUDO}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}") 
    print(f"Gradient Accumulation Steps: {GRAD_ACCUM_STEPS}")
    print(f"Random Init KV: {RANDOM_INIT_KV}")
    print(f"LoRA MLP: {ttt_model.lora_mlp}")
    print(f"LoRA QKV: {ttt_model.lora_qkv}")
    print(f"LoRA Rank: {ttt_model.lora_rank}")
    print(f"Dependent LR: {ttt_model.dependent_lr}")
    print(f"LR Base: {ttt_model.lr_base}")
    print(f"Pseudo Token Mask: {ttt_model.pseudo_token_mask}")
    print("="*40)