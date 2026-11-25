"""
debug.py — Data & tokenizer sanity checks for TTT training pipeline.
Run this before training to diagnose PPL issues.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from collections import Counter

# ── config — mirror your train.py settings ────────────────────────────────────
MODEL_NAMES = [
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]
DATA_PATH   = "/data/fineweb_edu/mini_fineweb_edu_lc/fineweb_edu_subset.jsonl"
SEQ_LEN     = 2048
MAX_DOCS    = 10     # small number just for diagnosis
N_DOCS_RAW  = 5     # how many raw docs to inspect

# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 60


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── 1. RAW FILE INSPECTION ────────────────────────────────────────────────────
section("1. RAW FILE INSPECTION")

raw_docs = []
with open(DATA_PATH, "r") as f:
    for i, line in enumerate(f):
        if i >= N_DOCS_RAW:
            break
        doc = json.loads(line)
        raw_docs.append(doc)
        text = doc.get("text", "")
        print(f"\nDoc {i}:")
        print(f"  keys        : {list(doc.keys())}")
        print(f"  text length : {len(text)} chars")
        print(f"  preview     : {text[:120]!r}")
        if not text:
            print("  ⚠️  EMPTY TEXT FIELD")

# check if 'text' key even exists
if raw_docs and "text" not in raw_docs[0]:
    print("\n❌  'text' key missing from docs! Check the field name above and fix DATA_PATH loader.")
    sys.exit(1)


# ── 2. TOKENIZER CHECKS ───────────────────────────────────────────────────────
section("2. TOKENIZER CHECKS (pythia-2.8b)")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")
print(f"Vocab size          : {tokenizer.vocab_size}")
print(f"UNK token id        : {tokenizer.unk_token_id}")
print(f"BOS token id        : {tokenizer.bos_token_id}")
print(f"EOS token id        : {tokenizer.eos_token_id}")
print(f"Model max length    : {tokenizer.model_max_length}")

for i, doc in enumerate(raw_docs[:3]):
    tokens = tokenizer(doc["text"], return_tensors="pt", truncation=False)["input_ids"][0]
    n_unk = (tokens == tokenizer.unk_token_id).sum().item() if tokenizer.unk_token_id is not None else 0
    print(f"\nDoc {i} tokenization:")
    print(f"  token count  : {len(tokens)}")
    print(f"  token range  : [{tokens.min().item()}, {tokens.max().item()}]")
    print(f"  UNK count    : {n_unk}  {'⚠️  HIGH' if n_unk > len(tokens) * 0.01 else '✓'}")
    print(f"  decoded[0:50]: {tokenizer.decode(tokens[:50])!r}")
    print(f"  decoded rt   : {tokenizer.decode(tokens[:50]) == doc['text'][:len(tokenizer.decode(tokens[:50]))]}")


# ── 3. CHUNK LOADER REPLICATION ───────────────────────────────────────────────
section("3. CHUNK LOADER")

def load_fineweb(path, tok, seq_len, max_docs=100):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            if len(chunks) >= max_docs:
                break
            doc = json.loads(line)
            tokens = tok(doc["text"], return_tensors="pt", truncation=False)["input_ids"][0]
            for i in range(0, len(tokens) - seq_len, seq_len):
                chunks.append(tokens[i : i + seq_len])
                if len(chunks) >= max_docs:
                    break
    return chunks

chunks = load_fineweb(DATA_PATH, tokenizer, SEQ_LEN, max_docs=MAX_DOCS)
print(f"Total chunks loaded : {len(chunks)}")
if chunks:
    print(f"Chunk shape         : {chunks[0].shape}")
    all_tokens = torch.cat(chunks)
    print(f"Global token range  : [{all_tokens.min().item()}, {all_tokens.max().item()}]")
    print(f"Vocab coverage      : {all_tokens.unique().numel()} / {tokenizer.vocab_size} tokens seen")

    # token frequency — check for pathological distributions
    counts = Counter(all_tokens.tolist())
    top10 = counts.most_common(10)
    print(f"Top-10 tokens       : {[(tokenizer.decode([t]), c) for t, c in top10]}")
else:
    print("❌  No chunks produced — check SEQ_LEN vs document lengths.")
    sys.exit(1)


# ── 4. QUICK PPL COMPARISON ACROSS MODELS ─────────────────────────────────────
section("4. QUICK PPL — BASE MODELS (first 5 chunks, no wrapper)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

eval_chunks = chunks[:5]

for model_name in MODEL_NAMES:
    print(f"\n  Loading {model_name} ...")
    model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for tokens in eval_chunks:
            input_ids = tokens.unsqueeze(0).to(device)
            out = model(input_ids, use_cache=False)
            shift_logits = out.logits[0, :-1].float()
            shift_labels = input_ids[0, 1:]
            loss = F.cross_entropy(shift_logits, shift_labels).item()
            losses.append(loss)

    mean_loss = np.mean(losses)
    print(f"  {model_name}")
    print(f"    per-chunk losses : {[f'{l:.3f}' for l in losses]}")
    print(f"    mean loss        : {mean_loss:.4f}")
    print(f"    mean PPL         : {np.exp(mean_loss):.2f}")

    del model
    torch.cuda.empty_cache()

    if np.exp(mean_loss) > 20:
        print("  ⚠️  PPL > 20 on a base model is abnormal — likely a data/tokenizer issue.")
    elif np.exp(mean_loss) < 10:
        print("  ✓  PPL looks reasonable for this model size.")


# ── 5. ENCODING ROUND-TRIP TEST ───────────────────────────────────────────────
section("5. ENCODE → DECODE ROUND-TRIP")

for i, doc in enumerate(raw_docs[:2]):
    original = doc["text"][:500]
    tokens = tokenizer(original, return_tensors="pt", truncation=False)["input_ids"][0]
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    match = original.strip() == decoded.strip()
    print(f"\nDoc {i}: round-trip {'✓ OK' if match else '⚠️  MISMATCH'}")
    if not match:
        print(f"  original : {original[:100]!r}")
        print(f"  decoded  : {decoded[:100]!r}")


print(f"\n{SEP}")
print("  Debug complete.")
print(SEP)