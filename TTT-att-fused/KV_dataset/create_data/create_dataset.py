from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import json
from config import config
import h5py
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

def text_chunks(dataset, min_words):
    buffer = []
    for sample in dataset:
        buffer.append(sample['text'])
        if sum(len(t.split()) for t in buffer) >= min_words:
            yield ' '.join(buffer)
            buffer = []

def create_kv_dataset(model, dataset, tokenizer, num_samples, max_length, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'kv_dataset.h5')

    with h5py.File(save_path, 'w') as f:
        f.attrs['model_name'] = config['model_name']
        f.attrs['dataset'] = config['dataset_name']
        f.attrs['max_length'] = max_length
        f.attrs['num_samples'] = num_samples
        f.attrs['num_layers'] = model.config.num_hidden_layers
        f.attrs['num_heads'] = model.config.num_attention_heads
        f.attrs['head_dim'] = model.config.hidden_size // model.config.num_attention_heads

        datasets_k = None
        datasets_v = None

        for i, text in enumerate(text_chunks(dataset, min_words=10000)):
            if i >= num_samples:
                break

            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=max_length,
                truncation=True
            )
            print(f"Processing sample {i}: {text[:50]}...")
            print(f"Token length: {inputs['input_ids'].shape[1]}")

            inputs = {k: v.long().to(model.device) if v.is_floating_point() else v.to(model.device)
                      for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs, use_cache=True)

            kv_cache = outputs.past_key_values

            # [num_layers, num_heads, seq_len, head_dim] -> [seq_len, num_layers, num_heads, head_dim]
            keys = torch.stack([layer[0].squeeze(0) for layer in kv_cache])
            values = torch.stack([layer[1].squeeze(0) for layer in kv_cache])
            keys = keys.permute(2, 0, 1, 3).cpu().numpy()
            values = values.permute(2, 0, 1, 3).cpu().numpy()
            # keys/values shape: [seq_len, num_layers, num_heads, head_dim]

            if datasets_k is None:
                seq_len, num_layers, num_heads, head_dim = keys.shape
                datasets_k = f.create_dataset(
                    'keys',
                    shape=(0, num_layers, num_heads, head_dim),
                    maxshape=(None, num_layers, num_heads, head_dim),
                    dtype=np.float16,
                    chunks=(1, num_layers, num_heads, head_dim)
                )
                datasets_v = f.create_dataset(
                    'values',
                    shape=(0, num_layers, num_heads, head_dim),
                    maxshape=(None, num_layers, num_heads, head_dim),
                    dtype=np.float16,
                    chunks=(1, num_layers, num_heads, head_dim)
                )

            datasets_k.resize(datasets_k.shape[0] + keys.shape[0], axis=0)
            datasets_v.resize(datasets_v.shape[0] + values.shape[0], axis=0)
            datasets_k[-keys.shape[0]:] = keys.astype(np.float16)
            datasets_v[-values.shape[0]:] = values.astype(np.float16)

            print(f"Saved sample {i}, dataset shape so far: {datasets_k.shape}")


if __name__ == "__main__":
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    print("Loading dataset...")
    dataset = load_dataset(config['dataset_name'], split='train', streaming=True)

    print("Creating KV dataset...")
    create_kv_dataset(model, dataset, tokenizer, config['num_samples'], config['max_length'], config['save_dir'])