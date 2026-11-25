import h5py
import numpy as np
from config import config
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F

class KVDataset(Dataset):
    def __init__(self, path, shuffle=True):
        self.file = h5py.File(path, 'r')
        self.metadata = dict(self.file.attrs)
        self.keys = self.file['keys']
        self.values = self.file['values']
        self.perm = torch.randperm(len(self.keys)) if shuffle else torch.arange(len(self.keys))

    def __len__(self):
        return len(self.perm)

    def __getitem__(self, idx):
        real_idx = self.perm[idx].item()
        k = torch.tensor(self.keys[real_idx], dtype=torch.float16)
        v = torch.tensor(self.values[real_idx], dtype=torch.float16)
        return k, v

    def close(self):
        self.file.close()

def train_pseudo_tokens(dataset, pseudo_context_length, save_dir, lr):
    head_dim = dataset.metadata['head_dim']
    num_layers = dataset.metadata['num_layers']
    num_heads = dataset.metadata['num_heads']

    pseudo_context_K = nn.Parameter(torch.randn(num_layers, num_heads, pseudo_context_length, head_dim))
    pseudo_context_V = nn.Parameter(torch.randn(num_layers, num_heads, pseudo_context_length, head_dim))
    optimizer = optim.AdamW([pseudo_context_K, pseudo_context_V], lr=lr)

    loader = DataLoader(dataset, batch_size=256, num_workers=0)
    for keys, values in loader:
        hidden = torch.einsum('blhd,lhkd->blhk', keys.float(), pseudo_context_K)
        hidden = F.softmax(hidden, dim=-1)
        recon_values = torch.einsum('blhk,lhkd->blhd', hidden, pseudo_context_V)
        loss = nn.MSELoss()(recon_values, values.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

    torch.save({
        'pseudo_context_K': pseudo_context_K.detach().cpu(),
        'pseudo_context_V': pseudo_context_V.detach().cpu(),
        'model_name': dataset.metadata['model_name'],
        'dataset': dataset.metadata['dataset'],
        'num_layers': dataset.metadata['num_layers'],
        'num_heads': dataset.metadata['num_heads'],
        'head_dim': dataset.metadata['head_dim'],
    }, os.path.join(save_dir, 'pseudo_tokens.pt'))

    dataset.close()


if __name__ == "__main__":
    dataset_path = os.path.join(config['dataset_path'], 'kv_dataset.h5')
    dataset = KVDataset(dataset_path, shuffle=True)
    print(f"Metadata: {dataset.metadata}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0][0].shape}")

    loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=0)
    
    train_pseudo_tokens(dataset, pseudo_context_length=config['pseudo_context_length'], save_dir=config['save_dir'], lr=config['lr'])