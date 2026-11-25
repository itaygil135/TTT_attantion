import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import threading
import queue
from tqdm import tqdm
import os

# Global tokenizer
TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
TOKENIZER.pad_token = TOKENIZER.eos_token


class StreamingFineWebDataset(IterableDataset):
    def __init__(self, seq_len=512, buffer_size=1000, split='train', local_path=None):
        super().__init__()
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.split = split
        if local_path:
            self.dataset = load_dataset('json', data_files=local_path, split=split, streaming=True)
        else:
            self.dataset = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split=split, streaming=True)
        self.current_index = 0
        
    def __iter__(self):
        buffer = []
        prefetch_queue = queue.Queue(maxsize=10)
        
        def prefetch_worker():
            for item in self.dataset:
                text = item['text']
                tokens = TOKENIZER.encode(text, add_special_tokens=True)
                
                if len(tokens) >= self.seq_len:
                    tokens = tokens[:self.seq_len]
                    prefetch_queue.put(tokens)
                    
                    if prefetch_queue.qsize() >= self.buffer_size:
                        break
        
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()
        
        for item in self.dataset:
            text = item['text']
            tokens = TOKENIZER.encode(text, add_special_tokens=True)
            
            if len(tokens) >= self.seq_len:
                tokens = tokens[:self.seq_len]
                buffer.append(tokens)
                self.current_index += 1
                
                if len(buffer) >= self.buffer_size:
                    for tokens in buffer:
                        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                        labels = torch.tensor(tokens[1:], dtype=torch.long)
                        yield input_ids, labels
                    buffer = []
        
        if buffer:
            for tokens in buffer:
                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                labels = torch.tensor(tokens[1:], dtype=torch.long)
                yield input_ids, labels


def collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return input_ids, labels


def get_dataloader(seq_len=512, batch_size=32, num_workers=0, local_path=None):
    dataset = StreamingFineWebDataset(seq_len=seq_len, local_path=local_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )


def train(model, optimizer, dataloader, num_steps, device, save_path, grad_clip=1.0, log_interval=100):
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    step = 0
    total_loss = 0
    pbar = tqdm(total=num_steps, desc="Training")
    
    dataloader_iter = iter(dataloader)
    
    while step < num_steps:
        try:
            input_ids, labels = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            input_ids, labels = next(dataloader_iter)
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        step += 1
        
        pbar.update(1)
        pbar.set_postfix({'loss': total_loss / step, 'current_loss': loss.item()})
        
        if step % log_interval == 0:
            avg_loss = total_loss / step
            print(f"\nStep {step}/{num_steps} - Avg Loss: {avg_loss:.4f}")
    
    pbar.close()
    
    print(f"\nTraining complete. Saving model to {save_path}")
    model.save_model(save_path)
    print(f"Model saved successfully")
    
    return total_loss / num_steps


if __name__ == '__main__':
    from model import Transformer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Transformer(
        vocab_size=TOKENIZER.vocab_size,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    dataloader = get_dataloader(seq_len=512, batch_size=64, num_workers=0)
    
    train(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        num_steps=200000,
        device=device,
        save_path='checkpoints/model.pt',
        grad_clip=1.0,
        log_interval=100
    )