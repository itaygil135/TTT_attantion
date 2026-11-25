"""
In this module, we check the loss of the model using the special TTT forward pass.
We first load samples from the dataset, then define the model and experiment parameters.
Finally, we compute the NTP loss and ensure it is better in TTT than skipping.
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import model
from model import Transformer
from train import TOKENIZER, get_dataloader
import torch
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

def test_ttt_loss_improvement(ttt_experiment_kwargs, validation_steps=10):
    # dataloader
    dataloader = get_dataloader(seq_len=512, batch_size=64, num_workers=0, local_path='/data/mini_fineweb_edu_lc/fineweb_edu_subset.jsonl')

    # Define model
    model_instance = Transformer.load_model('checkpoints/model.pt', device='cuda')

    model_instance.eval()

    skipping_losses = []
    ttt_losses = []
    ttt_two_hidden_losses = []
    normal_losses = []

    for idx, batch in enumerate(dataloader):
        if idx >= validation_steps:
            break

        input_ids, labels = batch

        with torch.no_grad():
            # Skipping TTT
            logits_skipping = model_instance.special_forward(input_ids.cuda(), mode='skipping', **ttt_experiment_kwargs)
            loss_skipping = F.cross_entropy(logits_skipping.view(-1, logits_skipping.size(-1)), labels.cuda().view(-1))
            skipping_losses.append(loss_skipping.item())

            # TTT forward
            logits_ttt = model_instance.special_forward(input_ids.cuda(), mode='TTT_MLP', **ttt_experiment_kwargs)
            loss_ttt = F.cross_entropy(logits_ttt.view(-1, logits_ttt.size(-1)), labels.cuda().view(-1))
            ttt_losses.append(loss_ttt.item())

            # TTT forward two layers
            logits_ttt = model_instance.special_forward(input_ids.cuda(), mode='TTT_MLP_two_hidden', **ttt_experiment_kwargs)
            loss_ttt = F.cross_entropy(logits_ttt.view(-1, logits_ttt.size(-1)), labels.cuda().view(-1))
            ttt_two_hidden_losses.append(loss_ttt.item())

            # normal forward
            logits_normal = model_instance(input_ids.cuda())
            loss_normal = F.cross_entropy(logits_normal.view(-1, logits_normal.size(-1)), labels.cuda().view(-1))
            normal_losses.append(loss_normal.item())


    avg_skipping_loss = sum(skipping_losses) / len(skipping_losses)
    avg_ttt_loss = sum(ttt_losses) / len(ttt_losses)
    avg_ttt_two_hidden_loss = sum(ttt_two_hidden_losses) / len(ttt_two_hidden_losses)
    avg_normal_loss = sum(normal_losses) / len(normal_losses)

    print(f"Average Normal Loss: {avg_normal_loss}")
    print(f"Average Skipping Loss: {avg_skipping_loss}")
    print(f"Average TTT Loss: {avg_ttt_loss}")
    print(f"Average TTT Two Hidden Loss: {avg_ttt_two_hidden_loss}")

    assert avg_ttt_loss < avg_skipping_loss, "TTT loss is not better than skipping loss"

if __name__ == "__main__":
    ttt_experiment_kwargs = {
        'ttt_lr': .1,
        'lr_sqrt_decay': False,
        'ttt_init_K': 0.005,
        'ttt_init_V': 0.01,
        'special_layers': [1, 3, 4, 6, 7],
        'TTT_hidden_dim': 16,
        'activation': 'sigmoid',
        'activation1': 'sigmoid',
        'activation2': 'softmax',
        'init_mode': 'normal',
    }
    test_ttt_loss_improvement(ttt_experiment_kwargs)
    print("TTT loss improvement test passed successfully.")