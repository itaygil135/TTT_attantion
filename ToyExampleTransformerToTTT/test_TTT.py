import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import model
from model import Transformer
import torch

def test_mlp_update():
    n_heads = 2
    head_dim = 4
    hidden_dim = 8
    L = 512
    B = 3

    W0, W1 = model.init_Ws(n_heads, head_dim, hidden_dim, init_K=1.0, init_V=0.01, device='cpu')

    q = torch.randn(B, n_heads, L, head_dim)
    k = torch.randn(B, n_heads, L, head_dim)
    v = torch.randn(B, n_heads, L, head_dim)

    lr = 0.01

    output, new_W0, new_W1 = model.TTT_MLP_operation_softmax(W0, W1, q, k, v, lr)

    assert output.shape == (B, n_heads, L, head_dim), "Output shape is incorrect"
    assert new_W0.shape == W0.repeat(B, 1, 1).shape, "W0 shape is incorrect"
    assert new_W1.shape == W1.repeat(B, 1, 1).shape, "W1 shape is incorrect"
    assert not torch.isnan(output).any(), "there are Nan values in output"
    assert not torch.isnan(new_W0).any(), "there are Nan values in new_W0"
    assert not torch.isnan(new_W1).any(), "there are Nan values in new_W1"

def test_whole_model():
    # defining model
    model = Transformer(
        vocab_size=100,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.1
    )

    # dummy input
    input_ids = torch.randint(0, 100, (2, 10))

    # define experiment kwargs
    experiment_kwargs = {
        'ttt_lr': 1.0,
        'ttt_init_K': 1.0,
        'ttt_init_V': 0.01,
        'special_layers': [0, 1],
        'TTT_hidden_dim': 32,
    }
    mode = 'TTT_MLP_softmax'

    # try special forward
    logit = model.special_forward(input_ids, mode, **experiment_kwargs)

    assert logit.shape == (2, 10, 100), "Logit shape is incorrect"
    assert not torch.isnan(logit).any(), "there are Nan values in logit"


if __name__ == "__main__":
    try:
        test_mlp_update()
        print("test_mlp_update passed.")
        test_whole_model()
        print("test_whole_model passed.")
    except Exception as e:
        print(f"Test failed: {e}")
    print("All tests passed.")