import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len):
        """
        x: (batch, seq_len, n_heads, head_dim)
        """
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        cos = emb.cos()[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = emb.sin()[None, :, None, :]  # (1, seq_len, 1, dim)
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to x"""
    # x: (batch, seq_len, n_heads, head_dim)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    # Rotate
    rx1 = x1 * cos[..., : x.shape[-1] // 2] - x2 * sin[..., x.shape[-1] // 2 :]
    rx2 = x1 * sin[..., : x.shape[-1] // 2] + x2 * cos[..., x.shape[-1] // 2 :]
    
    return torch.cat([rx1, rx2], dim=-1)

def backward_softmax(hidden, d_hidden):
    # before the softmax we apply normalization by sqrt(d_k)
    # so we need to take that into account in the backward pass
    s_dot_grad = torch.sum(hidden * d_hidden, dim=-1, keepdim=True)
    return hidden * (d_hidden - s_dot_grad) / (hidden.shape[-1] ** 0.5)


def backward_silu(hidden_pre_act, d_hidden):
    sigmoid_x = torch.sigmoid(hidden_pre_act)
    silu_x = hidden_pre_act * sigmoid_x
    return d_hidden * (silu_x + sigmoid_x * (1 - silu_x))


def backward_gelu(hidden_pre_act, d_hidden):
    cdf = 0.5 * (1.0 + torch.erf(hidden_pre_act / math.sqrt(2.0)))
    pdf = torch.exp(-0.5 * hidden_pre_act ** 2) / math.sqrt(2.0 * math.pi)
    return d_hidden * (cdf + hidden_pre_act * pdf)

def backward_sigmoid(hidden_pre_act, d_hidden):
    sigmoid_x = torch.sigmoid(hidden_pre_act)
    return d_hidden * sigmoid_x * (1 - sigmoid_x)

def apply_activation(hidden_pre_act, activation='softmax'):
    if activation == 'softmax':
        return F.softmax(hidden_pre_act / hidden_pre_act.shape[-1] ** 0.5, dim=-1)
    elif activation == 'silu':
        return F.silu(hidden_pre_act)
    elif activation == 'gelu':
        return F.gelu(hidden_pre_act)
    elif activation == 'sigmoid':
        return F.sigmoid(hidden_pre_act)
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def backward_activation(hidden_pre_act, hidden, d_hidden, activation='softmax'):
    if activation == 'softmax':
        return backward_softmax(hidden, d_hidden)
    elif activation == 'silu':
        return backward_silu(hidden_pre_act, d_hidden)
    elif activation == 'gelu':
        return backward_gelu(hidden_pre_act, d_hidden)
    elif activation == 'sigmoid':
        return backward_sigmoid(hidden_pre_act, d_hidden)
    else:
        raise ValueError(f"Unsupported activation: {activation}")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, L, D = x.shape

        # Project and reshape to (B, L, n_heads, head_dim)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        # Apply RoPE (RoPE expects (B, L, n_heads, head_dim))
        cos, sin = self.rope(q, L)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose to (B, n_heads, L, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (flash-attention like)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

    def special_forward(self, x, mode='normal', **kwargs):
        if mode == 'skipping':
            return x

        B, L, D = x.shape

        # Project and reshape to (B, L, n_heads, head_dim)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)

        # Apply RoPE
        cos, sin = self.rope(q, L)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose to (B, n_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mode == 'normal':
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=True)
        elif mode == 'TTT_MLP':
            TTT_hidden_dim = kwargs.get('TTT_hidden_dim')
            init_K = kwargs.get('init_K', 1.0)
            init_V = kwargs.get('init_V', .01)
            lr = kwargs.get('lr', 0.01)
            lr_sqrt_decay = kwargs.get('lr_sqrt_decay', False)
            init_mode = kwargs.get('init_mode', 'normal')
            activation = kwargs.get('activation', 'softmax')

            W0, W1 = init_Ws(self.n_heads, self.head_dim, TTT_hidden_dim, init_K, init_V, init_mode, device=x.device)

            out, _, _ = TTT_MLP_operation(W0, W1, q, k, v, lr=lr, activation=activation, lr_sqrt_decay=lr_sqrt_decay)
        elif mode == 'TTT_MLP_two_hidden':
            TTT_hidden_dim = kwargs.get('TTT_hidden_dim')
            init_K = kwargs.get('init_K', 1.0)
            init_V = kwargs.get('init_V', .01)
            lr = kwargs.get('lr', .1)
            lr_sqrt_decay = kwargs.get('lr_sqrt_decay', False)
            init_mode = kwargs.get('init_mode', 'normal')
            activation1 = kwargs.get('activation1', 'silu')
            activation2 = kwargs.get('activation2', 'sigmoid')

            W0, W1, W2 = init_Ws_two_hidden_layers(self.n_heads, self.head_dim, TTT_hidden_dim, TTT_hidden_dim, init_K, init_V, init_mode, device=x.device)

            out, _, _, _ = TTT_MLP_two_hidden_layers_operation(
                W0, W1, W2, q, k, v, lr=lr, activation1=activation1, activation2=activation2, lr_sqrt_decay=lr_sqrt_decay
            )
        else:
            raise ValueError("Unsupported mode for special_forward")

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

    def special_forward(self, x, mode='normal', **kwargs):
        x = x + self.attn.special_forward(self.ln1(x), mode=mode, **kwargs)
        x = x + self.ff(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, d_ff=3072, 
                 max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    @torch.compile(fullgraph=True)
    def forward(self, x, return_confidence_logits=False):
        x = self.token_emb(x)
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    def special_forward(self, x, mode='normal', **kwargs):
        special_layers = kwargs.get('special_layers')

        x = self.token_emb(x)
        x = self.dropout(x)
        
        for idx, block in enumerate(self.blocks):
            if special_layers is not None and idx in special_layers:
                x = block.special_forward(x, mode=mode, **kwargs)
            else:
                x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'config': {
                'n_layers': len(self.blocks),
                'n_heads': self.blocks[0].attn.n_heads,
                'd_ff': self.blocks[0].ff.net[0].out_features,
                'dropout': self.dropout.p,
            }
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_model(cls, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50, device='cuda'):
        self.eval()
        generated = input_ids.to(device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, top_k)
                    min_top_k = top_k_values[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_top_k, torch.full_like(logits, -float('Inf')), logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

        return generated

    def special_generate(self, input_ids, max_length, mode='normal', special_layers=None, 
                         temperature=1.0, top_k=50, device='cuda', **kwargs):
        self.eval()
        generated = input_ids.to(device)

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.special_forward(generated, mode=mode, special_layers=special_layers, **kwargs)
                logits = logits[:, -1, :] / temperature

                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, top_k)
                    min_top_k = top_k_values[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_top_k, torch.full_like(logits, -float('Inf')), logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

        return generated

def create_causal_mask(seq_len, device):
    """Create causal mask for autoregressive language modeling"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


def get_model_config(size='400M'):
    """Get model configuration for different sizes"""
    configs = {
        '400M': {
            'd_model': 1024,
            'n_layers': 24,
            'n_heads': 16,
            'd_ff': 4096,
        },
        '150M': {
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'd_ff': 3072,
        },
        '50M': {
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 8,
            'd_ff': 2048,
        }
    }
    return configs.get(size, configs['400M'])

def TTT_MLP_update_and_query(W0, W1, query, key, value, lr=0.01, activation='softmax'):
    with torch.no_grad():
        # Train on (key -> value)
        hidden_pre_act = torch.bmm(key.unsqueeze(1), W0).squeeze(1)
        hidden = apply_activation(hidden_pre_act, activation)
        output = torch.bmm(hidden.unsqueeze(1), W1).squeeze(1)
        
        # Gradients
        d_output = output - value
        d_W1 = torch.bmm(hidden.unsqueeze(-1), d_output.unsqueeze(1))
        
        d_hidden = torch.bmm(d_output.unsqueeze(1), W1.transpose(-2, -1)).squeeze(1)
        d_hidden_pre_act = backward_activation(hidden_pre_act, hidden, d_hidden, activation)
        d_W0 = torch.bmm(key.unsqueeze(-1), d_hidden_pre_act.unsqueeze(1))
        
        # Update weights
        W0.sub_(lr * d_W0)
        W1.sub_(lr * d_W1)
        
        # Test on query
        query_hidden_pre_act = torch.bmm(query.unsqueeze(1), W0).squeeze(1)
        query_hidden = apply_activation(query_hidden_pre_act, activation)
        output = torch.bmm(query_hidden.unsqueeze(1), W1).squeeze(1)
        
        return output, W0, W1
    
def TTT_MLP_operation(W0, W1, q, k, v, lr=0.01, activation='softmax', lr_sqrt_decay=False):
    B, n_heads, L, head_dim = q.shape
    hidden_dim = W0.shape[-1]
    
    cur_W0 = W0.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, head_dim, hidden_dim).contiguous()
    cur_W1 = W1.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, hidden_dim, head_dim).contiguous()
    
    outputs = []
    lr_scale = 1.0
    
    for i in range(L):
        if lr_sqrt_decay:
            lr_scale = 1.0 / ((i + 1) ** 0.5)

        query = q[:, :, i, :].reshape(B * n_heads, head_dim)
        key = k[:, :, i, :].reshape(B * n_heads, head_dim)
        value = v[:, :, i, :].reshape(B * n_heads, head_dim)
        
        out_i, cur_W0, cur_W1 = TTT_MLP_update_and_query(
            cur_W0, cur_W1, query, key, value, lr=lr * lr_scale, activation=activation
        )
        outputs.append(out_i.view(B, n_heads, 1, head_dim))
    
    output = torch.cat(outputs, dim=2)
    return output, cur_W0, cur_W1

def init_Ws(n_heads, head_dim, hidden_dim, init_K, init_V, mode='normal', device='cuda'):
    """
    Initialize W0 and W1 matrices with different strategies.
    
    Modes:
    - 'normal': Standard random initialization
    - 'rope': Apply RoPE to initial keys
    - 'low_rank': Low-rank initialization for simplicity
    - 'positional_smoothness': Adjacent positions correlated
    - 'asymmetric': W0 explores (larger), W1 adapts (smaller)
    - 'block_diagonal': Specialization via blocks
    - 'identity_biased': W1 biased toward identity
    """
    
    if mode == 'rope':
        K = torch.randn(1, hidden_dim, n_heads, head_dim, device=device) * init_K
        rope = RoPE(head_dim)
        cos, sin = rope(K, hidden_dim)
        K_rotated = apply_rotary_emb(K, cos, sin)
        W0 = K_rotated.squeeze(0).transpose(0, 1).transpose(1, 2)
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
        
    elif mode == 'low_rank':
        rank = hidden_dim // 4
        U = torch.randn(n_heads, head_dim, rank, device=device)
        V = torch.randn(n_heads, rank, hidden_dim, device=device)
        W0 = (U @ V) * init_K
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
        
    elif mode == 'positional_smoothness':
        W0 = torch.randn(n_heads, head_dim, hidden_dim, device=device)
        for i in range(1, hidden_dim):
            W0[:, :, i] = 0.7 * W0[:, :, i-1] + 0.3 * W0[:, :, i]
        W0 = W0 * init_K
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
        
    elif mode == 'block_diagonal':
        block_size = hidden_dim // 4
        W0 = torch.zeros(n_heads, head_dim, hidden_dim, device=device)
        for b in range(4):
            start = b * block_size
            end = start + block_size
            W0[:, :, start:end] = torch.randn(n_heads, head_dim, block_size, device=device) * init_K
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
        
    elif mode == 'identity_biased':
        W0 = torch.randn(n_heads, head_dim, hidden_dim, device=device) * init_K
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
        if hidden_dim >= head_dim:
            for h in range(n_heads):
                W1[h, :head_dim, :] += torch.eye(head_dim, device=device) * init_V
                
    else:  # 'normal'
        W0 = torch.randn(n_heads, head_dim, hidden_dim, device=device) * init_K
        W1 = torch.randn(n_heads, hidden_dim, head_dim, device=device) * init_V
    
    return W0, W1

def TTT_MLP_two_hidden_layers_update_and_query(W0, W1, W2, query, key, value, lr=0.01, activation1='silu', activation2='sigmoid'):
    with torch.no_grad():
        # Train on (key -> value)
        hidden1_pre_act = torch.bmm(key.unsqueeze(1), W0).squeeze(1)
        hidden1 = apply_activation(hidden1_pre_act, activation1)
        hidden2_pre_act = torch.bmm(hidden1.unsqueeze(1), W1).squeeze(1)
        hidden2 = apply_activation(hidden2_pre_act, activation2)
        output = torch.bmm(hidden2.unsqueeze(1), W2).squeeze(1)
        
        # Gradients
        d_output = output - value
        d_W2 = torch.bmm(hidden2.unsqueeze(-1), d_output.unsqueeze(1))
        
        d_hidden2 = torch.bmm(d_output.unsqueeze(1), W2.transpose(-2, -1)).squeeze(1)
        d_hidden2_pre_act = backward_activation(hidden2_pre_act, hidden2, d_hidden2, activation2)
        d_W1 = torch.bmm(hidden1.unsqueeze(-1), d_hidden2_pre_act.unsqueeze(1))
        
        d_hidden1 = torch.bmm(d_hidden2_pre_act.unsqueeze(1), W1.transpose(-2, -1)).squeeze(1)
        d_hidden1_pre_act = backward_activation(hidden1_pre_act, hidden1, d_hidden1, activation1)
        d_W0 = torch.bmm(key.unsqueeze(-1), d_hidden1_pre_act.unsqueeze(1))
        
        # Update weights
        W0.sub_(lr * d_W0)
        W1.sub_(lr * d_W1)
        W2.sub_(lr * d_W2)
        
        # Test on query
        query_hidden1_pre_act = torch.bmm(query.unsqueeze(1), W0).squeeze(1)
        query_hidden1 = apply_activation(query_hidden1_pre_act, activation1)
        query_hidden2_pre_act = torch.bmm(query_hidden1.unsqueeze(1), W1).squeeze(1)
        query_hidden2 = apply_activation(query_hidden2_pre_act, activation2)
        output = torch.bmm(query_hidden2.unsqueeze(1), W2).squeeze(1)
        
        return output, W0, W1, W2

def TTT_MLP_two_hidden_layers_operation(W0, W1, W2, q, k, v, lr=0.01, activation1='silu', activation2='sigmoid', lr_sqrt_decay=False):
    B, n_heads, L, head_dim = q.shape
    hidden_dim1 = W0.shape[-1]
    hidden_dim2 = W1.shape[-1]
    
    cur_W0 = W0.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, head_dim, hidden_dim1).contiguous()
    cur_W1 = W1.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, hidden_dim1, hidden_dim2).contiguous()
    cur_W2 = W2.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_heads, hidden_dim2, head_dim).contiguous()
    
    outputs = []
    lr_scale = 1.0
    
    for i in range(L):
        if lr_sqrt_decay:
            lr_scale = 1.0 / ((i + 1) ** 0.5)

        query = q[:, :, i, :].reshape(B * n_heads, head_dim)
        key = k[:, :, i, :].reshape(B * n_heads, head_dim)
        value = v[:, :, i, :].reshape(B * n_heads, head_dim)
        
        out_i, cur_W0, cur_W1, cur_W2 = TTT_MLP_two_hidden_layers_update_and_query(
            cur_W0, cur_W1, cur_W2, query, key, value, lr=lr * lr_scale,
            activation1=activation1, activation2=activation2
        )
        outputs.append(out_i.view(B, n_heads, 1, head_dim))
    
    output = torch.cat(outputs, dim=2)
    return output, cur_W0, cur_W1, cur_W2

def init_Ws_two_hidden_layers(n_heads, head_dim, hidden_dim1, hidden_dim2, init_K, init_V, mode='normal', device='cuda'):
    """
    Initialize W0, W1, and W2 matrices with different strategies for two hidden layers.
    """
    
    if mode == 'normal':
        W0 = torch.randn(n_heads, head_dim, hidden_dim1, device=device) * init_K
        W1 = torch.randn(n_heads, hidden_dim1, hidden_dim2, device=device) * init_K
        W2 = torch.randn(n_heads, hidden_dim2, head_dim, device=device) * init_V
    
    else:
        raise ValueError(f"Unsupported mode for two hidden layers: {mode}")
    
    return W0, W1, W2