import torch
from torch import nn
import torch.nn.functional as F
torch.set_float32_matmul_precision('high')

@torch.compile
def update_pseudo_contexts(K_chunk, V_chunk, WK, WV, lr_chunk=None, lr_base=0.1):
    B, n_head, chunk_size, d_head = K_chunk.size()

    attn_scores = torch.einsum('bhpd,bhkd->bhkp', WK, K_chunk) / (d_head ** 0.5)
    attn_weights = F.softmax(attn_scores, dim=-1)
    V_pred = torch.einsum('bhkp,bhpd->bhkd', attn_weights, WV)
    
    d_loss = V_pred - V_chunk

    if lr_chunk is not None:
        d_loss = d_loss * lr_chunk[:, :, :, None] * lr_base

    d_WV = torch.einsum('bhkp,bhkd->bhpd', attn_weights, d_loss)
    d_attn_weights = torch.einsum('bhkd,bhpd->bhkp', d_loss, WV)
    d_attn_scores = attn_weights * (d_attn_weights - 
                                     (attn_weights * d_attn_weights).sum(dim=-1, keepdim=True))
    d_WK = torch.einsum('bhkp,bhkd->bhpd', d_attn_scores, K_chunk) / (d_head ** 0.5)

    max_norm = 2.0
    total_norm = torch.sqrt((d_WK ** 2).sum() + (d_WV ** 2).sum())
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        d_WK = d_WK * clip_coef
        d_WV = d_WV * clip_coef

    if lr_chunk is None:
        return WK - lr_base * d_WK, WV - lr_base * d_WV
    
    return WK - d_WK, WV - d_WV

@torch.compile
def TTT_softmax_parallel(Q, K, V, init_WK, init_WV, chunk_size, history_size=512, lr=None, lr_base=0.01, pseudo_token_mask=0.0, debug=None):
    B, n_head, seq_len, d_head = Q.size()
    n_pseudo = init_WK.size(1)
    outputs = []
    
    WK, WV = init_WK.repeat(B, 1, 1, 1), init_WV.repeat(B, 1, 1, 1)
    
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size, seq_len)
        chunk_len = end - i
        
        Q_chunk = Q[:, :, i:end, :]
        K_chunk = K[:, :, max(0, i - history_size):end, :]
        V_chunk = V[:, :, max(0, i - history_size):end, :]

        lr_chunk = lr[:, :, max(0, i - history_size):end] if lr is not None else None
        
        K_with_pseudo = torch.cat([WK, K_chunk], dim=2)
        V_with_pseudo = torch.cat([WV, V_chunk], dim=2)
        
        mask = torch.zeros(chunk_len, K_with_pseudo.size(2), device=Q.device, dtype=Q.dtype)
        mask[:, n_pseudo + min(i, history_size):] = torch.triu(torch.ones(chunk_len, chunk_len, device=Q.device) * float('-inf'), diagonal=1)
        mask[:, :n_pseudo] = pseudo_token_mask

        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q_chunk, K_with_pseudo) / (d_head ** 0.5)
        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Debugging metrics
        if debug is not None:
            with torch.no_grad():
                last_attn = attn_weights[:, :, -1, :]  # [B, H, K_total]

                pseudo_attn = last_attn[:, :, :n_pseudo]
                real_attn = last_attn[:, :, n_pseudo:]

                debug.add("pseudo_mass", pseudo_attn.sum(dim=-1))
                debug.add("real_mass", real_attn.sum(dim=-1))

                if real_attn.size(-1) > 0:
                    near = min(history_size, real_attn.size(-1))
                    debug.add("near_mass", real_attn[:, :, -near:].sum(dim=-1))

                    if real_attn.size(-1) > near:
                        debug.add("far_mass", real_attn[:, :, :-near].sum(dim=-1))
                    else:
                        debug.add("far_mass", torch.zeros_like(real_attn[:, :, 0]))

                if debug.save_full_attn:
                    debug.add("full_attn", attn_weights)

                    
        output_chunk = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V_with_pseudo)
        
        outputs.append(output_chunk)
        
        WK, WV = update_pseudo_contexts(K_chunk, V_chunk, WK, WV, lr_chunk, lr_base)
    
    return torch.cat(outputs, dim=2)

if __name__ == "__main__":
    B, n_head, seq_len, d_head = 2, 4, 4096, 8
    chunk_size = 1024
    history_size = 512
    n_pseudo = 2
    
    Q = torch.randn(B, n_head, seq_len, d_head)
    K = torch.randn(B, n_head, seq_len, d_head)
    V = torch.randn(B, n_head, seq_len, d_head)
    
    init_WK = torch.randn(n_head, n_pseudo, d_head)
    init_WV = torch.randn(n_head, n_pseudo, d_head)
    
    output = TTT_softmax_parallel(Q, K, V, init_WK, init_WV, chunk_size, history_size, lr_base=0.01)
    print(output.shape)