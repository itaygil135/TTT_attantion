import torch
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .operations.TTT_softmax_parallel import TTT_softmax_parallel


class DebugCollector:
    def __init__(self, save_full_attn=False):
        self.save_full_attn = save_full_attn
        self.data = {}

    def add(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value.detach().float().cpu())

    def clear(self):
        self.data.clear()


class TTTAttentionWrapper(nn.Module):
    def __init__(
        self,
        original_attn,
        init_WK,
        init_WV,
        chunk_size=256,
        history_size=256,
        dependent_lr=False,
        lr_base=0.01,
        pseudo_token_mask=0.0,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.chunk_size = chunk_size
        self.history_size = history_size
        self.lr_base = lr_base
        self.pseudo_token_mask = pseudo_token_mask

        self.WK = nn.Parameter(init_WK.clone())
        self.WV = nn.Parameter(init_WV.clone())

        self.debug = None

        self.lr_proj = None
        if dependent_lr:
            cfg = original_attn.config
            self.lr_proj = nn.Sequential(
                nn.Linear(cfg.hidden_size, cfg.num_attention_heads),
                nn.Sigmoid(),
            )
            self.lr_proj[0].bias.data.fill_(0.0)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        attn = self.original_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_norm(
            attn.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        key_states = attn.k_norm(
            attn.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if attn.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(
                attn.num_key_value_groups, dim=1
            )
            value_states = value_states.repeat_interleave(
                attn.num_key_value_groups, dim=1
            )

        lr = None
        if self.lr_proj is not None:
            lr = self.lr_proj(hidden_states).transpose(1, 2)

        attn_output = TTT_softmax_parallel(
            query_states,
            key_states,
            value_states,
            self.WK,
            self.WV,
            self.chunk_size,
            self.history_size,
            lr=lr,
            lr_base=self.lr_base,
            pseudo_token_mask=self.pseudo_token_mask,
            debug=self.debug,
        )

        attn_output = (
            attn_output.transpose(1, 2)
            .reshape(*input_shape, -1)
            .contiguous()
        )

        attn_output = attn.o_proj(attn_output)

        return attn_output, None


class LoRAWrapper(nn.Module):
    def __init__(self, original_linear, rank=4):
        super().__init__()
        self.original_linear = original_linear

        d_out, d_in = original_linear.weight.shape
        self.A = nn.Parameter(torch.randn(d_out, rank) * (2 / (d_out + rank)) ** 0.5)
        self.B = nn.Parameter(torch.zeros(rank, d_in))

    def forward(self, x):
        return self.original_linear(x) + (x @ self.B.t() @ self.A.t())


class Qwen3TTTWrapper(nn.Module):
    def __init__(
        self,
        model_name,
        init_WK_all,
        init_WV_all,
        chunk_size=256,
        history_size=256,
        lora_mlp=False,
        lora_qkv=False,
        lora_rank=4,
        dependent_lr=False,
        lr_base=0.01,
        pseudo_token_mask=0.0,
    ):
        super().__init__()

        self.lora_mlp = lora_mlp
        self.lora_qkv = lora_qkv
        self.lora_rank = lora_rank
        self.dependent_lr = dependent_lr
        self.lr_base = lr_base
        self.pseudo_token_mask = pseudo_token_mask

        self.debug = None

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.requires_grad_(False)

        text_model = self.model.model

        for i, layer in enumerate(text_model.layers):
            layer_type = getattr(layer.self_attn, "layer_type", "full_attention")

            if layer_type != "full_attention":
                continue

            layer.self_attn = TTTAttentionWrapper(
                original_attn=layer.self_attn,
                init_WK=init_WK_all[i],
                init_WV=init_WV_all[i],
                chunk_size=chunk_size,
                history_size=history_size,
                dependent_lr=dependent_lr,
                lr_base=lr_base,
                pseudo_token_mask=pseudo_token_mask,
            )

            if lora_mlp:
                mlp = layer.mlp
                mlp.gate_proj = LoRAWrapper(mlp.gate_proj, rank=lora_rank)
                mlp.up_proj = LoRAWrapper(mlp.up_proj, rank=lora_rank)
                mlp.down_proj = LoRAWrapper(mlp.down_proj, rank=lora_rank)

            if lora_qkv:
                attn = layer.self_attn.original_attn
                attn.q_proj = LoRAWrapper(attn.q_proj, rank=lora_rank)
                attn.k_proj = LoRAWrapper(attn.k_proj, rank=lora_rank)
                attn.v_proj = LoRAWrapper(attn.v_proj, rank=lora_rank)

    def set_debug(self, debug):
        self.debug = debug

        for layer in self.model.model.layers:
            if isinstance(layer.self_attn, TTTAttentionWrapper):
                layer.self_attn.debug = debug

    def clear_debug(self):
        self.set_debug(None)

    def reset_pseudo_contexts(self, init_WK_all, init_WV_all):
        text_model = self.model.model

        for i, layer in enumerate(text_model.layers):
            if isinstance(layer.self_attn, TTTAttentionWrapper):
                layer.self_attn.WK.data.copy_(init_WK_all[i])
                layer.self_attn.WV.data.copy_(init_WV_all[i])

    def forward(self, input_ids, attention_mask=None, debug=None, **kwargs):
        if debug is not None:
            self.set_debug(debug)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )

        return outputs