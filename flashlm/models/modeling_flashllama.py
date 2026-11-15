from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    LlamaMLP,
    LlamaAttention,
    LlamaModel,
    LlamaForCausalLM,
)


class FlashLlamaMLP(LlamaMLP):
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FlashLlamaAttention(LlamaAttention):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(q_len, device=hidden_states.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary_emb(value_states, position_ids=position_ids)
        
        # kv_seq_len = key_states.shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # position_ids = torch.arange(0, q_len, dtype=torch.long, device=hidden_states.device)
        # position_ids = position_ids.unsqueeze(0).view(-1, q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class FlashLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = FlashLlamaAttention(config=config)
        self.mlp = FlashLlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FlashLlamaModel(LlamaModel):
    supports_gradient_checkpointing = False
    _no_split_modules = ["FlashLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([FlashLlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class FlashLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = FlashLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits

