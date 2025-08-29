# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
import types

from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    LlamaMLP,
    repeat_kv,
)

logger = logging.get_logger(__name__)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    scale: dict = None,
    **kwargs: Unpack[TransformersKwargs],
):  
    
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Handle different attention mask shapes
        if attention_mask.dim() == 2:
            # For 2D masks (batch_size, seq_len), expand to 4D
            batch_size, seq_len = attention_mask.shape
            # Create causal mask
            causal_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
        elif attention_mask.dim() == 4:
            # For 4D masks, use as-is but slice to key length
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        else:
            # For other dimensions, try to handle gracefully
            causal_mask = attention_mask
        
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    if scale is not None:
        head_idx = torch.tensor(scale["heads"]).to(query.device)
        attn_weights[:, head_idx, :, :] = attn_weights[:, head_idx, :, :] * scale["values"]

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # check if self has scale
        if hasattr(self, "scale"):
            scale = self.scale
            # check if self.layer_idx in scale["heads"].keys()
            if self.layer_idx in scale["heads"].keys():
                # Create a new scale dict with heads for current layer
                scale = {
                    "heads": scale["heads"][self.layer_idx],
                    "values": scale["values"]
                }
            else:
                scale = None
        else:
            scale = None

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            scale=scale,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
 
def enable_monkey_patched_llama(model):
    # recursively patch the model to the attention layer
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(
                    module,
                )
            if "self_attn" == name[-9:]:
                model._modules[name].forward = types.MethodType(
                    LlamaAttention.forward, model._modules[name]
                )

    recursive_patch(model)

def add_property(model, module_name, property_name, property_value):
    # recursively patch the model
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(
                    module,
                )
            if module_name in name:
                setattr(model._modules[name], property_name, property_value)

    recursive_patch(model)
