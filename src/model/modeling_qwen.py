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
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_layers import GradientCheckpointingLayer

from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen2MLP,
    Qwen2RMSNorm,
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
    input_shape: tuple = None,
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

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output_o = attn_output.transpose(1, 2).contiguous()
    attn_output_o = attn_output_o.reshape(*input_shape, -1).contiguous()
    attn_output_o_norm = attn_output_o.norm()

    if scale is not None:
        for head_idx in scale["heads"]:
            attn_output[:, head_idx, :, :] = attn_output[:, head_idx, :, :] * scale["values"][module.layer_idx][head_idx]
        attn_output = attn_output / (attn_output.norm() / attn_output_o_norm)

    return attn_output, attn_weights


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        # Qwen2 specific: sliding window support
        self.sliding_window = getattr(config, 'sliding_window', None)
        if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types):
            if config.layer_types[layer_idx] == "sliding_attention":
                self.sliding_window = config.sliding_window
        else:
            self.sliding_window = None

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
        scale = None
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
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            scale=scale,
            input_shape=input_shape,
            **kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        if hasattr(self, "refusal_head"):
            all_outputs = []
            for head_idx in range(attn_output.shape[2]):
                zero_tensor = torch.zeros_like(attn_output).to(attn_output.device)
                zero_tensor[:, :, head_idx, :] = attn_output[:, :, head_idx, :]
                zero_tensor = self.o_proj(zero_tensor.reshape(*input_shape, -1).contiguous()[:, -1, :]).mean(dim=0)
                refusal_vector = self.refusal_head["refusal_vector"].to(zero_tensor.device).to(zero_tensor.dtype)
                # rescale zero vector and refusal_vector to the same norm
                score = torch.dot(zero_tensor, refusal_vector) / (refusal_vector.norm()).cpu()
                all_outputs.append(score.item())
                
            if len(self.refusal_head["all_outputs"]) == 0:
                self.refusal_head["all_outputs"] = all_outputs
            else:
                for idx, output in enumerate(all_outputs):
                    self.refusal_head["all_outputs"][idx] += output

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

 
class Qwen2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Qwen2 specific: attention type tracking
        if hasattr(config, 'layer_types') and layer_idx < len(config.layer_types):
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"  # default

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if hasattr(self, "refusal_direction"):
            scale = self.scale / self.refusal_direction.norm() * hidden_states.norm()
            hidden_states = hidden_states + self.refusal_direction * scale

        return hidden_states


def enable_monkey_patched_qwen(model):
    # recursively patch the model to the attention layer
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(
                    module,
                )
            if "self_attn" == name[-9:]:
                model._modules[name].forward = types.MethodType(
                    Qwen2Attention.forward, model._modules[name]
                )

    recursive_patch(model)

def enable_monkey_patched_qwen_decoder(model):
    """Monkey patch the Qwen2DecoderLayer forward method"""
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(module)
            # Check if this is a decoder layer by looking for common patterns
            if hasattr(module, 'self_attn') and hasattr(module, 'mlp') and hasattr(module, 'input_layernorm'):
                model._modules[name].forward = types.MethodType(
                    Qwen2DecoderLayer.forward, model._modules[name]
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

def clean_property(model, module_name, property_name):
    # recursively patch the model
    def recursive_patch(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                recursive_patch(
                    module,
                )
            if module_name in name:
                delattr(model._modules[name], property_name)
    
    recursive_patch(model)

def set_thinking_scale_heads(model, head_indices, layer_idx=None):
    """Set which attention heads should have thinking tokens scaled down
    
    Args:
        model: The model to modify
        head_indices: List of head indices to scale (e.g., [0, 1, 5, 10])
        layer_idx: If specified, only set for that layer. If None, set for all layers.
    """
    if layer_idx is not None:
        # Set for specific layer
        add_property(model, f"layers.{layer_idx}.self_attn", "thinking_scale_heads", head_indices)
    else:
        # Set for all attention layers
        add_property(model, "self_attn", "thinking_scale_heads", head_indices) 