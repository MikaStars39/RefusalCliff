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
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    repeat_kv,
    Qwen2MLP,
    Qwen2RMSNorm,
)

logger = logging.get_logger(__name__)

def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    scale: dict = None,
    input_shape: tuple = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    if scale is not None:
        attn_output_o_norm = attn_output.reshape(*input_shape, -1).contiguous().norm()
        for head_idx in scale["heads"]:
            attn_output[:, :, head_idx, :] = attn_output[:, :, head_idx, :] * scale["values"][module.layer_idx][head_idx]
        attn_output = attn_output / (attn_output.norm() / attn_output_o_norm)

    return attn_output, None

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
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = sdpa_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            scale=scale,
            input_shape=input_shape,
            **kwargs,
        )

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