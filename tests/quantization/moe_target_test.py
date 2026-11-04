# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The MoE FP8 config quantizes exactly the target modules."""

import copy

import modelopt.torch.quantization as mtq
import pytest
import torch
from torch import nn

from cosmos_framework.quantization.moe import (
    MOE_FP8_TARGET,
    assert_target_coverage,
    build_moe_quant_config,
    count_moe_fp8_targets,
)


class TinyLayerModel(nn.Module):
    """Layer 0 with a target ``q_proj`` and a non-target fused ``qkv_proj``."""

    def __init__(self):
        super().__init__()
        self.net = nn.Module()
        self.net.language_model = nn.Module()
        self.net.language_model.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        layer.self_attn.qkv_proj = nn.Linear(4, 4, bias=False)
        self.net.language_model.model.layers = nn.ModuleList([layer])

    def forward(self, x):
        layer = self.net.language_model.model.layers[0]
        return layer.self_attn.qkv_proj(layer.self_attn.q_proj(x))


def test_target_matches_projections_and_experts_only():
    layer = "net.language_model.model.layers.7."
    for name in ("self_attn.k_proj", "self_attn.o_proj_moe_gen", "mlp.experts", "mlp_moe_gen.shared_expert.down_proj"):
        assert MOE_FP8_TARGET.fullmatch(layer + name), name
    for name in ("self_attn.qkv_proj", "mlp.gate", "mlp_moe_gen.shared_expert_gate", "input_layernorm"):
        assert not MOE_FP8_TARGET.fullmatch(layer + name), name


def test_config_quantizes_only_the_target():
    model = TinyLayerModel()
    x = torch.randn(2, 4)

    mtq.quantize(model, build_moe_quant_config(model), lambda m: m(x))

    assert_target_coverage(model)
    layer = model.net.language_model.model.layers[0]
    assert layer.self_attn.q_proj.weight_quantizer.is_enabled
    assert not layer.self_attn.qkv_proj.weight_quantizer.is_enabled


def test_coverage_rejects_a_quantized_non_target():
    model = TinyLayerModel()
    mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m(torch.randn(2, 4)))
    with pytest.raises(ValueError, match=r"extra \['net\.language_model\.model\.layers\.0\.self_attn\.qkv_proj'\]"):
        assert_target_coverage(model)


def _layers_model(*mlps_moe_gen: nn.Module) -> nn.Module:
    model = nn.Module()
    model.net = nn.Module()
    model.net.language_model = nn.Module()
    model.net.language_model.model = nn.Module()
    model.net.language_model.model.layers = nn.ModuleList()
    for mlp in mlps_moe_gen:
        layer = nn.Module()
        layer.mlp_moe_gen = mlp
        model.net.language_model.model.layers.append(layer)
    return model


def _moe(shared_expert: bool) -> nn.Module:
    mlp = nn.Module()
    mlp.experts = nn.Module()
    mlp.shared_expert = nn.Module() if shared_expert else None
    return mlp


def test_count_adds_the_shared_expert_only_where_present():
    assert count_moe_fp8_targets(_layers_model(_moe(shared_expert=True), _moe(shared_expert=False))) == 12 + 10
