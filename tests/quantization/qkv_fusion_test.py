# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The FP8 calibration path shares QKV scales and restores exportable modules."""

import copy

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import pytest
import torch
from torch import nn

from cosmos_framework.quantization.model_utils import share_qkv_weight_amax, temporary_fuse_named_linears


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        for scale, linear in zip((0.1, 1.0, 10.0), (self.q_proj, self.k_proj, self.v_proj), strict=True):
            nn.init.constant_(linear.weight, scale)

    def forward(self, x):
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


class SkewedAttention(Attention):
    def forward(self, x):
        return self.q_proj(x), self.k_proj(2 * x), self.v_proj(x)


class TinyFrameworkModel(nn.Module):
    def __init__(self, attention_cls=Attention):
        super().__init__()
        self.net = nn.Module()
        self.net.language_model = nn.Module()
        self.net.language_model.model = nn.Module()
        self.net.language_model.model.layers = nn.ModuleList([nn.Module()])
        self.net.language_model.model.layers[0].self_attn = attention_cls()

    def forward(self, x):
        return self.net.language_model.model.layers[0].self_attn(x)


def test_fp8_qkv_fusion_restores_modules_and_modelopt_state():
    model = TinyFrameworkModel()
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    pattern = r"net\.language_model\.model\.layers\.(\d+)\.self_attn$"

    with temporary_fuse_named_linears(model, pattern, "qkv_proj", ["q_proj", "k_proj", "v_proj"]):
        mtq.quantize(model, config, lambda: model(x))

    attention = model.net.language_model.model.layers[0].self_attn
    assert not hasattr(attention, "qkv_proj")
    modules = [getattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")]
    assert all(isinstance(module, nn.Linear) for module in modules)
    assert all(torch.equal(module.weight_quantizer.amax, torch.tensor(10.0)) for module in modules)
    assert all(torch.equal(module.input_quantizer.amax, modules[0].input_quantizer.amax) for module in modules)
    state = dict(mto.modelopt_state(model)["modelopt_state_dict"])["quantize"]["metadata"]["quantizer_state"]
    assert not any("qkv_proj" in name for name in state)
    assert all(
        any(f"self_attn.{name}.weight_quantizer" in key for key in state) for name in ("q_proj", "k_proj", "v_proj")
    )


def test_shared_qkv_weight_amax_matches_fusion():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    pattern = r"net\.language_model\.model\.layers\.(\d+)\.self_attn$"
    fused, shared = TinyFrameworkModel(), TinyFrameworkModel()
    with temporary_fuse_named_linears(fused, pattern, "qkv_proj", ["q_proj", "k_proj", "v_proj"]):
        mtq.quantize(fused, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda: fused(x))
    mtq.quantize(shared, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda: shared(x))

    assert share_qkv_weight_amax(shared) == 1

    for name in ("q_proj", "k_proj", "v_proj"):
        a = getattr(fused.net.language_model.model.layers[0].self_attn, name)
        b = getattr(shared.net.language_model.model.layers[0].self_attn, name)
        assert torch.equal(b.weight_quantizer.amax, a.weight_quantizer.amax), name
        assert torch.equal(b.input_quantizer.amax, a.input_quantizer.amax), name


def test_share_qkv_weight_amax_rejects_unequal_input_amax():
    model = TinyFrameworkModel(SkewedAttention)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda: model(x))
    with pytest.raises(ValueError, match="same input"):
        share_qkv_weight_amax(model)


def test_share_qkv_weight_amax_rejects_no_match():
    with pytest.raises(ValueError, match="no module matches"):
        share_qkv_weight_amax(TinyFrameworkModel(), parent_regex=r"missing$")
