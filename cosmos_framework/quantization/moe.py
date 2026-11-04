# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ModelOpt FP8 calibration of the packed grouped-mm MoE experts.

Calibrates and fake-quantizes the expert inputs through the wrapper's ``forward``. The FP8 expert
weights, with one scale per expert, are computed from the BF16 source by
:func:`~cosmos_framework.quantization.streaming_export.export_streaming`.
This code does not support weight-only.

Call :func:`register_packed_experts` before ``mtq.quantize``. Without it, ModelOpt's generic
``_QuantFusedExperts`` claims these experts and, since it only intercepts ``F.linear`` and they
run ``torch._grouped_mm``, quantizes nothing. :data:`MOE_FP8_TARGET` and
:func:`build_moe_quant_config` select the modules quantized in an MoE checkpoint.
"""

from __future__ import annotations

import copy
import re

import modelopt.torch.quantization as mtq
import torch
from modelopt.torch.quantization.nn import QuantModule, QuantModuleRegistry, TensorQuantizer
from torch import nn

from cosmos_framework.model.generator.moe.interface import MoEExperts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import Qwen3VLMoeTextExpertsGroupedMm

_PACKED_EXPERT_TYPES = (Qwen3VLMoeTextExpertsGroupedMm, MoEExperts)

# Framework module names quantized in an MoE checkpoint; everything else stays BF16.
MOE_FP8_TARGET = re.compile(
    r"net\.language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:[qkvo]_proj|[qkvo]_proj_moe_gen)"
    r"|mlp(?:_moe_gen)?\.experts"
    r"|mlp_moe_gen\.shared_expert\.(?:gate_up_proj|down_proj))"
)
_TARGETS_PER_LAYER = 10
_TARGETS_PER_SHARED_EXPERT = 2


def count_moe_fp8_targets(model: nn.Module) -> int:
    """How many modules MOE_FP8_TARGET should match, counted from the model's layers."""
    count = 0
    for i, layer in enumerate(model.net.language_model.model.layers):
        if not hasattr(layer.mlp_moe_gen, "experts"):
            raise ValueError(f"layer {i} has a dense MLP; only MoE layers are supported")
        count += _TARGETS_PER_LAYER + _TARGETS_PER_SHARED_EXPERT * (layer.mlp_moe_gen.shared_expert is not None)
    return count


class QuantPackedExperts(QuantModule):
    """Calibrates and fake-quantizes expert inputs; per-expert FP8 weights are quantized on export."""

    def _setup(self):
        self.gate_up_proj_input_quantizer = TensorQuantizer()
        self.down_proj_input_quantizer = TensorQuantizer()

    def forward(
        self,
        hidden_states: torch.Tensor,  # [num_tokens,hidden_size]
        topk_scores: torch.Tensor,  # [num_tokens,top_k]
        expert_indices: torch.Tensor,  # [num_tokens,top_k]
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        token_mask: torch.Tensor | None = None,  # [num_tokens]
    ) -> torch.Tensor:  # [num_tokens,hidden_size]
        # Group the real (token, slot) pairs by expert; ``num_tokens_per_expert`` counts exactly these.
        top_k = expert_indices.shape[1]
        slots = torch.arange(expert_indices.numel(), device=expert_indices.device)  # [num_tokens*top_k]
        if token_mask is not None:
            slots = slots[token_mask.repeat_interleave(top_k)]  # [num_routed]
        if slots.numel() == 0:
            return torch.zeros_like(hidden_states)  # [num_tokens,hidden_size]
        slots = slots[torch.argsort(expert_indices.reshape(-1)[slots], stable=True)]  # [num_routed]
        tokens = slots // top_k  # [num_routed]
        scores = topk_scores.reshape(-1)[slots].to(hidden_states.dtype)  # [num_routed]
        x = hidden_states[tokens]  # [num_routed,hidden_size]
        routed = self._quantized_experts(x, num_tokens_per_expert, scores)  # [num_routed,hidden_size]
        return torch.zeros_like(hidden_states).index_add_(0, tokens, routed)  # [num_tokens,hidden_size]

    def _quantized_experts(
        self,
        x: torch.Tensor,  # [num_routed,hidden_size]  (grouped by expert)
        num_tokens_per_expert: torch.Tensor,  # [num_experts]
        scores: torch.Tensor,  # [num_routed]
    ) -> torch.Tensor:  # [num_routed,hidden_size]
        # a2 observes ``act(gate) * up`` before the routing scores, because vLLM applies the scores
        # after the down GEMM. Per-expert weight scales are computed directly on export.
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)  # [num_experts]
        x = self.gate_up_proj_input_quantizer(x)
        h = torch._grouped_mm(x, self.gate_up_proj, offs=offsets)
        g, u = torch.chunk(h, chunks=2, dim=-1)  # 2x [num_routed,moe_intermediate_size]
        a = self.down_proj_input_quantizer(self.act_fn(g) * u)  # [num_routed,moe_intermediate_size]
        return torch._grouped_mm(a * scores.unsqueeze(-1), self.down_proj, offs=offsets)  # [num_routed,hidden_size]


def register_packed_experts() -> None:
    """Register :class:`QuantPackedExperts` before ``mtq.quantize`` so ``_QuantFusedExperts`` skips it."""
    pending: dict[type[nn.Module], str] = {}
    for expert_cls in _PACKED_EXPERT_TYPES:
        registered = QuantModuleRegistry.get(expert_cls)
        if registered is not None and issubclass(registered, QuantPackedExperts):
            continue
        if registered is not None:
            # An earlier ``mtq.quantize`` that saw this type unregistered has globally registered
            # ModelOpt's generic ``_QuantFusedExperts`` for it, which would quantize nothing here.
            QuantModuleRegistry.unregister(expert_cls)
        pending[expert_cls] = expert_cls.__name__
    if pending:
        QuantModuleRegistry.register(pending)(QuantPackedExperts)


def assert_quantizers_calibrated(model: nn.Module) -> None:
    """Fail export instead of writing a missing or zero scale for an enabled quantizer."""
    problems = []
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and not module.is_enabled:
            continue
        if isinstance(module, TensorQuantizer):
            amax = getattr(module, "amax", None)
            if amax is None:
                problems.append(f"{name}: no amax")
            elif not torch.isfinite(amax).all():
                problems.append(f"{name}: non-finite amax")
            elif name.endswith("input_quantizer") and not (amax > 0).all():
                problems.append(f"{name}: non-positive amax")
        elif (
            isinstance(getattr(module, "gate_up_proj", None), nn.Parameter)
            and module.gate_up_proj.ndim == 3
            and not isinstance(module, QuantPackedExperts)
        ):
            problems.append(f"{name}: packed experts {type(module).__name__} are not quantized")
    if problems:
        shown = "; ".join(problems[:10])
        raise ValueError(f"{len(problems)} quantizer problem(s), first {min(len(problems), 10)}: {shown}")


def build_moe_quant_config(model: nn.Module, target: re.Pattern = MOE_FP8_TARGET) -> dict:
    """``FP8_DEFAULT_CFG`` with every quantizer outside the ``target`` modules disabled."""
    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    targets = [name for name, _ in model.named_modules() if target.fullmatch(name)]
    for name, module in model.named_modules():
        if module not in QuantModuleRegistry or target.fullmatch(name):
            continue
        if any(t.startswith(name + ".") for t in targets):
            # An ancestor of a target: disable only its own quantizers, not the subtree.
            patterns = [f"{name}.{q}_quantizer" for q in ("input", "weight", "output")]
        else:
            patterns = [f"{name}.*"]
        config["quant_cfg"] += [{"quantizer_name": p, "enable": False} for p in patterns]
    return config


def assert_target_coverage(model: nn.Module, target: re.Pattern = MOE_FP8_TARGET) -> None:
    """The modules owning an enabled quantizer are exactly the ``target`` modules."""
    enabled = {
        name
        for name, module in model.named_modules()
        if any(isinstance(q, TensorQuantizer) and q.is_enabled for q in module.children())
    }
    targets = {name for name, _ in model.named_modules() if target.fullmatch(name)}
    if enabled != targets:
        extra, missing = sorted(enabled - targets), sorted(targets - enabled)
        raise ValueError(f"coverage mismatch: extra {extra[:10]}, missing {missing[:10]}")
