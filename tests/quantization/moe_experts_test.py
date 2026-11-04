# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""ModelOpt FP8 calibration of the packed grouped-mm MoE experts.

The CPU tests replace ``torch._grouped_mm`` and Triton's ``_generate_permute_indices`` with
PyTorch references.
"""

import copy

import modelopt.torch.quantization as mtq
import pytest
import torch
import torch.nn.functional as F
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts
from torch import nn

import cosmos_framework.model.generator.moe.native as native_grouped_mm
from cosmos_framework.model.generator.moe.interface import MoEExperts
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoeTextConfig
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.moe import (
    Qwen3VLMoeTextExpertsGroupedMm,
    Qwen3VLMoeTextExpertsNaive,
)
from cosmos_framework.quantization.moe import (
    QuantPackedExperts,
    assert_quantizers_calibrated,
    register_packed_experts,
)

QUANTIZERS = (
    "gate_up_proj_input_quantizer",
    "down_proj_input_quantizer",
)


def _reference_grouped_mm(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    # torch._grouped_mm leaves the rows past offs[-1] uninitialized; a large value stands in for that garbage.
    out = x.new_full((x.shape[0], w.shape[-1]), 1e4)
    start = 0
    for group, end in enumerate(offs.tolist()):
        out[start:end] = x[start:end] @ w[group]
        start = end
    return out


def _reference_permute_indices(
    num_tokens_per_expert: torch.Tensor, num_experts: int, max_len: int, alignment: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch version of the Triton ``_generate_permute_indices``: -1 marks a padding slot."""
    sizes = num_tokens_per_expert.clamp_min(alignment).to(torch.int32)
    sizes = (sizes + alignment - 1) // alignment * alignment  # [num_experts]
    permuted_indices = torch.full((max_len,), -1, dtype=torch.int32)
    start = write = 0
    for count, size in zip(num_tokens_per_expert.tolist(), sizes.tolist(), strict=True):
        permuted_indices[write : write + count] = torch.arange(start, start + count, dtype=torch.int32)
        start, write = start + count, write + size
    return permuted_indices, sizes


@pytest.fixture
def reference_routing(monkeypatch):
    monkeypatch.setattr(torch, "_grouped_mm", _reference_grouped_mm)
    monkeypatch.setattr(native_grouped_mm, "_generate_permute_indices", _reference_permute_indices)


def _config() -> Qwen3VLMoeTextConfig:
    return Qwen3VLMoeTextConfig(
        num_experts=4, hidden_size=16, moe_intermediate_size=8, num_experts_per_tok=2, hidden_act="silu"
    )


class Model(nn.Module):
    def __init__(self, experts: nn.Module | None = None):
        super().__init__()
        self.experts = Qwen3VLMoeTextExpertsGroupedMm(_config()) if experts is None else experts
        generator = torch.Generator().manual_seed(0)
        for param in self.experts.parameters():
            param.data.normal_(generator=generator)


# The packed expert types ``register_packed_experts`` claims, built for a given top_k.
EXPERTS = {
    "qwen_grouped_mm": lambda top_k: Qwen3VLMoeTextExpertsGroupedMm(_config(), top_k=top_k),
    "moe_experts": lambda top_k: MoEExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=top_k),
}


def _forward_inputs(
    seed: int = 3, dtype: torch.dtype = torch.float32, top_k: int = 2, routing: str = "padding"
) -> tuple:
    """``forward`` arguments for 6 tokens, top-``top_k`` of 4 experts, padded per ``routing``.

    Only the last token picks expert 2, so padding it leaves expert 2 without tokens.
    """
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(6, 16, generator=generator, dtype=dtype)
    topk_scores = torch.rand(6, top_k, generator=generator, dtype=dtype)
    expert_indices = torch.tensor([[0, 1, 3], [1, 3, 0], [0, 3, 1], [3, 0, 1], [1, 0, 3], [2, 1, 0]])
    expert_indices = expert_indices[:, :top_k].contiguous()
    token_mask = torch.tensor([True, True, True, True, True, False])
    if routing == "no_padding":
        token_mask = None
    elif routing == "all_padding":
        token_mask = torch.zeros(6, dtype=torch.bool)
    elif routing == "empty":
        hidden_states, topk_scores = hidden_states[:0], topk_scores[:0]
        expert_indices, token_mask = expert_indices[:0], token_mask[:0]
    else:
        assert routing == "padding", routing
    real = expert_indices if token_mask is None else expert_indices[token_mask]
    num_tokens_per_expert = torch.bincount(real.reshape(-1), minlength=4)
    return hidden_states, topk_scores, expert_indices, num_tokens_per_expert, token_mask


def _quantize(model: nn.Module, calibration: list[tuple] | None = None) -> nn.Module:
    register_packed_experts()

    def forward_loop(model: nn.Module) -> None:
        for inputs in calibration or [_forward_inputs()]:
            model.experts(*inputs)

    config = copy.deepcopy(mtq.FP8_DEFAULT_CFG)
    config["quant_cfg"].append({"quantizer_name": "*weight_quantizer", "enable": False})
    return mtq.quantize(model, config, forward_loop)


@pytest.fixture
def unregistered_experts():
    if QuantModuleRegistry.get(Qwen3VLMoeTextExpertsGroupedMm) is not None:
        QuantModuleRegistry.unregister(Qwen3VLMoeTextExpertsGroupedMm)
    yield
    register_packed_experts()


@pytest.mark.usefixtures("reference_routing", "unregistered_experts")
def test_registration_replaces_generic_fused_experts_wrapper():
    generic = Model()
    mtq.quantize(generic, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda model: None)
    assert isinstance(generic.experts, _QuantFusedExperts)

    model = _quantize(Model())

    assert isinstance(model.experts, QuantPackedExperts)
    assert issubclass(QuantModuleRegistry.get(Qwen3VLMoeTextExpertsGroupedMm), QuantPackedExperts)


@pytest.mark.usefixtures("reference_routing")
@pytest.mark.parametrize("routing", ["padding", "no_padding", "all_padding", "empty"])
@pytest.mark.parametrize("top_k", [2, 3])
@pytest.mark.parametrize("experts", sorted(EXPERTS))
def test_wrapper_matches_native_with_quantizers_disabled(experts, top_k, routing):
    # QuantPackedExperts has its own forward; this catches it drifting from the native one.
    model = Model(EXPERTS[experts](top_k))
    native = copy.deepcopy(model.experts)
    inputs = _forward_inputs(seed=4, top_k=top_k, routing=routing)
    expected = native(*inputs)
    model = _quantize(model, [_forward_inputs(top_k=top_k)])
    assert isinstance(model.experts, QuantPackedExperts)

    for name in QUANTIZERS:
        getattr(model.experts, name).disable()
    # On CPU ``MoEExperts`` runs the dense ``native.naive``, so the two match up to rounding only.
    torch.testing.assert_close(model.experts(*inputs), expected, rtol=1e-5, atol=1e-3)


@pytest.mark.usefixtures("reference_routing")
def test_each_input_quantizer_changes_the_wrapper_output():
    model = _quantize(Model())
    inputs = _forward_inputs(seed=4)
    for name in QUANTIZERS:
        getattr(model.experts, name).disable()
    unquantized = model.experts(*inputs)

    for name in QUANTIZERS:
        quantizer = getattr(model.experts, name)
        quantizer.enable()
        assert not torch.equal(model.experts(*inputs), unquantized), name
        quantizer.disable()


@pytest.mark.usefixtures("reference_routing")
def test_forward_calibrates_unpadded_activations():
    calibration = []
    for seed in (1, 2):
        hidden_states, _, expert_indices, counts, token_mask = _forward_inputs(seed)
        hidden_states[~token_mask] = 1e3  # A masked token must not reach a1.
        calibration.append((hidden_states, torch.full((6, 2), 0.25), expert_indices, counts, token_mask))

    model = _quantize(Model(), calibration)

    assert_quantizers_calibrated(model)
    experts = model.experts
    real = [(x[mask], indices[mask]) for x, _, indices, _, mask in calibration]
    assert torch.equal(experts.gate_up_proj_input_quantizer.amax, torch.stack([x.abs().max() for x, _ in real]).max())
    # a2 sees act(gate) * up on the real (token, expert) rows only, before the routing scores.
    gate_up = torch.cat([torch.einsum("th,tkhf->tkf", x, experts.gate_up_proj[indices]) for x, indices in real])
    gate, up = gate_up.chunk(2, dim=-1)
    expected = (F.silu(gate) * up).abs().max()
    amax = experts.down_proj_input_quantizer.amax
    torch.testing.assert_close(amax, expected)
    assert not torch.allclose(amax, 0.25 * expected)


def test_assert_quantizers_calibrated_rejects_uncalibrated_and_unwrapped():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m[0](torch.randn(2, 4)))
    with pytest.raises(ValueError, match=r"1\.input_quantizer"):
        assert_quantizers_calibrated(model)

    model = nn.Sequential(nn.Linear(4, 4))
    model.experts = Qwen3VLMoeTextExpertsNaive(_config())
    # ModelOpt calibrates the generic fused-experts weights, so they must not be uninitialized memory.
    generator = torch.Generator().manual_seed(0)
    for param in model.experts.parameters():
        param.data.normal_(generator=generator)
    mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m[0](torch.randn(2, 4)))
    with pytest.raises(ValueError, match="experts"):
        assert_quantizers_calibrated(model)


@pytest.mark.gpus(1)
def test_wrapper_matches_native_block_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch._grouped_mm and the Triton routing kernels")
    from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock

    torch.manual_seed(0)
    block = Qwen3VLMoeTextSparseMoeBlock(_config())
    for param in block.experts.parameters():
        param.data.normal_()
    block = block.to(device="cuda", dtype=torch.bfloat16).eval()
    native = copy.deepcopy(block)
    hidden_states = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16)
    token_mask = torch.ones(32, dtype=torch.bool, device="cuda")
    token_mask[-5:] = False

    register_packed_experts()
    mtq.quantize(block, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda model: model(hidden_states, token_mask=token_mask))
    assert isinstance(block.experts, QuantPackedExperts)
    mtq.disable_quantizer(block, "*")

    with torch.no_grad():
        expected, _ = native(hidden_states, token_mask=token_mask)
        actual, _ = block(hidden_states, token_mask=token_mask)
    assert torch.equal(actual, expected)
