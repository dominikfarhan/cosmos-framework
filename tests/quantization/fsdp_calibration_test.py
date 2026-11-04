# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""FSDP2-sharded calibration gives the unsharded amax on every rank.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/quantization/fsdp_calibration_test.py

Each rank shards a tiny model with FSDP2 (``fully_shard`` per block, then the root) before
``mtq.quantize`` and calibrates on inputs seeded by its rank. The reference is an unsharded copy
calibrated locally on every rank's inputs.
"""

import copy
import os

import modelopt.torch.quantization as mtq
import pytest
import torch
import torch.distributed as dist
from qkv_fusion_test import TinyFrameworkModel
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from cosmos_framework.model.generator.moe.interface import MoEExperts
from cosmos_framework.quantization.calibration_state import load_calibration_state, save_calibration_state
from cosmos_framework.quantization.model_utils import share_qkv_weight_amax
from cosmos_framework.quantization.moe import QuantPackedExperts, register_packed_experts

WORLD_SIZE = 2
NUM_EXPERTS = 4


@pytest.fixture(scope="module")
def fsdp_group():
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        pytest.skip(f"requires torchrun --nproc_per_node={WORLD_SIZE}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        yield dist.get_rank(), init_device_mesh("cuda", (WORLD_SIZE,))
    finally:
        dist.destroy_process_group()


class DenseModel(TinyFrameworkModel):
    """The q/k/v attention model plus a root Linear on its output."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.proj = nn.Linear(4, 4, bias=False)
        self.cuda()

    def blocks(self) -> list[nn.Module]:
        # FSDP2 gathers a unit's parameters in that unit's own forward pre-hook, and the forward
        # calls self_attn directly, so the attention modules are the units.
        return [layer.self_attn for layer in self.net.language_model.model.layers]

    def forward(self, x):
        q, k, v = super().forward(x)
        return self.proj(q + k + v)


class ExpertsBlock(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = nn.Linear(16, 16, bias=False)
        self.experts = MoEExperts(hidden_size=16, intermediate_size=8, num_experts=NUM_EXPERTS, top_k=2)
        for param in self.experts.parameters():
            param.data.normal_()

    def forward(self, hidden_states, *routing):
        self.linear(hidden_states)
        return self.experts(hidden_states, *routing)


class ExpertsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ExpertsBlock()])
        self.to(device="cuda", dtype=torch.bfloat16).requires_grad_(False).eval()

    def blocks(self) -> list[nn.Module]:
        return list(self.layers)

    def forward(self, *inputs):
        return self.layers[0](*inputs)


def _dense_inputs(rank: int) -> tuple:
    return (torch.randn(8, 4, generator=torch.Generator().manual_seed(rank)).cuda(),)


def _expert_inputs(expert_indices: torch.Tensor, seed: int) -> tuple:
    """``MoEExperts.forward`` arguments for ``expert_indices`` [num_tokens,2]; the last token is padding."""
    num_tokens = expert_indices.shape[0]
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(num_tokens, 16, generator=generator)
    topk_scores = torch.rand(num_tokens, 2, generator=generator)
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    token_mask[-1] = False
    num_tokens_per_expert = torch.bincount(expert_indices[token_mask].view(-1), minlength=NUM_EXPERTS)
    bf16 = torch.bfloat16
    return (
        hidden_states.to("cuda", bf16),
        topk_scores.to("cuda", bf16),
        expert_indices.cuda(),
        num_tokens_per_expert.cuda(),
        token_mask.cuda(),
    )


def _packed_inputs(rank: int) -> tuple:
    """Every expert receives tokens on every rank."""
    return _expert_inputs(torch.tensor([[0, 1], [1, 3], [0, 3], [3, 0], [1, 2], [2, 1]]), seed=rank)


def _skewed_inputs(rank: int) -> tuple:
    """Rank 0 routes no token to expert 0, not even the padding token; rank 1 does."""
    if rank == 0:
        expert_indices = torch.tensor([[1, 2], [2, 3], [3, 1], [1, 3], [2, 1], [3, 2]])
    else:
        expert_indices = torch.tensor([[0, 1], [0, 2], [3, 0], [0, 1], [2, 3], [1, 0]])
    return _expert_inputs(expert_indices, seed=rank)


def _calibrate(model: nn.Module, batches: list[tuple]) -> nn.Module:
    def forward_loop(m):
        with torch.no_grad():
            for inputs in batches:
                m(*inputs)

    return mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), forward_loop)


def _sharded(build, mesh) -> nn.Module:
    model = build()
    for block in model.blocks():
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    assert all(isinstance(p, DTensor) for p in model.parameters())
    return model


def _amax(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: buffer.cpu() for name, buffer in model.named_buffers() if name.endswith("._amax")}


def _assert_amax_equal(actual: dict, expected: dict, what: str) -> None:
    assert actual.keys() == expected.keys(), f"{what}: names differ {sorted(actual.keys() ^ expected.keys())}"
    for name, amax in expected.items():
        assert torch.equal(actual[name], amax), f"{what}: {name} {actual[name].flatten()} != {amax.flatten()}"


def _assert_matches_reference_on_all_ranks(sharded: nn.Module, reference: nn.Module) -> None:
    amax = _amax(sharded)
    _assert_amax_equal(amax, _amax(reference), "sharded vs reference")
    gathered = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, amax)
    _assert_amax_equal(gathered[1], gathered[0], "rank 1 vs rank 0")


def test_dense_and_shared_qkv_amax(fsdp_group):
    rank, mesh = fsdp_group
    sharded = _calibrate(_sharded(DenseModel, mesh), [_dense_inputs(rank)])
    reference = _calibrate(DenseModel(), [_dense_inputs(r) for r in range(WORLD_SIZE)])
    assert len(_amax(sharded)) == 8, sorted(_amax(sharded))  # input and weight amax of q/k/v and proj
    assert share_qkv_weight_amax(sharded) == share_qkv_weight_amax(reference) == 1
    _assert_matches_reference_on_all_ranks(sharded, reference)


def _check_experts(make_inputs, rank: int, mesh) -> None:
    register_packed_experts()
    sharded = _calibrate(_sharded(ExpertsModel, mesh), [make_inputs(rank)])
    assert isinstance(sharded.layers[0].experts, QuantPackedExperts)
    assert len(_amax(sharded)) == 4, sorted(_amax(sharded))  # linear input/weight + 2 expert inputs
    reference = _calibrate(ExpertsModel(), [make_inputs(r) for r in range(WORLD_SIZE)])
    _assert_matches_reference_on_all_ranks(sharded, reference)


def test_packed_experts(fsdp_group):
    _check_experts(_packed_inputs, *fsdp_group)


def test_expert_without_tokens_on_one_rank(fsdp_group):
    tokens_for_expert_0 = [_skewed_inputs(r)[3][0].item() for r in range(WORLD_SIZE)]
    assert tokens_for_expert_0[0] == 0 and tokens_for_expert_0[1] > 0, tokens_for_expert_0
    _check_experts(_skewed_inputs, *fsdp_group)


def test_only_rank_0_saves(fsdp_group, tmp_path):
    rank, mesh = fsdp_group
    register_packed_experts()
    sharded = _calibrate(_sharded(ExpertsModel, mesh), [_packed_inputs(rank)])
    paths = [tmp_path / "calib.pth"]
    dist.broadcast_object_list(paths, src=0)  # each rank has its own tmp_path
    path = paths[0]
    # Rank 1 saves first, alone: it must write nothing.
    if rank == 1:
        save_calibration_state(sharded, path, quant_config={}, record={})
    dist.barrier()
    assert not path.exists(), "rank 1 wrote the calibration file"
    dist.barrier()
    if rank == 0:
        save_calibration_state(sharded, path, quant_config={}, record={})
        quantizers = load_calibration_state(path)["quantizer_state"]
        saved = {f"{name}._amax": state["_amax"] for name, state in quantizers.items() if "_amax" in state}
        _assert_amax_equal(saved, _amax(sharded), "saved vs sharded")
    dist.barrier()
