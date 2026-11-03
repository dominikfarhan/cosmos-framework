# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Streaming FP8 export of a checkpoint with packed MoE experts, without building the model.

Reads the BF16 source ``transformer/`` one safetensors shard at a time and quantizes the tensors
of the calibrated modules from the amax in a calibration file
(:func:`~cosmos_framework.quantization.calibration_state.save_calibration_state`). Host memory
holds one source shard; GPU memory one tensor. Dense linears match ModelOpt's ``mtq.compress`` +
export byte for byte; packed experts get one weight scale per expert.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.quantization.calibration_state import load_calibration_state
from cosmos_framework.quantization.checkpoint_io import resolve_checkpoint_path
from cosmos_framework.quantization.export import (
    FP8_MAX,
    Fp8DiffusersExporter,
    _build_net_to_diffusers_key_map,
    add_diffusion_step_policy,
    assemble_output_dir,
)

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
_PACKED_PARAMS = ("gate_up_proj", "down_proj")


def quantize_dense(
    weight: torch.Tensor, weight_amax: torch.Tensor, input_amax: torch.Tensor
) -> dict[str, torch.Tensor]:
    """FP8 weight, scales and quantizer buffers of one per-tensor linear, as ModelOpt exports them.

    Runs on CUDA: it divides by a Python float via the reciprocal, the CPU does not, so only CUDA
    reproduces GPU-made exports byte for byte.
    """
    if weight.dtype != torch.bfloat16 or weight_amax.dtype != torch.bfloat16:
        raise ValueError(f"weight and weight_amax must be bfloat16, got {weight.dtype} and {weight_amax.dtype}")
    # enforce CUDA to ensure reproducibility.
    # If we ran on CPU, we would still get a reasonable export but generally a different from the GPU-made one.
    weight, weight_amax = weight.cuda(), weight_amax.cuda()
    # As ModelOpt: divide in BF16 by the BF16 scale, without clamping (overflow becomes NaN).
    fp8 = (weight / (weight_amax / FP8_MAX)).to(torch.float8_e4m3fn)
    weight_scale = weight_amax.float() / FP8_MAX
    input_scale = input_amax.cuda().float() / FP8_MAX
    exported = {
        "weight": fp8,
        "weight_scale": weight_scale,
        "input_scale": input_scale,
        "weight_quantizer._amax": weight_scale * FP8_MAX,
        "weight_quantizer._scale": weight_scale,
        "input_quantizer._amax": input_scale * FP8_MAX,
    }
    return {key: value.contiguous().cpu() for key, value in exported.items()}


def quantize_packed(weight: torch.Tensor, input_amax: torch.Tensor) -> dict[str, torch.Tensor]:
    """FP8 packed experts ``[num_experts, in, out]`` with one weight scale per expert, keyed by key suffix."""
    weight = weight.cuda()
    weight_amax = weight.abs().amax(dim=(1, 2), keepdim=True).float()  # [num_experts,1,1]
    weight_scale = torch.where(weight_amax == 0, 1.0, weight_amax / FP8_MAX)
    fp8 = (weight.float() / weight_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    input_scale = (input_amax.cuda().float() / FP8_MAX).reshape(())
    exported = {"": fp8, ".weight_scale": weight_scale, ".input_scale": input_scale}
    return {key: value.contiguous().cpu() for key, value in exported.items()}


@dataclass(frozen=True)
class QuantSpec:
    """How to quantize one source tensor: its kind and the amax from the calibration file."""

    kind: Literal["dense", "packed"]
    weight_amax: torch.Tensor | None
    input_amax: torch.Tensor


def _amax(quantizer_state: dict, quantizer: str) -> torch.Tensor:
    amax = quantizer_state.get(quantizer, {}).get("_amax")
    if amax is None or not (torch.isfinite(amax).all() and (amax > 0).all()):
        raise ValueError(f"calibration state has no usable amax for {quantizer}: {amax}")
    return amax


def plan_quantization(
    quantizer_state: dict, source_keys: Iterable[str], target: re.Pattern, expected_targets: int
) -> dict[str, QuantSpec]:
    """Which Diffusers source tensors to quantize, keyed by source key, from key names alone.

    The calibrated modules (a quantizer with an ``_amax``) must be exactly the ``target`` modules,
    ``expected_targets`` of them, and each must own ``M.weight`` (dense) or ``M.gate_up_proj`` and
    ``M.down_proj`` (packed experts) in the source.
    """
    # Quantizer names are "M.<x>_quantizer" for dense and "M.<param>_<x>_quantizer" for packed.
    owner = {name: name.rsplit(".", 1)[0] for name in quantizer_state}
    calibrated = {owner[name] for name, state in quantizer_state.items() if "_amax" in state}
    targets = {module for module in owner.values() if target.fullmatch(module)}
    if calibrated != targets:
        raise ValueError(
            "calibrated modules differ from the target modules: "
            f"amax outside target {sorted(calibrated - targets)}, target without amax {sorted(targets - calibrated)}"
        )
    if len(calibrated) != expected_targets:
        raise ValueError(f"{len(calibrated)} calibrated TARGET modules, expected {expected_targets}")

    net_to_diffusers = _build_net_to_diffusers_key_map(dict.fromkeys(source_keys))
    plan, unmapped = {}, []
    for module in sorted(calibrated):
        net_module = module.removeprefix("net.")
        if f"{net_module}.weight" in net_to_diffusers:
            plan[net_to_diffusers[f"{net_module}.weight"]] = QuantSpec(
                "dense",
                _amax(quantizer_state, f"{module}.weight_quantizer"),
                _amax(quantizer_state, f"{module}.input_quantizer"),
            )
        elif all(f"{net_module}.{param}" in net_to_diffusers for param in _PACKED_PARAMS):
            for param in _PACKED_PARAMS:
                plan[net_to_diffusers[f"{net_module}.{param}"]] = QuantSpec(
                    "packed", None, _amax(quantizer_state, f"{module}.{param}_input_quantizer")
                )
        else:
            unmapped.append(module)
    if unmapped:
        raise ValueError(f"{len(unmapped)} calibrated modules have no source tensor: {unmapped[:10]}")
    return plan


def _quantize_shard(tensors: dict[str, torch.Tensor], plan: dict[str, QuantSpec]) -> dict[str, torch.Tensor]:
    """One source shard's output tensors: planned weights quantized with their scales, the rest as is."""
    out = {}
    for key, tensor in tensors.items():
        spec = plan.get(key)
        if spec is None:
            out[key] = tensor
        elif spec.kind == "dense":
            exported = quantize_dense(tensor, spec.weight_amax, spec.input_amax)
            # Packed experts are clamped; dense weights are not, so a low weight amax makes NaN.
            if ((exported["weight"].view(torch.uint8) & 0x7F) == 0x7F).any():
                raise ValueError(f"FP8 NaN in {key}: its weight amax is below max|weight|")
            stem = key.removesuffix(".weight")
            out.update({f"{stem}.{suffix}": value for suffix, value in exported.items()})
        else:
            out.update({key + suffix: value for suffix, value in quantize_packed(tensor, spec.input_amax).items()})
    return out


def write_transformer(
    source_transformer_dir: Path,
    plan: dict[str, QuantSpec],
    out_dir: Path,
    *,
    quant_config: dict,
    mixed_precision_steps: int = 3,
) -> None:
    """Write a new FP8 ``transformer/`` directory, one source shard in memory at a time.

    Each output shard keeps its source shard's file name and holds its tensors, with every quantized
    weight's scales next to it.
    """
    source_transformer_dir, out_dir = Path(source_transformer_dir), Path(out_dir)
    source_weight_map = json.loads((source_transformer_dir / INDEX_NAME).read_text())["weight_map"]
    config = json.loads((source_transformer_dir / "config.json").read_text())
    config["quantization_config"] = Fp8DiffusersExporter().build_quant_config_json(quant_config)
    add_diffusion_step_policy(config, mixed_precision_steps)

    out_dir.mkdir(parents=True)
    weight_map, total_size = {}, 0
    filenames = list(dict.fromkeys(source_weight_map.values()))
    start = time.monotonic()
    for i, filename in enumerate(filenames, 1):
        tensors = _quantize_shard(load_file(source_transformer_dir / filename), plan)
        save_file(tensors, out_dir / filename, metadata={"format": "pt"})
        print(f"[write] {i}/{len(filenames)} {filename} ({time.monotonic() - start:.0f} s)", flush=True)
        for key, tensor in tensors.items():
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()
        del tensors  # Free this shard before loading the next.
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (out_dir / INDEX_NAME).write_text(json.dumps(index, indent=2))
    (out_dir / "config.json").write_text(json.dumps(config, indent=4))


def export_streaming(
    model_name_or_path: str | Path,
    calibration_state: Path,
    output_dir: Path,
    *,
    target: re.Pattern,
    expected_targets: int,
    mixed_precision_steps: int = 3,
) -> dict:
    """Write the FP8 drop-in checkpoint to a new ``output_dir``; return the loaded calibration state.

    ``target`` matches the framework names of the modules to quantize. The transformer is built in
    ``.<output_dir>.transformer`` next to ``output_dir``; both are removed on failure.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; pass a new directory")
    state = load_calibration_state(calibration_state)
    input_dir = resolve_checkpoint_path(str(model_name_or_path))
    transformer_dir = input_dir / "transformer"
    source_keys = json.loads((transformer_dir / INDEX_NAME).read_text())["weight_map"]
    plan = plan_quantization(state["quantizer_state"], source_keys, target, expected_targets)
    print(f"[plan] {len(plan)} source tensors to FP8 from {expected_targets} target modules", flush=True)

    stage = output_dir.parent / f".{output_dir.name}.transformer"
    try:
        write_transformer(
            transformer_dir,
            plan,
            stage,
            quant_config=state["quant_config"],
            mixed_precision_steps=mixed_precision_steps,
        )
        assemble_output_dir(input_dir, output_dir, stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return state
