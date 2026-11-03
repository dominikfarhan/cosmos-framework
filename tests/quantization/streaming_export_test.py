# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The streaming FP8 export writes a packed-MoE checkpoint from the BF16 source and a calibration file."""

import copy
import json
import re
from types import SimpleNamespace

import modelopt.torch.quantization as mtq
import pytest
import torch
from modelopt.torch.export.diffusers_utils import hide_quantizers_from_state_dict
from modelopt.torch.export.unified_export_hf import _process_quantized_modules
from safetensors.torch import load_file, save_file
from torch import nn

from cosmos_framework.quantization.export import Fp8DiffusersExporter, add_diffusion_step_policy
from cosmos_framework.quantization.streaming_export import (
    INDEX_NAME,
    QuantSpec,
    export_streaming,
    plan_quantization,
    quantize_dense,
    quantize_packed,
    write_transformer,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the FP8 arithmetic runs on CUDA")

NET = "net.language_model.model.layers.0."
TARGET = re.compile(r"net\.language_model\.model\.layers\.\d+\.(?:self_attn\.q_proj|mlp\.experts)")
SHAPES = {
    "layers.0.input_layernorm.weight": [16],
    "layers.0.self_attn.to_q.weight": [16, 16],
    "layers.0.mlp.gate.weight": [4, 16],
    "layers.0.mlp.experts.gate_up_proj": [4, 16, 16],
    "layers.0.mlp.experts.down_proj": [4, 8, 16],
}


@pytest.fixture
def checkpoint(tmp_path):
    """A one-layer BF16 Diffusers checkpoint in two shards, and its calibration state saved as ``calib``."""
    generator = torch.Generator().manual_seed(0)
    tensors = {key: torch.randn(shape, generator=generator).to(torch.bfloat16) for key, shape in SHAPES.items()}
    root = tmp_path / "source"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    keys = list(tensors)
    weight_map = {}
    for filename, shard_keys in (("model-1.safetensors", keys[:2]), ("model-2.safetensors", keys[2:])):
        save_file({key: tensors[key] for key in shard_keys}, transformer / filename)
        weight_map.update(dict.fromkeys(shard_keys, filename))
    (transformer / INDEX_NAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (transformer / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    (root / "model_index.json").write_text("{}")

    bf16 = torch.bfloat16
    quantizer_state = {
        f"{NET}self_attn.q_proj.input_quantizer": {"_amax": torch.tensor(2.0, dtype=bf16)},
        f"{NET}self_attn.q_proj.weight_quantizer": {"_amax": tensors["layers.0.self_attn.to_q.weight"].abs().amax()},
        f"{NET}self_attn.q_proj.output_quantizer": {},
        f"{NET}mlp.experts.gate_up_proj_input_quantizer": {"_amax": torch.tensor(3.0, dtype=bf16)},
        f"{NET}mlp.experts.down_proj_input_quantizer": {"_amax": torch.tensor(1.5, dtype=bf16)},
        f"{NET}mlp.gate.input_quantizer": {},
        f"{NET}mlp.gate.weight_quantizer": {},
    }
    state = {
        "modelopt_state": {},
        "quantizer_state": quantizer_state,
        "quant_config": copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        "record": {"step": "calibrate"},
    }
    calib = tmp_path / "calib.pth"
    torch.save(state, calib)
    return SimpleNamespace(root=root, tensors=tensors, state=state, calib=calib)


def _plan(checkpoint, expected_targets: int = 2) -> dict[str, QuantSpec]:
    return plan_quantization(checkpoint.state["quantizer_state"], checkpoint.tensors, TARGET, expected_targets)


def _dense_reference(weight: torch.Tensor, weight_amax: torch.Tensor, input_amax: torch.Tensor) -> dict:
    """``weight``, ``weight_scale`` and ``input_scale`` as ModelOpt's compress and export produce them."""
    model = nn.Sequential(nn.Linear(64, 32, bias=False)).to(device="cuda", dtype=torch.bfloat16)
    model[0].weight.data.copy_(weight)
    model = mtq.quantize(
        model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m(torch.randn(4, 64, device="cuda", dtype=torch.bfloat16))
    )
    model[0].weight_quantizer.amax = weight_amax.cuda()
    model[0].input_quantizer.amax = input_amax.cuda()
    mtq.compress(model)
    _process_quantized_modules(model, torch.bfloat16)
    with hide_quantizers_from_state_dict(model):
        state = model.state_dict()
    return {key: state[f"0.{key}"].cpu() for key in ("weight", "weight_scale", "input_scale")}


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)


@needs_cuda
@pytest.mark.parametrize("weight_amax", [3.0, 0.7], ids=["covering", "below_max_weight"])
def test_quantize_dense_matches_modelopt(weight_amax):
    weight = (torch.randn(32, 64, generator=torch.Generator().manual_seed(0)) * 0.8).to(torch.bfloat16)
    weight_amax, input_amax = torch.tensor(weight_amax, dtype=torch.bfloat16), torch.tensor(5.0, dtype=torch.bfloat16)

    expected = _dense_reference(weight, weight_amax, input_amax)
    actual = quantize_dense(weight, weight_amax, input_amax)

    for key in ("weight", "weight_scale", "input_scale"):
        assert actual[key].dtype == expected[key].dtype and actual[key].shape == expected[key].shape, key
        assert torch.equal(_bytes(actual[key]), _bytes(expected[key])), key
    assert torch.equal(actual["weight_quantizer._amax"], expected["weight_scale"] * 448)
    assert torch.equal(actual["input_quantizer._amax"], expected["input_scale"] * 448)


@needs_cuda
def test_quantize_packed_per_expert_scale():
    weight = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)

    out = quantize_packed(weight, torch.tensor(3.0, dtype=torch.bfloat16))

    weight_amax = weight.float().abs().amax(dim=(1, 2), keepdim=True)
    assert out[""].dtype == torch.float8_e4m3fn and out[""].shape == weight.shape
    torch.testing.assert_close(out[".weight_scale"], weight_amax / 448)
    torch.testing.assert_close(out[".input_scale"], torch.tensor(3.0 / 448))
    # Dequantization must recover each weight within FP8 rounding error: half the largest E4M3 step, 32 / 2.
    assert ((out[""].float() * out[".weight_scale"] - weight.float()).abs() <= 16 * out[".weight_scale"]).all()


@needs_cuda
def test_quantize_packed_zero_expert():
    weight = torch.randn(4, 8, 16, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    weight[0] = 0

    out = quantize_packed(weight, torch.tensor(3.0, dtype=torch.bfloat16))

    assert out[".weight_scale"][0].item() == 1.0
    assert (out[""][0].float() == 0).all()


def test_plan_maps_every_target_to_its_source_key(checkpoint):
    plan = _plan(checkpoint)

    quantizers = checkpoint.state["quantizer_state"]
    assert {key: spec.kind for key, spec in plan.items()} == {
        "layers.0.self_attn.to_q.weight": "dense",
        "layers.0.mlp.experts.gate_up_proj": "packed",
        "layers.0.mlp.experts.down_proj": "packed",
    }
    to_q = plan["layers.0.self_attn.to_q.weight"]
    assert to_q.weight_amax is quantizers[f"{NET}self_attn.q_proj.weight_quantizer"]["_amax"]
    assert to_q.input_amax is quantizers[f"{NET}self_attn.q_proj.input_quantizer"]["_amax"]
    assert (
        plan["layers.0.mlp.experts.down_proj"].input_amax
        is quantizers[f"{NET}mlp.experts.down_proj_input_quantizer"]["_amax"]
    )


def test_plan_rejects_target_without_amax(checkpoint):
    checkpoint.state["quantizer_state"][f"{NET}self_attn.q_proj.input_quantizer"] = {}
    checkpoint.state["quantizer_state"][f"{NET}self_attn.q_proj.weight_quantizer"] = {}
    with pytest.raises(ValueError, match=r"layers\.0\.self_attn\.q_proj"):
        _plan(checkpoint)


def test_plan_rejects_amax_outside_target(checkpoint):
    checkpoint.state["quantizer_state"][f"{NET}mlp.gate.weight_quantizer"] = {"_amax": torch.tensor(1.0)}
    with pytest.raises(ValueError, match=r"layers\.0\.mlp\.gate"):
        _plan(checkpoint)


def test_plan_rejects_wrong_target_count(checkpoint):
    with pytest.raises(ValueError, match=r"2 calibrated TARGET modules, expected 3"):
        _plan(checkpoint, expected_targets=3)


def test_plan_rejects_amax_without_source_key(checkpoint):
    layer1 = "net.language_model.model.layers.1.self_attn.q_proj"
    checkpoint.state["quantizer_state"][f"{layer1}.input_quantizer"] = {"_amax": torch.tensor(1.0)}
    with pytest.raises(ValueError, match=r"layers\.1\.self_attn\.q_proj"):
        _plan(checkpoint, expected_targets=3)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_plan_rejects_unusable_amax(checkpoint, bad):
    checkpoint.state["quantizer_state"][f"{NET}mlp.experts.down_proj_input_quantizer"]["_amax"] = torch.tensor(bad)
    with pytest.raises(ValueError, match=r"down_proj_input_quantizer"):
        _plan(checkpoint)


@needs_cuda
def test_write_transformer_matches_per_tensor_reference(checkpoint, tmp_path):
    plan = _plan(checkpoint)
    expected = {}
    for key, tensor in checkpoint.tensors.items():
        spec = plan.get(key)
        if spec is None:
            expected[key] = tensor
        elif spec.kind == "dense":
            for suffix, value in quantize_dense(tensor, spec.weight_amax, spec.input_amax).items():
                expected[f"{key.removesuffix('.weight')}.{suffix}"] = value
        else:
            for suffix, value in quantize_packed(tensor, spec.input_amax).items():
                expected[key + suffix] = value
    out_dir = tmp_path / "out"

    write_transformer(checkpoint.root / "transformer", plan, out_dir, quant_config=checkpoint.state["quant_config"])

    weight_map = json.loads((out_dir / INDEX_NAME).read_text())["weight_map"]
    actual = {}
    for filename in set(weight_map.values()):
        actual.update(load_file(out_dir / filename))
    assert weight_map.keys() == actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype and torch.equal(_bytes(actual[key]), _bytes(value)), key


def test_write_transformer_config(checkpoint, tmp_path):
    out_dir = tmp_path / "out"

    write_transformer(checkpoint.root / "transformer", {}, out_dir, quant_config=checkpoint.state["quant_config"])

    expected = {"num_hidden_layers": 1}
    expected["quantization_config"] = Fp8DiffusersExporter().build_quant_config_json(checkpoint.state["quant_config"])
    add_diffusion_step_policy(expected, 3)
    assert (out_dir / "config.json").read_text() == json.dumps(expected, indent=4)


@needs_cuda
def test_write_transformer_refuses_nan_fp8(checkpoint, tmp_path):
    plan = _plan(checkpoint)
    key = "layers.0.self_attn.to_q.weight"
    small_amax = (checkpoint.tensors[key].abs().max() / 4).to(torch.bfloat16)
    plan[key] = QuantSpec("dense", small_amax, plan[key].input_amax)

    with pytest.raises(ValueError, match=f"NaN.*{re.escape(key)}"):
        write_transformer(
            checkpoint.root / "transformer", plan, tmp_path / "out", quant_config=checkpoint.state["quant_config"]
        )


@needs_cuda
def test_export_streaming_assembles_output(checkpoint, tmp_path):
    output = tmp_path / "fp8"

    state = export_streaming(checkpoint.root, checkpoint.calib, output, target=TARGET, expected_targets=2)

    assert state["record"] == {"step": "calibrate"}
    assert load_file(output / "transformer" / "model-2.safetensors")["layers.0.mlp.experts.down_proj"].dtype == (
        torch.float8_e4m3fn
    )
    assert (output / "hf_quant_config.json").is_file() and (output / "model_index.json").is_symlink()
    assert not (output / "modelopt_state.pth").exists()
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".")) == []


def test_export_streaming_refuses_existing_output(checkpoint, tmp_path):
    output = tmp_path / "fp8"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_streaming(checkpoint.root, tmp_path / "missing.pth", output, target=TARGET, expected_targets=2)


@needs_cuda
def test_export_streaming_failure_leaves_no_output(checkpoint, tmp_path):
    key = f"{NET}self_attn.q_proj.weight_quantizer"
    checkpoint.state["quantizer_state"][key]["_amax"] = checkpoint.state["quantizer_state"][key]["_amax"] / 4
    torch.save(checkpoint.state, checkpoint.calib)
    output = tmp_path / "fp8"

    with pytest.raises(ValueError, match="NaN"):
        export_streaming(checkpoint.root, checkpoint.calib, output, target=TARGET, expected_targets=2)

    assert not output.exists()
    assert sorted(path.name for path in tmp_path.iterdir() if path.name.startswith(".")) == []
