# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The calibration file written after sharded calibration holds every calibrated amax."""

import copy

import modelopt.torch.quantization as mtq
import pytest
import torch
from torch import nn

from cosmos_framework.quantization.calibration_state import load_calibration_state, save_calibration_state


def test_saved_state_carries_calibrated_amax(tmp_path):
    """The streaming export reads each quantizer's ``_amax`` and the caller's record from the file."""
    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    mtq.quantize(model, copy.deepcopy(mtq.FP8_DEFAULT_CFG), lambda m: m(torch.randn(4, 16)))
    path = tmp_path / "calib.pth"
    save_calibration_state(model, path, quant_config={}, record={"step": "calibrate"})

    state = load_calibration_state(path)

    linear = model[0]
    assert torch.equal(state["quantizer_state"]["0.input_quantizer"]["_amax"], linear.input_quantizer.amax)
    assert torch.equal(state["quantizer_state"]["0.weight_quantizer"]["_amax"], linear.weight_quantizer.amax)
    assert state["record"] == {"step": "calibrate"}


def test_load_rejects_non_state(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"x": 1}, path)
    with pytest.raises(ValueError, match="missing"):
        load_calibration_state(path)
