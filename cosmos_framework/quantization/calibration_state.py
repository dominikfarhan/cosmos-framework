# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The calibration file: the calibrated ModelOpt state, written once by rank 0 after sharded calibration."""

from __future__ import annotations

from pathlib import Path

import modelopt.torch.opt as mto
import torch
import torch.distributed as dist
from modelopt.torch.quantization.utils import get_quantizer_state_dict
from torch import nn

_STATE_KEYS = ("modelopt_state", "quantizer_state", "quant_config", "record")


def save_calibration_state(model: nn.Module, path: Path, *, quant_config: dict, record: dict) -> None:
    """Write the ModelOpt and quantizer state from rank 0 only, via a temporary file and a rename."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    path = Path(path)
    state = {
        "modelopt_state": mto.modelopt_state(model),
        "quantizer_state": get_quantizer_state_dict(model),
        "quant_config": quant_config,
        "record": record,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_calibration_state(path: Path) -> dict:
    # weights_only=False: modelopt_state holds ModelOpt config objects, not just tensors.
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [key for key in _STATE_KEYS if key not in state]
    if missing:
        raise ValueError(f"{path}: not a calibration state, missing {missing}")
    return state
