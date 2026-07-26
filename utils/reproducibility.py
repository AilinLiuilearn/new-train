# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict

import numpy as np
import torch


_VALID_MODES = {"off", "balanced", "strict"}


def _set_tf32(enabled: bool) -> None:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enabled


def configure_reproducibility(seed: int, deterministic_mode: str = "strict") -> Dict[str, Any]:
    mode = str(deterministic_mode).strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported deterministic_mode={deterministic_mode!r}; expected one of {_VALID_MODES}")

    state = {
        "seed": int(seed),
        "deterministic_mode": mode,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    if mode == "off":
        return state

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    _set_tf32(False)

    if mode == "balanced":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True, warn_only=False)

    state.update(
        {
            "cudnn.deterministic": getattr(torch.backends.cudnn, "deterministic", None),
            "cudnn.benchmark": getattr(torch.backends.cudnn, "benchmark", None),
            "allow_tf32": {
                "cuda.matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None) if hasattr(torch.backends, "cuda") else None,
                "cudnn": getattr(torch.backends.cudnn, "allow_tf32", None) if hasattr(torch.backends, "cudnn") else None,
            },
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
    )
    return state


def describe_reproducibility_env() -> Dict[str, Any]:
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn.deterministic": getattr(torch.backends.cudnn, "deterministic", None),
        "cudnn.benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        "allow_tf32": {
            "cuda.matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None) if hasattr(torch.backends, "cuda") else None,
            "cudnn": getattr(torch.backends.cudnn, "allow_tf32", None) if hasattr(torch.backends, "cudnn") else None,
        },
        "torch.are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }
