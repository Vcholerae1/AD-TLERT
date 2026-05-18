"""Shared dtype definitions for Torch arrays."""

from __future__ import annotations

import os
import numpy as np
import torch

from deepert.utils.torch_compat import torch_runtime, jnp


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() not in {"", "0", "false", "no", "off"}


if _env_truthy("DEEPERT_ENABLE_FLOAT64") or _env_truthy("JAX_ENABLE_X64"):
    torch_runtime.config.update("torch_enable_float64", True)

FLOAT_DTYPE = jnp.float64 if torch_runtime.config.torch_enable_float64 else jnp.float32
NP_FLOAT_DTYPE = np.float64 if torch_runtime.config.torch_enable_float64 else np.float32
INT_DTYPE = torch.int32
