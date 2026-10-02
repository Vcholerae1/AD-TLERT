"""Default floating-point precision (``ADTLERT_ENABLE_FLOAT64=1`` selects float64 at import)."""

from __future__ import annotations

import os

import numpy as np
import torch

if os.environ.get("ADTLERT_ENABLE_FLOAT64", "").strip().lower() not in {"", "0", "false", "no", "off"}:
    torch.set_default_dtype(torch.float64)

FLOAT_DTYPE = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
NP_FLOAT_DTYPE = np.float64 if FLOAT_DTYPE == torch.float64 else np.float32
INT_DTYPE = torch.int32
