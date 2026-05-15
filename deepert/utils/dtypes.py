"""Shared dtype definitions for JAX arrays."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

FLOAT_DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
NP_FLOAT_DTYPE = np.float64 if jax.config.jax_enable_x64 else np.float32
INT_DTYPE = jnp.int32
