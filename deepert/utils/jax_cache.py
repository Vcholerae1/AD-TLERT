"""JAX persistent compilation cache helpers."""

from __future__ import annotations

import os
from pathlib import Path

import jax

DEEPERT_JIT_CACHE_DIR_ENV = "DEEPERT_JIT_CACHE_DIR"


def normalize_jit_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """Resolve a Deepert JIT cache directory from an argument or environment."""

    if cache_dir is None:
        env_cache_dir = os.environ.get(DEEPERT_JIT_CACHE_DIR_ENV)
        if env_cache_dir is None or env_cache_dir.strip() == "":
            return None
        cache_dir = env_cache_dir
    return Path(cache_dir).expanduser().resolve()


def configure_jax_compilation_cache(cache_dir: str | Path | None) -> Path | None:
    """Enable JAX's persistent compilation cache for Deepert JIT kernels.

    JAX owns the executable cache key, including argument shapes, dtypes, static
    constants, backend, and JAX/XLA version inputs. Deepert only supplies the
    directory and lowers the default compile-time threshold so short but repeated
    kernels are persisted too.
    """

    resolved = normalize_jit_cache_dir(cache_dir)
    if resolved is None:
        return None

    resolved.mkdir(parents=True, exist_ok=True)
    current = jax.config.jax_compilation_cache_dir
    if current is not None and Path(current).expanduser().resolve() != resolved:
        raise ValueError(
            "JAX compilation cache directory is already configured as "
            f"{current!r}, cannot switch to {str(resolved)!r} in the same process"
        )

    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(resolved))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return resolved


__all__ = [
    "DEEPERT_JIT_CACHE_DIR_ENV",
    "configure_jax_compilation_cache",
    "normalize_jit_cache_dir",
]
