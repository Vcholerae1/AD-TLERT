"""JIT cache path helpers kept for migration-time API compatibility."""

from __future__ import annotations

import os
from pathlib import Path

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
    """Resolve and create the optional JIT cache directory.

    Torch eager execution does not need the former JAX persistent compilation
    cache. Deepert still accepts ``jit_cache_dir`` so existing scripts keep
    working while the migration removes JAX-specific behavior.
    """

    resolved = normalize_jit_cache_dir(cache_dir)
    if resolved is None:
        return None

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


configure_torch_jit_cache = configure_jax_compilation_cache


__all__ = [
    "DEEPERT_JIT_CACHE_DIR_ENV",
    "configure_jax_compilation_cache",
    "configure_torch_jit_cache",
    "normalize_jit_cache_dir",
]
