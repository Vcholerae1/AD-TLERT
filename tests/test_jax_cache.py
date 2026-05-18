from __future__ import annotations

import os
from pathlib import Path

from deepert.utils.jax_cache import configure_jax_compilation_cache, configure_torch_jit_cache


def test_configure_jit_cache_resolves_and_creates_directory(tmp_path: Path) -> None:
    os.environ.pop("DEEPERT_JIT_CACHE_DIR", None)
    cache_dir = tmp_path / "torch-cache"

    assert configure_jax_compilation_cache(cache_dir) == cache_dir.resolve()
    assert cache_dir.is_dir()
    assert configure_torch_jit_cache(cache_dir) == cache_dir.resolve()
