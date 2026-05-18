from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_configure_jax_compilation_cache_persists_jit_entries(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        from pathlib import Path
        import sys

        import jax
        import jax.numpy as jnp

        from deepert.utils.jax_cache import configure_jax_compilation_cache

        cache_dir = Path(sys.argv[1])
        configured = configure_jax_compilation_cache(cache_dir)
        assert configured == cache_dir
        assert jax.config.jax_compilation_cache_dir == str(cache_dir)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 0.0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0

        @jax.jit
        def kernel(x):
            return (x + 1.0) * (x - 1.0)

        jax.block_until_ready(kernel(jnp.arange(8, dtype=jnp.float32)))
        assert any(cache_dir.iterdir()), "no JIT cache entries were written"
        """
    )
    env = os.environ.copy()
    env.pop("DEEPERT_JIT_CACHE_DIR", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
