#!/usr/bin/env python3
"""Compatibility entry point for example/window_363/inversion_pygimli.py."""

from pathlib import Path
import runpy

runpy.run_path(
    Path(__file__).resolve().parents[1] / "example/window_363/inversion_pygimli.py",
    run_name="__main__",
)
