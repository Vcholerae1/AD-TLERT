#!/usr/bin/env python3
"""Compatibility entry point for example/single_time/inversion_adtlert.py."""

import runpy
from pathlib import Path

runpy.run_path(
    Path(__file__).resolve().parents[1] / "example/single_time/inversion_adtlert.py",
    run_name="__main__",
)
