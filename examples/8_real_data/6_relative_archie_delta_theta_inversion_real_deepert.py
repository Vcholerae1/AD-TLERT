#!/usr/bin/env python3
"""Relative Archie delta-theta inversion for real data without sensor constraints.

This wrapper reuses the 7_* script but hard-clamps the soft sensor constraint
weight to zero, so the inversion runs without any sensor-based penalty.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_SOURCE_PATH = _SCRIPT_DIR / "7_relative_archie_delta_theta_inversion_real_deepert.py"
_SOURCE_MODULE_NAME = "_real_data_relative_archie_delta_theta_inversion_7"

for path in (str(_SCRIPT_DIR), str(_REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_source_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(_SOURCE_MODULE_NAME, _SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load source script at {_SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SOURCE = _load_source_module()
_ORIGINAL_BUILD_PARSER = _SOURCE.build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_PARSER()
    parser.set_defaults(
        output_dir="result/8_real_data/6_relative_archie_delta_theta_inversion_real_deepert",
        regularization=20.0,
        temporal_regularization=5.0,
        saturation_exponent=0.5,
        max_iterations=8,
        sensor_constraint_lambda=0.0,
        sensor_constraint_target_mode="delta_from_model_baseline",
        sensor_constraint_sigma=0.03,
        sensor_constraint_horizontal_radius=5.0,
        sensor_constraint_vertical_radius=0.3,
        sensor_constraint_min_cells=3,
    )

    original_parse_args = parser.parse_args

    def parse_args(args=None, namespace=None):
        parsed = original_parse_args(args=args, namespace=namespace)
        parsed.sensor_constraint_lambda = 0.0
        return parsed

    parser.parse_args = parse_args  # type: ignore[assignment]
    return parser


_SOURCE.build_parser = build_parser


def main(argv: list[str] | None = None) -> int:
    return _SOURCE.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
