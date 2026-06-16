from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMON_ARGS = [
    "--inversion-mode",
    "windowed",
    "--window-size",
    "3",
    "--window-step",
    "1",
    "--optimizer",
    "gauss_newton_cgls",
    "--data-misfit",
    "weighted_log_l2",
    "--spatial-regularization",
    "first_order_smoothness",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_script_path(project_root: Path) -> Path:
    return project_root / "examples" / "2_timelapsedERT_inversion_deepert.py"


def run_temporal_regularization_case(
    *,
    temporal_regularization_type: str,
    temporal_regularization: float,
    output_name: str,
    extra_cli_args: list[str] | None = None,
) -> int:
    project_root = _project_root()
    base_script = _base_script_path(project_root)
    output_dir = f"result/5_temporal_regularization/{output_name}"
    cli_tail = extra_cli_args if extra_cli_args is not None else sys.argv[1:]
    command = [
        sys.executable,
        str(base_script),
        "--project-root",
        str(project_root),
        "--forward-dir",
        "result/1_timelapsedERT_forward_deepert",
        "--true-model-dir",
        "resistivity_models_2d",
        "--temporal-regularization-type",
        str(temporal_regularization_type),
        "--temporal-regularization",
        str(float(temporal_regularization)),
        "--output-dir",
        output_dir,
        *COMMON_ARGS,
        *cli_tail,
    ]
    print("Running:", " ".join(command))
    return subprocess.call(command, cwd=project_root)
