from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from adtlert.inversion import build_petrophysical_transform

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
    "--temporal-regularization-type",
    "temporal_smoothness",
    "--temporal-regularization",
    "10.0",
    "--no-plot",
    "--quiet",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _base_script_path(project_root: Path) -> Path:
    return project_root / "example" / "window_363" / "inversion_adtlert.py"


def _load_geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"x_nodes", "z_top", "layer_thickness"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required arrays: {sorted(missing)}")
        return {name: np.asarray(data[name], dtype=float).ravel() for name in required}


def _grid2d_to_cells(
    values_2d: np.ndarray,
    nodes: np.ndarray,
    cells: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> np.ndarray:
    """Map terrain-following ParFlow 2D grid values onto inversion cell centers."""

    grid = np.asarray(values_2d)
    x_nodes = np.asarray(geometry["x_nodes"], dtype=float).ravel()
    z_top = np.asarray(geometry["z_top"], dtype=float).ravel()
    layer_thickness = np.asarray(geometry["layer_thickness"], dtype=float).ravel()
    expected_shape = (layer_thickness.size, x_nodes.size - 1)
    if grid.shape != expected_shape:
        raise ValueError(
            f"2D grid shape {grid.shape} does not match expected {expected_shape}"
        )

    centers = np.asarray(nodes, dtype=float)[np.asarray(cells, dtype=np.int32)].mean(
        axis=1
    )
    x_center = centers[:, 0]
    z_center = centers[:, 1]

    column = np.searchsorted(x_nodes, x_center, side="right") - 1
    column = np.clip(column, 0, x_nodes.size - 2)
    surface_z = np.interp(x_center, x_nodes, z_top)
    depth = np.maximum(surface_z - z_center, 0.0)
    layer_top_to_bottom = np.searchsorted(
        np.cumsum(layer_thickness), depth, side="right"
    )
    layer_top_to_bottom = np.clip(layer_top_to_bottom, 0, layer_thickness.size - 1)

    grid_top_to_bottom = grid[::-1, :]
    return np.asarray(grid_top_to_bottom[layer_top_to_bottom, column], dtype=float)


def _load_petrophysical_parameters_on_mesh(
    *,
    project_root: Path,
    y_index: int,
    preset: str,
    nodes: np.ndarray,
    cells: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    parameter_dir = project_root / "parflow_models" / "petrophysical_models_2d"
    files = {
        "rho_sat": parameter_dir / f"rho_sat2d_y{y_index}_base_{preset}.npy",
        "rho_sat_s": parameter_dir / f"rho_sat_s2d_y{y_index}_base_{preset}.npy",
        "n": parameter_dir / f"n2d_y{y_index}_base_{preset}.npy",
        "phi": parameter_dir / f"phi2d_y{y_index}_base_{preset}.npy",
    }
    missing_required = [
        str(files[name]) for name in ("rho_sat", "n", "phi") if not files[name].exists()
    ]
    if missing_required:
        raise FileNotFoundError(
            f"Missing petrophysical parameter files: {missing_required}"
        )

    mapped: dict[str, np.ndarray] = {}
    for name, path in files.items():
        if path.exists():
            mapped[name] = _grid2d_to_cells(np.load(path), nodes, cells, geometry)
    return mapped


def convert_resistivity_result_to_water_content(
    *,
    output_dir: Path,
    project_root: Path,
    y_index: int = 2,
    preset: str = "table1_mean",
    saturation_floor: float = 1.0e-4,
) -> dict[str, object]:
    """Post-process a resistivity inversion result into saturation/theta models."""

    mesh_path = output_dir / "timelapse_inversion_mesh.npz"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing inversion mesh: {mesh_path}")
    final_models_path = output_dir / "final_models.npy"
    if not final_models_path.exists():
        raise FileNotFoundError(f"Missing resistivity result: {final_models_path}")

    with np.load(mesh_path) as mesh_data:
        nodes = np.asarray(mesh_data["nodes"], dtype=float)
        cells = np.asarray(mesh_data["cells"], dtype=np.int32)

    geometry_path = (
        project_root
        / "result"
        / "1_timelapsedERT_forward_adtlert"
        / "forward_geometry.npz"
    )
    geometry = _load_geometry(geometry_path)
    params = _load_petrophysical_parameters_on_mesh(
        project_root=project_root,
        y_index=y_index,
        preset=preset,
        nodes=nodes,
        cells=cells,
        geometry=geometry,
    )

    rho_models = np.asarray(np.load(final_models_path), dtype=float)
    if rho_models.ndim != 2:
        raise ValueError(
            f"Expected 2D final_models array, got shape {rho_models.shape}"
        )

    rho_safe = np.clip(rho_models, np.finfo(float).tiny, None)
    log_rho = np.log(rho_safe)

    transform = build_petrophysical_transform(
        "saturation",
        n_cells=rho_models.shape[0],
        model_transform="log",
        saturation_floor=float(saturation_floor),
        parameters={
            "rho_sat": params["rho_sat"],
            "rho_sat_s": params.get("rho_sat_s"),
            "n": params["n"],
        },
    )

    state = transform.state_from_log_resistivity(log_rho)
    saturation = np.asarray(transform.parameter_from_state(state), dtype=float)
    phi = np.asarray(params["phi"], dtype=float).reshape(-1, 1)
    water_content = saturation * phi

    np.save(output_dir / "final_saturation_from_resistivity_models.npy", saturation)
    np.save(
        output_dir / "final_water_content_from_resistivity_models.npy", water_content
    )

    steps_path = output_dir / "steps.npy"
    if steps_path.exists():
        steps = np.asarray(np.load(steps_path), dtype=int).ravel()
        if steps.size == water_content.shape[1]:
            for col, step in enumerate(steps):
                np.save(
                    output_dir
                    / f"inverted_water_content_from_resistivity_t{int(step):05d}.npy",
                    water_content[:, col],
                )

    summary = {
        "source_final_models": str(final_models_path),
        "output_saturation": str(
            output_dir / "final_saturation_from_resistivity_models.npy"
        ),
        "output_water_content": str(
            output_dir / "final_water_content_from_resistivity_models.npy"
        ),
        "y_index": int(y_index),
        "preset": str(preset),
        "saturation_floor": float(saturation_floor),
        "shape": [int(water_content.shape[0]), int(water_content.shape[1])],
        "water_content_min": float(np.nanmin(water_content)),
        "water_content_max": float(np.nanmax(water_content)),
    }
    (output_dir / "resistivity_to_water_content_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _run_inversion_case(
    *,
    output_name: str,
    extra_case_args: list[str] | None = None,
    extra_cli_args: list[str] | None = None,
) -> tuple[int, Path]:
    project_root = _project_root()
    base_script = _base_script_path(project_root)
    output_dir = project_root / "result" / "7_physical_parameter" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cli_tail = extra_cli_args if extra_cli_args is not None else sys.argv[1:]
    case_args = extra_case_args or []

    command = [
        sys.executable,
        str(base_script),
        "--project-root",
        str(project_root),
        "--forward-dir",
        "result/1_timelapsedERT_forward_adtlert",
        "--true-model-dir",
        "resistivity_models_2d",
        "--output-dir",
        f"result/7_physical_parameter/{output_name}",
        *COMMON_ARGS,
        *case_args,
        *cli_tail,
    ]
    print("Running:", " ".join(command))
    code = subprocess.call(command, cwd=project_root)
    return int(code), output_dir


def run_resistivity_then_convert_case(
    *, extra_cli_args: list[str] | None = None
) -> int:
    code, output_dir = _run_inversion_case(
        output_name="resistivity_then_convert",
        extra_case_args=[],
        extra_cli_args=extra_cli_args,
    )
    if code != 0:
        return int(code)

    summary = convert_resistivity_result_to_water_content(
        output_dir=output_dir,
        project_root=_project_root(),
        y_index=2,
        preset="table1_mean",
        saturation_floor=1.0e-4,
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_ad_saturation_case(*, extra_cli_args: list[str] | None = None) -> int:
    code, output_dir = _run_inversion_case(
        output_name="ad_saturation",
        extra_case_args=[
            "--petrophysical-transform",
            "saturation",
        ],
        extra_cli_args=extra_cli_args,
    )
    if code != 0:
        return int(code)

    # Safety fallback: if water-content file is absent, reconstruct from saturation and phi.
    wc_path = output_dir / "final_water_content_models.npy"
    sat_path = output_dir / "final_saturation_models.npy"
    if (not wc_path.exists()) and sat_path.exists():
        mesh_path = output_dir / "timelapse_inversion_mesh.npz"
        with np.load(mesh_path) as mesh_data:
            nodes = np.asarray(mesh_data["nodes"], dtype=float)
            cells = np.asarray(mesh_data["cells"], dtype=np.int32)
        geometry_path = (
            _project_root()
            / "result"
            / "1_timelapsedERT_forward_adtlert"
            / "forward_geometry.npz"
        )
        geometry = _load_geometry(geometry_path)
        params = _load_petrophysical_parameters_on_mesh(
            project_root=_project_root(),
            y_index=2,
            preset="table1_mean",
            nodes=nodes,
            cells=cells,
            geometry=geometry,
        )
        saturation = np.asarray(np.load(sat_path), dtype=float)
        phi = np.asarray(params["phi"], dtype=float).reshape(-1, 1)
        water_content = saturation * phi
        np.save(wc_path, water_content)

    if not wc_path.exists():
        raise FileNotFoundError(f"Expected AD water-content output missing: {wc_path}")

    return 0
