from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


BASELINE_ENV = "DEEPERT_EXAMPLE_PARITY_BASELINE_ROOT"
CANDIDATE_ENV = "DEEPERT_EXAMPLE_PARITY_CANDIDATE_ROOT"

FORWARD_RTOL = float(os.environ.get("DEEPERT_EXAMPLE_PARITY_FORWARD_RTOL", "1.0e-3"))
FORWARD_ATOL = float(os.environ.get("DEEPERT_EXAMPLE_PARITY_FORWARD_ATOL", "1.0e-6"))
INVERSION_RTOL = float(os.environ.get("DEEPERT_EXAMPLE_PARITY_INVERSION_RTOL", "5.0e-2"))
INVERSION_ATOL = float(os.environ.get("DEEPERT_EXAMPLE_PARITY_INVERSION_ATOL", "1.0e-6"))


def _artifact_roots() -> tuple[Path, Path]:
    baseline = os.environ.get(BASELINE_ENV)
    candidate = os.environ.get(CANDIDATE_ENV)
    if not baseline or not candidate:
        pytest.skip(f"set {BASELINE_ENV} and {CANDIDATE_ENV} to run example parity checks")

    baseline_root = Path(baseline)
    candidate_root = Path(candidate)
    if not baseline_root.exists():
        pytest.fail(f"baseline artifact root does not exist: {baseline_root}")
    if not candidate_root.exists():
        pytest.fail(f"candidate artifact root does not exist: {candidate_root}")
    return baseline_root, candidate_root


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _assert_array_close(name: str, baseline: np.ndarray, candidate: np.ndarray, *, rtol: float, atol: float) -> None:
    baseline_array = np.asarray(baseline, dtype=float)
    candidate_array = np.asarray(candidate, dtype=float)
    if baseline_array.shape != candidate_array.shape:
        pytest.fail(f"{name}: shape differs, {baseline_array.shape} != {candidate_array.shape}")
    np.testing.assert_allclose(
        candidate_array,
        baseline_array,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=name,
    )


def _assert_npz_arrays_close(
    name: str,
    baseline_path: Path,
    candidate_path: Path,
    *,
    keys: tuple[str, ...],
    rtol: float,
    atol: float,
) -> None:
    if not candidate_path.exists():
        pytest.fail(f"{name}: missing candidate file {candidate_path}")
    with np.load(baseline_path) as baseline, np.load(candidate_path) as candidate:
        for key in keys:
            if key not in baseline.files:
                pytest.fail(f"{name}: missing baseline key {key!r} in {baseline_path}")
            if key not in candidate.files:
                pytest.fail(f"{name}: missing candidate key {key!r} in {candidate_path}")
            _assert_array_close(f"{name}:{key}", baseline[key], candidate[key], rtol=rtol, atol=atol)


def _assert_npy_close(
    name: str,
    baseline_dir: Path,
    candidate_dir: Path,
    file_name: str,
    *,
    rtol: float,
    atol: float,
) -> None:
    baseline_path = baseline_dir / file_name
    candidate_path = candidate_dir / file_name
    if not baseline_path.exists():
        pytest.fail(f"{name}: missing baseline file {baseline_path}")
    if not candidate_path.exists():
        pytest.fail(f"{name}: missing candidate file {candidate_path}")
    _assert_array_close(name, np.load(baseline_path), np.load(candidate_path), rtol=rtol, atol=atol)


def _assert_summary_scalars_close(
    name: str,
    baseline_path: Path,
    candidate_path: Path,
    keys: tuple[str, ...],
    *,
    rtol: float,
    atol: float,
) -> None:
    if not baseline_path.exists():
        pytest.fail(f"{name}: missing baseline summary {baseline_path}")
    if not candidate_path.exists():
        pytest.fail(f"{name}: missing candidate summary {candidate_path}")
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    for key in keys:
        if key not in baseline:
            pytest.fail(f"{name}: missing baseline summary key {key!r}")
        if key not in candidate:
            pytest.fail(f"{name}: missing candidate summary key {key!r}")
        _assert_array_close(
            f"{name}:{key}",
            np.asarray([baseline[key]], dtype=float),
            np.asarray([candidate[key]], dtype=float),
            rtol=rtol,
            atol=atol,
        )


def test_example_outputs_match_migration_baseline() -> None:
    baseline_root, candidate_root = _artifact_roots()

    single_forward_base = baseline_root / "deepert_single_forward_t04368"
    single_forward_candidate = candidate_root / "deepert_single_forward_t04368"
    _assert_npz_arrays_close(
        "1_forward_t04368_deepert.py",
        single_forward_base / "synthetic_ert_terrain_vardz_t04368.npz",
        single_forward_candidate / "synthetic_ert_terrain_vardz_t04368.npz",
        keys=("rhoa", "err", "a", "b", "m", "n", "elec_x", "elec_z"),
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )
    _assert_summary_scalars_close(
        "1_forward_t04368_deepert.py",
        single_forward_base / "forward_summary.json",
        single_forward_candidate / "forward_summary.json",
        keys=("rhoa_min", "rhoa_max", "mesh_cells", "measurements"),
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )

    series_forward_base = baseline_root / "timelapsedERT_forward"
    series_forward_candidate = candidate_root / "timelapsedERT_forward"
    baseline_series = sorted(series_forward_base.glob("synthetic_ert_terrain_vardz_t*.npz"))
    if not baseline_series:
        pytest.fail(f"1_forward_1year_deepert.py: no baseline forward npz files in {series_forward_base}")
    for baseline_path in baseline_series:
        _assert_npz_arrays_close(
            f"1_forward_1year_deepert.py:{baseline_path.name}",
            baseline_path,
            series_forward_candidate / baseline_path.name,
            keys=("rhoa", "err", "a", "b", "m", "n", "elec_x", "elec_z"),
            rtol=FORWARD_RTOL,
            atol=FORWARD_ATOL,
        )
    _assert_summary_scalars_close(
        "1_forward_1year_deepert.py",
        series_forward_base / "forward_summary.json",
        series_forward_candidate / "forward_summary.json",
        keys=("n_selected", "n_ok", "n_failed", "first_step", "last_step", "mesh_cells", "measurements"),
        rtol=0.0,
        atol=0.0,
    )

    single_inversion_base = baseline_root / "deepert_single_inversion_t04368"
    single_inversion_candidate = candidate_root / "deepert_single_inversion_t04368"
    for file_name in ("final_model.npy", "predicted_rhoa.npy", "coverage.npy", "chi2_history.npy"):
        _assert_npy_close(
            f"2_single_inversion_t04368_deepert.py:{file_name}",
            single_inversion_base,
            single_inversion_candidate,
            file_name,
            rtol=INVERSION_RTOL,
            atol=INVERSION_ATOL,
        )
    _assert_summary_scalars_close(
        "2_single_inversion_t04368_deepert.py",
        single_inversion_base / "inversion_summary.json",
        single_inversion_candidate / "inversion_summary.json",
        keys=("final_chi2", "iterations", "mesh_cells", "measurements"),
        rtol=INVERSION_RTOL,
        atol=INVERSION_ATOL,
    )

    timelapse_inversion_base = baseline_root / "timelapsedERT_inversion"
    timelapse_inversion_candidate = candidate_root / "timelapsedERT_inversion"
    for file_name in ("final_models.npy", "predicted_rhoa.npy", "coverage.npy", "chi2_all.npy", "steps.npy"):
        _assert_npy_close(
            f"2_timelapsedERT_inversion_deepert.py:{file_name}",
            timelapse_inversion_base,
            timelapse_inversion_candidate,
            file_name,
            rtol=INVERSION_RTOL,
            atol=INVERSION_ATOL,
        )
    _assert_summary_scalars_close(
        "2_timelapsedERT_inversion_deepert.py",
        timelapse_inversion_base / "timelapsed_inversion_summary.json",
        timelapse_inversion_candidate / "timelapsed_inversion_summary.json",
        keys=("n_timesteps", "n_cells", "base_forward_mesh_cells", "forward_mesh_cells"),
        rtol=0.0,
        atol=0.0,
    )
