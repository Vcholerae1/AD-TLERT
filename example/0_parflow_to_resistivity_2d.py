#!/usr/bin/env python3
"""Convert ParFlow saturation models to 2D resistivity slices.

This script follows the petrophysical setup described in:
Chen, H., & Niu, Q. (2022). Improving moisture content estimation from field
resistivity measurements with subsurface structure information.
Journal of Hydrology, 613(A), 128343. https://doi.org/10.1016/j.jhydrol.2022.128343

We apply a structural-unit specific conversion from saturation S to conductivity:
  sigma = sigma_sat_p * S^n + sigma_sat_s * S^(n-1)
and then resistivity rho = 1 / sigma.

When surface conduction is ignored (sigma_sat_s = 0), the equation reduces to
Archie's form: rho = rho_sat * S^(-n).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from parflow.tools.io import read_pfb
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Failed to import parflow.tools.io.read_pfb.\n"
        "Install example dependencies first, for example:\n"
        "  uv sync --extra examples"
    ) from exc


SATURATION_NAME_RE = re.compile(r"sc2d_6\.out\.satur\.(\d+)\.pfb$")


def _project_root() -> Path:
    """Locate repository root so defaults work when script lives in examples/."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() and (
            parent / "parflow_models"
        ).exists():
            return parent
    # Fallback: historical layout expects parflow_models next to cwd/script.
    return Path.cwd().resolve()


PROJECT_ROOT = _project_root()


@dataclass(frozen=True)
class UnitParams:
    """Petrophysical parameters for one structural unit."""

    name: str
    rho_sat: float
    n: float
    rho_sat_s: float | None = None
    porosity: float | None = None


@dataclass(frozen=True)
class UnitRange:
    """Optional random ranges for one structural unit."""

    rho_sat: tuple[float, float] | None = None
    rho_sat_s: tuple[float, float] | None = None
    n: tuple[float, float] | None = None
    porosity: tuple[float, float] | None = None


FIXED_PRESETS: dict[str, dict[int, UnitParams]] = {
    # Table 1 values (synthetic model) from Chen & Niu (2022).
    "table1_mean": {
        1: UnitParams(
            name="regolith", rho_sat=170.0, rho_sat_s=510.0, n=2.2, porosity=0.40
        ),
        2: UnitParams(name="fractured_bedrock", rho_sat=1100.0, n=1.8, porosity=0.18),
        3: UnitParams(name="fresh_bedrock", rho_sat=2400.0, n=2.5, porosity=0.05),
    },
    # Midpoints of Table 2 field ranges in Chen & Niu (2022).
    "table2_mid": {
        1: UnitParams(
            name="regolith", rho_sat=150.0, rho_sat_s=1800.0, n=1.75, porosity=0.375
        ),
        2: UnitParams(name="fractured_bedrock", rho_sat=257.5, n=2.1, porosity=0.25),
        3: UnitParams(name="fresh_bedrock", rho_sat=662.5, n=2.0, porosity=0.10),
    },
}


RANGE_PRESETS: dict[str, dict[int, UnitRange]] = {
    # Table 1 "Mean value and variation range" (NOT "Range used in MC").
    # Regolith and fractured bedrock use variation ranges; fresh bedrock keeps true values.
    "table1_range": {
        1: UnitRange(
            rho_sat=(100.0, 350.0), rho_sat_s=(400.0, 1400.0), porosity=(0.25, 0.5)
        ),
        2: UnitRange(rho_sat=(500.0, 1920.0), porosity=(0.11, 0.25)),
        3: UnitRange(),
    }
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "parflow_models"),
        help="Directory containing ParFlow PFB files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "resistivity_models_2d"),
        help="Output directory for .npy slices.",
    )
    parser.add_argument(
        "--class-file",
        default="class.pfb",
        help="Class indicator PFB file (relative to input-dir unless absolute path).",
    )
    parser.add_argument(
        "--porosity-file",
        default="sc2d_6.out.porosity.pfb",
        help=(
            "Porosity PFB file used to build phi2d models (relative to input-dir unless absolute path). "
            "This overrides any preset porosity values."
        ),
    )
    parser.add_argument(
        "--fixed-preset",
        default="table1_mean",
        choices=sorted(FIXED_PRESETS),
        help="Fixed petrophysical parameter preset.",
    )
    parser.add_argument(
        "--range-preset",
        default=None,
        choices=sorted(RANGE_PRESETS),
        help=(
            "Optional variation-range preset. When set, each timestep samples layer parameters "
            "from the provided ranges and then converts saturation to resistivity."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed used when --range-preset is enabled.",
    )
    parser.add_argument(
        "--y-index", type=int, default=2, help="Y-index of the 2D slice."
    )
    parser.add_argument(
        "--saturation-floor",
        type=float,
        default=1.0e-4,
        help="Lower bound used when clipping saturation to avoid division by zero.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing .npy outputs."
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Optional cap for quick tests."
    )
    parser.add_argument(
        "--save-petrophysical-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also save 2D petrophysical parameter models (rho_sat, rho_sat_s, n, phi, class) "
            "for later resistivity-to-moisture inversion."
        ),
    )
    parser.add_argument(
        "--petrophysical-output-dir",
        default=None,
        help=(
            "Directory for saved 2D petrophysical parameter models. "
            "Defaults to <input-dir>/petrophysical_models_2d."
        ),
    )
    return parser.parse_args(argv)


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _read_pfb_slice(path: Path, y_index: int) -> np.ndarray:
    values_3d = np.asarray(read_pfb(str(path)), dtype=float)
    if values_3d.ndim != 3:
        raise ValueError(f"Expected 3D PFB at {path}, got shape={values_3d.shape}")
    if y_index < 0 or y_index >= values_3d.shape[1]:
        raise ValueError(
            f"y-index {y_index} out of bounds for {path.name}: valid range [0, {values_3d.shape[1] - 1}]"
        )
    return np.asarray(values_3d[:, y_index, :], dtype=float)


def _discover_saturation_files(input_dir: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for path in sorted(input_dir.glob("sc2d_6.out.satur.*.pfb")):
        match = SATURATION_NAME_RE.match(path.name)
        if not match:
            continue
        pairs.append((int(match.group(1)), path))
    return pairs


def _validate_params(params: dict[int, UnitParams]) -> None:
    for class_id, unit in params.items():
        if class_id <= 0:
            raise ValueError(f"class ids must be positive integers, got {class_id}")
        if unit.rho_sat <= 0:
            raise ValueError(
                f"class {class_id} ({unit.name}) has non-positive rho_sat={unit.rho_sat}"
            )
        if unit.n <= 0:
            raise ValueError(
                f"class {class_id} ({unit.name}) has non-positive n={unit.n}"
            )
        if unit.rho_sat_s is not None and unit.rho_sat_s <= 0:
            raise ValueError(
                f"class {class_id} ({unit.name}) has non-positive rho_sat_s={unit.rho_sat_s}"
            )
        if unit.rho_sat_s is not None and unit.rho_sat_s <= unit.rho_sat:
            raise ValueError(
                f"class {class_id} ({unit.name}) needs rho_sat_s > rho_sat to keep sigma_sat_p positive "
                f"(got rho_sat_s={unit.rho_sat_s}, rho_sat={unit.rho_sat})"
            )


def _validate_ranges(ranges: dict[int, UnitRange]) -> None:
    for class_id, unit_range in ranges.items():
        for name, value in (
            ("rho_sat", unit_range.rho_sat),
            ("rho_sat_s", unit_range.rho_sat_s),
            ("n", unit_range.n),
            ("porosity", unit_range.porosity),
        ):
            if value is None:
                continue
            lo, hi = float(value[0]), float(value[1])
            if lo <= 0 or hi <= 0:
                raise ValueError(
                    f"class {class_id} has non-positive {name} range={value}"
                )
            if hi < lo:
                raise ValueError(
                    f"class {class_id} has invalid {name} range={value} (hi < lo)"
                )


def _sample_params(
    base: dict[int, UnitParams],
    ranges: dict[int, UnitRange],
    rng: np.random.Generator,
) -> dict[int, UnitParams]:
    sampled: dict[int, UnitParams] = {}
    for class_id, base_unit in base.items():
        unit_range = ranges.get(class_id, UnitRange())

        rho_sat = (
            float(rng.uniform(*unit_range.rho_sat))
            if unit_range.rho_sat is not None
            else base_unit.rho_sat
        )
        rho_sat_s = (
            float(rng.uniform(*unit_range.rho_sat_s))
            if unit_range.rho_sat_s is not None
            else base_unit.rho_sat_s
        )
        n = (
            float(rng.uniform(*unit_range.n))
            if unit_range.n is not None
            else base_unit.n
        )
        porosity = (
            float(rng.uniform(*unit_range.porosity))
            if unit_range.porosity is not None
            else base_unit.porosity
        )

        sampled[class_id] = UnitParams(
            name=base_unit.name,
            rho_sat=rho_sat,
            rho_sat_s=rho_sat_s,
            n=n,
            porosity=porosity,
        )
    return sampled


def saturation_to_resistivity(
    saturation: np.ndarray,
    class_ids: np.ndarray,
    params: dict[int, UnitParams],
    *,
    saturation_floor: float,
) -> np.ndarray:
    sat = np.asarray(saturation, dtype=float)
    classes = np.asarray(class_ids, dtype=np.int32)
    if sat.shape != classes.shape:
        raise ValueError(
            f"saturation shape {sat.shape} does not match class shape {classes.shape}"
        )
    sat = np.clip(sat, saturation_floor, 1.0)

    missing = sorted(
        int(value) for value in np.unique(classes) if int(value) not in params
    )
    if missing:
        raise ValueError(
            f"Class ids {missing} are present in class.pfb but missing in preset parameters {sorted(params)}"
        )

    rho = np.empty_like(sat, dtype=float)
    for class_id, unit in params.items():
        mask = classes == int(class_id)
        if not np.any(mask):
            continue
        sat_local = sat[mask]
        sigma_sat = 1.0 / unit.rho_sat
        if unit.rho_sat_s is None:
            sigma_local = sigma_sat * np.power(sat_local, unit.n)
        else:
            sigma_sat_s = 1.0 / unit.rho_sat_s
            sigma_sat_p = sigma_sat - sigma_sat_s
            sigma_local = sigma_sat_p * np.power(
                sat_local, unit.n
            ) + sigma_sat_s * np.power(sat_local, unit.n - 1.0)
        rho[mask] = 1.0 / sigma_local
    return rho


def _params_to_jsonable(
    params: dict[int, UnitParams],
) -> dict[str, dict[str, float | str | None]]:
    output: dict[str, dict[str, float | str | None]] = {}
    for class_id, unit in sorted(params.items()):
        output[str(class_id)] = {
            "name": unit.name,
            "rho_sat": unit.rho_sat,
            "rho_sat_s": unit.rho_sat_s,
            "n": unit.n,
            "porosity": unit.porosity,
        }
    return output


def _ranges_to_jsonable(
    ranges: dict[int, UnitRange],
) -> dict[str, dict[str, list[float] | None]]:
    output: dict[str, dict[str, list[float] | None]] = {}
    for class_id, unit in sorted(ranges.items()):
        output[str(class_id)] = {
            "rho_sat": None
            if unit.rho_sat is None
            else [float(unit.rho_sat[0]), float(unit.rho_sat[1])],
            "rho_sat_s": None
            if unit.rho_sat_s is None
            else [float(unit.rho_sat_s[0]), float(unit.rho_sat_s[1])],
            "n": None if unit.n is None else [float(unit.n[0]), float(unit.n[1])],
            "porosity": None
            if unit.porosity is None
            else [float(unit.porosity[0]), float(unit.porosity[1])],
        }
    return output


def _map_from_class_values(
    class_ids: np.ndarray, values_by_class: dict[int, float | None]
) -> np.ndarray:
    model = np.full(class_ids.shape, np.nan, dtype=float)
    for class_id, value in values_by_class.items():
        if value is None:
            continue
        model[class_ids == int(class_id)] = float(value)
    return model


def _save_petrophysical_models(
    *,
    output_dir: Path,
    class_ids: np.ndarray,
    porosity_2d: np.ndarray,
    porosity_source_file: Path,
    y_index: int,
    fixed_preset: str,
    base_params: dict[int, UnitParams],
    range_preset: str | None,
    range_params: dict[int, UnitRange] | None,
    input_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ytag = f"y{int(y_index)}"

    models: dict[str, np.ndarray] = {}
    models[f"class2d_{ytag}"] = np.asarray(class_ids, dtype=np.int32)
    phi_source = np.asarray(porosity_2d, dtype=float)
    if phi_source.shape != class_ids.shape:
        raise ValueError(
            f"porosity slice shape {phi_source.shape} does not match class slice shape {class_ids.shape}"
        )

    base_value_getters = {
        "rho_sat": lambda unit: unit.rho_sat,
        "rho_sat_s": lambda unit: unit.rho_sat_s,
        "n": lambda unit: unit.n,
    }
    for key, getter in base_value_getters.items():
        values = {class_id: getter(unit) for class_id, unit in base_params.items()}
        models[f"{key}2d_{ytag}_base_{fixed_preset}"] = _map_from_class_values(
            class_ids, values
        )
    models[f"phi2d_{ytag}_base_{fixed_preset}"] = phi_source

    if range_params is not None and range_preset is not None:
        range_value_getters = {
            "rho_sat": lambda unit: unit.rho_sat,
            "rho_sat_s": lambda unit: unit.rho_sat_s,
            "n": lambda unit: unit.n,
        }
        for key, getter in range_value_getters.items():
            lo_values: dict[int, float | None] = {}
            hi_values: dict[int, float | None] = {}
            for class_id in base_params:
                value = getter(range_params.get(class_id, UnitRange()))
                if value is None:
                    lo_values[class_id] = None
                    hi_values[class_id] = None
                else:
                    lo_values[class_id] = float(value[0])
                    hi_values[class_id] = float(value[1])
            models[f"{key}2d_{ytag}_range_{range_preset}_min"] = _map_from_class_values(
                class_ids, lo_values
            )
            models[f"{key}2d_{ytag}_range_{range_preset}_max"] = _map_from_class_values(
                class_ids, hi_values
            )

    files: dict[str, str] = {}
    for name, values in models.items():
        path = output_dir / f"{name}.npy"
        np.save(path, values)
        files[name] = str(path)

    npz_name = f"petrophysical_models2d_{ytag}_{fixed_preset}"
    if range_preset is not None:
        npz_name = f"{npz_name}_{range_preset}"
    npz_path = output_dir / f"{npz_name}.npz"
    np.savez_compressed(npz_path, **models)
    files["bundle_npz"] = str(npz_path)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "y_index": int(y_index),
        "shape": [int(class_ids.shape[0]), int(class_ids.shape[1])],
        "fixed_preset": fixed_preset,
        "range_preset": range_preset,
        "porosity_source_file": str(porosity_source_file),
        "porosity_source": "parflow_pfb_slice",
        "note": (
            "Variation ranges are from Table 1 mean value and variation range when range_preset=table1_range. "
            "Range used in MC is not used in this configuration. "
            "phi2d is always read from porosity_file, not from preset porosity values."
        ),
        "base_parameters": _params_to_jsonable(base_params),
        "variation_ranges": None
        if range_params is None
        else _ranges_to_jsonable(range_params),
        "files": files,
    }
    summary_name = f"petrophysical_models2d_{ytag}_{fixed_preset}"
    if range_preset is not None:
        summary_name = f"{summary_name}_{range_preset}"
    summary_path = output_dir / f"{summary_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    files["summary_json"] = str(summary_path)
    return files


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_params = FIXED_PRESETS[args.fixed_preset]
    _validate_params(base_params)
    range_params: dict[int, UnitRange] | None = None
    rng: np.random.Generator | None = None
    if args.range_preset is not None:
        range_params = RANGE_PRESETS[args.range_preset]
        _validate_ranges(range_params)
        rng = np.random.default_rng(args.seed)

    class_file = _resolve(args.class_file, input_dir)
    class_slice = _read_pfb_slice(class_file, args.y_index).astype(np.int32, copy=False)
    porosity_file = _resolve(args.porosity_file, input_dir)
    porosity_slice = _read_pfb_slice(porosity_file, args.y_index)
    if porosity_slice.shape != class_slice.shape:
        raise ValueError(
            f"porosity slice shape {porosity_slice.shape} does not match class slice shape {class_slice.shape}"
        )

    petrophysical_files: dict[str, str] | None = None
    petrophysical_output_dir: Path | None = None
    if args.save_petrophysical_models:
        petrophysical_output_dir = (
            Path(args.petrophysical_output_dir).resolve()
            if args.petrophysical_output_dir is not None
            else (input_dir / "petrophysical_models_2d").resolve()
        )
        petrophysical_files = _save_petrophysical_models(
            output_dir=petrophysical_output_dir,
            class_ids=class_slice,
            porosity_2d=porosity_slice,
            porosity_source_file=porosity_file,
            y_index=args.y_index,
            fixed_preset=args.fixed_preset,
            base_params=base_params,
            range_preset=args.range_preset,
            range_params=range_params,
            input_dir=input_dir,
        )

    pairs = _discover_saturation_files(input_dir)
    if args.max_files is not None:
        pairs = pairs[: max(int(args.max_files), 0)]
    if not pairs:
        raise SystemExit(
            f"No saturation files found in {input_dir} matching sc2d_6.out.satur.*.pfb"
        )

    written = 0
    skipped = 0
    rho_min = np.inf
    rho_max = -np.inf
    outputs: list[str] = []

    for step, sat_file in pairs:
        out_file = output_dir / f"resistivity2d_y{args.y_index}_t{step:05d}.npy"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue

        sat_slice = _read_pfb_slice(sat_file, args.y_index)
        if range_params is None:
            params = base_params
        else:
            assert rng is not None
            params = _sample_params(base_params, range_params, rng)
            _validate_params(params)
        rho_2d = saturation_to_resistivity(
            sat_slice,
            class_slice,
            params,
            saturation_floor=float(args.saturation_floor),
        )
        np.save(out_file, rho_2d)
        written += 1
        outputs.append(out_file.name)
        rho_min = min(rho_min, float(np.nanmin(rho_2d)))
        rho_max = max(rho_max, float(np.nanmax(rho_2d)))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "class_file": str(class_file),
        "porosity_file": str(porosity_file),
        "y_index": int(args.y_index),
        "fixed_preset": args.fixed_preset,
        "range_preset": args.range_preset,
        "seed": None if args.range_preset is None else int(args.seed),
        "saturation_floor": float(args.saturation_floor),
        "n_input_files": len(pairs),
        "n_written": int(written),
        "n_skipped_existing": int(skipped),
        "rho_min_written": None if written == 0 else float(rho_min),
        "rho_max_written": None if written == 0 else float(rho_max),
        "outputs_written": outputs,
        "base_parameters": _params_to_jsonable(base_params),
        "variation_ranges": None
        if range_params is None
        else _ranges_to_jsonable(range_params),
        "petrophysical_output_dir": None
        if petrophysical_output_dir is None
        else str(petrophysical_output_dir),
        "petrophysical_files": petrophysical_files,
    }
    summary_path = output_dir / "conversion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
