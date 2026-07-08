"""I/O helpers for reduced heat-budget plotting datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

from . import config
from .specs import DatasetPaths


def _validate_dataset_name(dataset: str) -> None:
    if not dataset:
        raise ValueError("Dataset name must not be empty.")
    if Path(dataset).name != dataset:
        raise ValueError(f"Dataset name must be a single folder name: {dataset!r}")


def dataset_paths(
    dataset: str,
    *,
    data_root: Path | str = config.DATA_ROOT,
    plots_root: Path | str = config.PLOTS_ROOT,
) -> DatasetPaths:
    """Return input and output paths for one static plotting dataset."""
    _validate_dataset_name(dataset)

    data_dir = Path(data_root) / dataset
    output_dir = Path(plots_root) / dataset
    return DatasetPaths(
        data_dir=data_dir,
        heat_budget_path=data_dir / "heat_budget.nc",
        run_info_path=data_dir / "run_info.json",
        results_dir=output_dir / "results",
        diagnostics_dir=output_dir / "diagnostics",
    )


def available_datasets(data_root: Path | str = config.DATA_ROOT) -> list[str]:
    """List dataset folders that currently contain a reduced heat_budget.nc."""
    root = Path(data_root)
    if not root.exists():
        return []

    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "heat_budget.nc").is_file()
    )


def load_run_info(
    dataset: str,
    *,
    data_root: Path | str = config.DATA_ROOT,
) -> dict[str, Any]:
    """Load run metadata if present; return an empty dict for metadata-less data."""
    paths = dataset_paths(dataset, data_root=data_root)
    if not paths.run_info_path.is_file():
        return {}

    with paths.run_info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_heat_budget(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    data_root: Path | str = config.DATA_ROOT,
) -> xr.Dataset:
    """Open a reduced heat-budget NetCDF and optionally apply a time range."""
    paths = dataset_paths(dataset, data_root=data_root)
    if not paths.heat_budget_path.is_file():
        raise FileNotFoundError(
            f"Dataset {dataset!r} does not contain heat_budget.nc at "
            f"{paths.heat_budget_path}."
        )

    ds = xr.open_dataset(paths.heat_budget_path)
    if "time" not in ds.coords and "time" not in ds.dims:
        ds.close()
        raise ValueError(
            f"Dataset {dataset!r} must contain a 'time' coordinate or dimension."
        )

    if start is not None or end is not None:
        ds = ds.sel(time=slice(start, end))

    if ds.sizes.get("time", 0) == 0:
        ds.close()
        raise ValueError(
            f"Dataset {dataset!r} has no time samples in the requested range "
            f"start={start!r}, end={end!r}."
        )

    return ds
