"""I/O helpers for reduced heat-budget plotting datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .specs import DatasetPaths, PlotRequest, TemperatureContext


def _validate_dataset_name(dataset: str) -> None:
    if not dataset:
        raise ValueError("Dataset name must not be empty.")
    if dataset in {".", ".."} or Path(dataset).name != dataset:
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
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise TypeError("run_info.json must contain a JSON object.")
    return metadata


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

    ds = xr.open_dataset(paths.heat_budget_path, engine="h5netcdf")
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


def file_provenance(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "path": str(path.absolute()),
        "resolved_path": str(path.resolve()),
        "sha256": checksum,
        "bytes": path.stat().st_size,
    }


def _require_attr(attrs, key, expected, context):
    if attrs.get(key) != expected:
        raise ValueError(
            f"{context}: {key} must be {expected!r}, got {attrs.get(key)!r}."
        )


def _time_index(ds, coordinate, context):
    if coordinate not in ds.coords or ds[coordinate].dims != (coordinate,):
        raise ValueError(f"{context}: missing one-dimensional {coordinate} coordinate.")
    if not np.issubdtype(ds[coordinate].dtype, np.datetime64):
        raise ValueError(
            f"{context}: {coordinate} must use Gregorian datetime64 values."
        )
    index = pd.DatetimeIndex(ds[coordinate].values)
    if (
        index.empty
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise ValueError(
            f"{context}: {coordinate} must be nonempty, unique and increasing."
        )
    if not index.equals(index.floor("h")):
        raise ValueError(f"{context}: timestamps must be whole UTC hours.")
    return index


def _temperature(ds, name, dims, context, *, input_units=None):
    if name not in ds or ds[name].dims != dims:
        raise ValueError(f"{context}: requires {name}{dims}.")
    units = ds[name].attrs.get("units")
    if units == "same as input" and input_units == "K":
        return ds[name]
    if str(units).lower() not in {"k", "kelvin", "kelvins"}:
        raise ValueError(f"{context}: {name} must have Kelvin units, got {units!r}.")
    return ds[name]


def _finite(values, context):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{context}: selected values must all be finite.")
    return values


def _positions(source, target, context):
    positions = source.get_indexer(target)
    if (positions < 0).any():
        missing = target[positions < 0][0]
        raise ValueError(f"{context}: missing coverage for {missing}.")
    return positions


def _validate_domain(budget, stage1, climatology, threshold, run_info):
    request = run_info.get("request", {})
    cli_args = run_info.get("cli_args", {})
    if not isinstance(request, dict) or not isinstance(cli_args, dict):
        raise TypeError("Invalid production run_info.json domain metadata.")
    region = cli_args.get("region")
    if not region or not {"bbox", "zg_top_pressure", "zg_bottom"} <= request.keys():
        raise ValueError(
            "Temperature context requires production run_info.json domain metadata."
        )
    for label, source in (
        ("Stage-1", stage1),
        ("Climatology", climatology),
        ("Threshold", threshold),
    ):
        _require_attr(source.attrs, "region", region, label)
    bounds = np.asarray(request["bbox"], dtype=float)
    tas_attrs = stage1["tas_region"].attrs
    tas_bounds = np.asarray(
        [*tas_attrs.get("lat_bounds", []), *tas_attrs.get("lon_bounds", [])]
    )
    if (
        bounds.shape != (4,)
        or tas_bounds.shape != (4,)
        or not np.array_equal(bounds, tas_bounds)
    ):
        raise ValueError("Stage-1 T2m regional bounds do not match the EHB request.")
    top = request["zg_top_pressure"]
    bottom_mode = request["zg_bottom"]
    bottom_label = "surface" if bottom_mode == "surface_pressure" else None
    if bottom_label is None:
        pressure = request.get("zg_bottom_pressure")
        if pressure is None:
            raise ValueError("Missing fixed-pressure bottom boundary in run_info.json.")
        bottom_label = f"{pressure / 100:g}hPa"
    for label, source in (("Stage-1", stage1), ("Climatology", climatology)):
        _require_attr(source.attrs, "heat_budget_bottom_boundary", bottom_label, label)
        _require_attr(
            source.attrs, "heat_budget_top_boundary", f"{top / 100:g}hPa", label
        )
    for label, mean in (
        ("Budget", budget["T_domain_avg"]),
        ("Stage-1", stage1["T_mean"]),
        ("Climatology", climatology["T_mean"]),
    ):
        _require_attr(mean.attrs, "zg_top_pressure_pa", top, label)
        _require_attr(mean.attrs, "zg_bottom_mode", bottom_mode, label)
        if bottom_mode != "surface_pressure":
            _require_attr(
                mean.attrs,
                "zg_bottom_pressure_pa",
                request["zg_bottom_pressure"],
                label,
            )
    return {
        "region": region,
        "bbox": bounds.tolist(),
        "top_pressure_pa": top,
        "bottom": bottom_label,
    }


def _budget_temperature_units(budget, stage1_values):
    mean = budget["T_domain_avg"]
    if str(mean.attrs.get("units")).lower() in {"k", "kelvin", "kelvins"}:
        return {
            "original": mean.attrs["units"],
            "resolved": "K",
            "method": "explicit units",
        }
    known_mask = (
        mean.attrs.get("units") == "1"
        and mean.attrs.get("long_name")
        == "Fractional layer volume above surface pressure"
    )
    values = _finite(mean.values, "Budget T_domain_avg")
    tolerance = (
        8
        * np.finfo(np.float64).eps
        * np.maximum(1, np.maximum(abs(values), abs(stage1_values)))
    )
    difference = abs(values - stage1_values)
    if not known_mask or not (difference <= tolerance).all():
        raise ValueError(
            "Budget T_domain_avg has unverified temperature units; cannot resolve against Stage-1."
        )
    return {
        "original": "1",
        "resolved": "K",
        "method": "known inherited volume-mask metadata; "
        "EHB integral(T dV)/V definition and Kelvin Stage-1 T_mean agreement within eight float64 epsilons",
        "max_abs_difference_K": float(difference.max()),
    }


def load_temperature_context(
    request: PlotRequest, budget: xr.Dataset, run_info: dict[str, Any]
) -> TemperatureContext | None:
    """Load existing references only; never estimate statistics or modify inputs."""
    if request.stage1_file is None:
        return None
    if "T_domain_avg" not in budget or budget.T_domain_avg.dims != ("time",):
        raise ValueError("Budget requires one-dimensional T_domain_avg(time).")
    times = _time_index(budget, "time", "Budget")
    if len(times) > 1 and not (np.diff(times.values) == np.timedelta64(1, "h")).all():
        raise ValueError(
            "Temperature context requires contiguous hourly budget coverage."
        )
    for label, value, actual in (
        ("start", request.time_start, times[0]),
        ("end", request.time_end, times[-1]),
    ):
        if value is not None:
            expected = pd.Timestamp(value)
            if label == "end" and len(value) == 10:
                expected += pd.Timedelta(hours=23)
            if expected != actual:
                raise ValueError(
                    f"Budget does not cover the requested {label} {value}."
                )
    files = {
        "stage1": request.stage1_file,
        "temperature_climatology": request.temperature_climatology_file,
        "tas_threshold": request.tas_threshold_file,
    }
    inputs = {name: file_provenance(path) for name, path in files.items()}
    with (
        xr.open_dataset(
            request.stage1_file, engine="h5netcdf", decode_timedelta=False
        ) as stage1,
        xr.open_dataset(
            request.temperature_climatology_file,
            engine="h5netcdf",
            decode_timedelta=False,
        ) as clim,
        xr.open_dataset(
            request.tas_threshold_file, engine="h5netcdf", decode_timedelta=False
        ) as thresh,
    ):
        _require_attr(
            stage1.attrs,
            "pipeline_stage",
            "stage_1_harmonized_regional_timeseries",
            "Stage-1",
        )
        _require_attr(stage1.attrs, "stage1_contract_version", 2, "Stage-1")
        _require_attr(
            clim.attrs,
            "pipeline_stage",
            "stage_1_regional_hourly_climatology",
            "Climatology",
        )
        _require_attr(
            clim.attrs,
            "climatology_matching",
            "calendar month, day, and UTC hour",
            "Climatology",
        )
        _require_attr(
            clim.attrs,
            "climatology_method",
            "arithmetic mean over source years",
            "Climatology",
        )
        _require_attr(clim.attrs, "climatology_event_exclusion", "none", "Climatology")
        _require_attr(
            clim.attrs,
            "source_stage1_sha256",
            inputs["stage1"]["sha256"],
            "Climatology",
        )
        for key, value in (
            ("method", "evolving"),
            ("dataset", "ERA5"),
            ("variable", "tas"),
        ):
            _require_attr(thresh.attrs, key, value, "Threshold")
        if float(thresh.attrs.get("quantile", np.nan)) != 90:
            raise ValueError("Threshold product must be q90.")
        tas = _temperature(stage1, "tas_region", ("time",), "Stage-1")
        mean = _temperature(stage1, "T_mean", ("time",), "Stage-1")
        t_clim = _temperature(clim, "T_mean", ("climatology_time",), "Climatology")
        tas_clim = _temperature(
            thresh, "climatology", ("year", "dayofyear"), "Threshold", input_units="K"
        )
        tas_q90 = _temperature(
            thresh, "threshold", ("year", "dayofyear"), "Threshold", input_units="K"
        )
        _require_attr(tas.attrs, "native_time_resolution", "daily", "Stage-1 T2m")
        for label, source in (("Stage-1 T_mean", mean), ("Climatology T_mean", t_clim)):
            _require_attr(source.attrs, "source_variable", "T_domain_avg", label)
            _require_attr(source.attrs, "native_time_resolution", "hourly", label)
        _require_attr(
            t_clim.attrs, "climatology_statistic", "mean", "Climatology T_mean"
        )
        _require_attr(
            tas_q90.attrs, "doy_definition", "1..365, Feb-29 dropped", "Threshold"
        )
        _require_attr(tas_q90.attrs, "percentile", 90, "Threshold")
        domain = _validate_domain(budget, stage1, clim, thresh, run_info)
        pos = _positions(_time_index(stage1, "time", "Stage-1"), times, "Stage-1")
        tas_values = _finite(tas.isel(time=pos).values, "T2m")
        mean_values = _finite(mean.isel(time=pos).values, "Stage-1 T_mean")
        _finite(budget["T_domain_avg"].values, "Budget T_domain_avg")
        units = _budget_temperature_units(budget, mean_values)
        daily = pd.Series(tas_values, index=times).groupby(times.normalize())
        if not np.array_equal(daily.min().values, daily.max().values):
            raise ValueError("Stage-1 T2m is not constant within each calendar day.")

        reference_times = _time_index(clim, "climatology_time", "Climatology")

        def calendar_keys(index):
            return pd.Index((index.month * 100 + index.day) * 100 + index.hour)

        reference_keys = calendar_keys(reference_times)
        if not reference_keys.is_unique:
            raise ValueError("Climatology has duplicate calendar-hour keys.")
        clim_pos = _positions(
            reference_keys, calendar_keys(times), "Hourly climatology"
        )
        mean_clim_values = _finite(
            t_clim.isel(climatology_time=clim_pos).values, "Hourly climatology"
        )

        if ((times.month == 2) & (times.day == 29)).any():
            raise ValueError(
                "February 29 is unsupported by the 365-day T2m threshold calendar."
            )
        if "dayofyear" not in thresh.coords or not np.array_equal(
            thresh.dayofyear.values, np.arange(1, 366)
        ):
            raise ValueError("Threshold dayofyear must be exactly 1..365.")
        if "year" not in thresh.coords:
            raise ValueError("Threshold requires a year coordinate.")
        years = pd.Index(thresh.year.values)
        if (
            not np.issubdtype(years.dtype, np.integer)
            or not years.is_unique
            or not years.is_monotonic_increasing
        ):
            raise ValueError("Threshold years must be unique increasing integers.")
        year_pos = _positions(years, pd.Index(times.year), "Year-specific threshold")
        doy = times.dayofyear - (times.is_leap_year & (times.month > 2)).astype(int)
        indexers = {
            "year": xr.DataArray(year_pos, dims="time"),
            "dayofyear": xr.DataArray(np.asarray(doy) - 1, dims="time"),
        }
        surface_mean = _finite(tas_clim.isel(indexers).values, "T2m climatology")
        surface_q90 = _finite(tas_q90.isel(indexers).values, "T2m q90")
        if not {"climatology_start_year", "climatology_end_year"} <= clim.attrs.keys():
            raise ValueError("Climatology is missing baseline years.")
        start_year = int(clim.attrs["climatology_start_year"])
        end_year = int(clim.attrs["climatology_end_year"])
        if end_year < start_year:
            raise ValueError("Climatology baseline years are reversed.")
        provenance = {
            "inputs": inputs,
            "domain": domain,
            "budget_temperature_units": units,
            "T_mean_climatology": {
                "start_year": start_year,
                "end_year": end_year,
                "attributes": dict(clim.attrs),
            },
            "T2m": {
                "attributes": dict(tas.attrs),
                "representation": "daily steps, no hourly smoothing",
            },
            "T2m_reference": {
                "selected_years": sorted(set(times.year.tolist())),
                "climatology_attributes": dict(tas_clim.attrs),
                "threshold_attributes": dict(tas_q90.attrs),
                "attributes": dict(thresh.attrs),
                "resolved_units": "K",
                "units_basis": "validated Kelvin Stage-1 tas_region",
                "calendar_mapping": "year and no-leap month/day; February 29 rejected",
            },
        }
    series = xr.Dataset(
        {
            name: ("time", values, {"units": "K"})
            for name, values in (
                ("tas", tas_values),
                ("T_mean_climatology", mean_clim_values),
                ("tas_climatology", surface_mean),
                ("tas_q90", surface_q90),
            )
        },
        coords={"time": budget.time.values},
    )
    return TemperatureContext(series=series, provenance=provenance)
