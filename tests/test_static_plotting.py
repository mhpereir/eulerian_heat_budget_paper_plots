from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src import cli, io, plot_diagnostics


def _write_minimal_dataset(path):
    times = np.array(
        ["1941-05-01T00:00:00", "1941-05-02T00:00:00", "1941-05-03T00:00:00"],
        dtype="datetime64[ns]",
    )
    ds = xr.Dataset(
        {"domain_volume": ("time", np.array([1.0, 2.0, 3.0]))},
        coords={"time": times},
    )
    ds.to_netcdf(path)


def test_available_datasets_lists_folders_with_heat_budget(tmp_path):
    data_root = tmp_path / "data"
    ready = data_root / "ready"
    empty = data_root / "empty"
    ready.mkdir(parents=True)
    empty.mkdir()
    (ready / "heat_budget.nc").write_bytes(b"placeholder")

    assert io.available_datasets(data_root) == ["ready"]


def test_dataset_paths_constructs_input_and_output_paths(tmp_path):
    paths = io.dataset_paths(
        "sample_full_atmosphere",
        data_root=tmp_path / "data",
        plots_root=tmp_path / "plots",
    )

    assert paths.data_dir == tmp_path / "data" / "sample_full_atmosphere"
    assert paths.heat_budget_path == paths.data_dir / "heat_budget.nc"
    assert paths.run_info_path == paths.data_dir / "run_info.json"
    assert paths.results_dir == tmp_path / "plots" / "sample_full_atmosphere" / "results"
    assert (
        paths.diagnostics_dir
        == tmp_path / "plots" / "sample_full_atmosphere" / "diagnostics"
    )


def test_load_heat_budget_fails_when_netcdf_missing(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "missing_data").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="does not contain heat_budget.nc"):
        io.load_heat_budget("missing_data", data_root=data_root)


def test_load_heat_budget_applies_inclusive_date_slice(tmp_path):
    data_root = tmp_path / "data"
    dataset_dir = data_root / "ready"
    dataset_dir.mkdir(parents=True)
    _write_minimal_dataset(dataset_dir / "heat_budget.nc")

    ds = io.load_heat_budget(
        "ready",
        start="1941-05-02T00:00:00",
        end="1941-05-03T00:00:00",
        data_root=data_root,
    )
    try:
        assert ds.sizes["time"] == 2
        assert str(ds["time"].values[0]) == "1941-05-02T00:00:00.000000000"
    finally:
        ds.close()


def test_load_heat_budget_rejects_empty_date_slice(tmp_path):
    data_root = tmp_path / "data"
    dataset_dir = data_root / "ready"
    dataset_dir.mkdir(parents=True)
    _write_minimal_dataset(dataset_dir / "heat_budget.nc")

    with pytest.raises(ValueError, match="no time samples"):
        io.load_heat_budget(
            "ready",
            start="1950-01-01T00:00:00",
            end="1950-01-02T00:00:00",
            data_root=data_root,
        )


def test_benchmark_diagnostics_mode_resolution():
    assert cli.should_run_benchmark_diagnostics("case_full_atmosphere", "auto")
    assert not cli.should_run_benchmark_diagnostics("case_surface_700hPa", "auto")
    assert cli.should_run_benchmark_diagnostics("case_surface_700hPa", "on")
    assert not cli.should_run_benchmark_diagnostics("case_full_atmosphere", "off")


def test_figure5_bundle_requires_per_face_benchmark_fields(tmp_path):
    times = np.array(
        ["1941-05-01T00:00:00", "1941-05-02T00:00:00"],
        dtype="datetime64[ns]",
    )
    data_vars = {
        name: ("time", np.ones(times.size))
        for name in plot_diagnostics.FIG5_NET_VARIABLES
    }
    data_vars["T_domain_avg"] = ("time", np.ones(times.size) * 280.0)
    ds = xr.Dataset(data_vars, coords={"time": times})

    with pytest.raises(ValueError, match="benchmark_mass_flux_north"):
        plot_diagnostics.plot_figure5_diagnostics(ds, tmp_path)
