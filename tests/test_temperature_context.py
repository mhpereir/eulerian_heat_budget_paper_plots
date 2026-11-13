import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from PIL import Image

from src import cli, io, plot_results, plot_style
from src.specs import PlotRequest


def make_campaign(tmp_path, start="2021-06-01", periods=72):
    times = pd.date_range(start, periods=periods, freq="h")
    domain = {
        "region": "pnw_bartusek",
        "heat_budget_bottom_boundary": "surface",
        "heat_budget_top_boundary": "700hPa",
    }
    mean_attrs = {
        "units": "K",
        "zg_top_pressure_pa": 70000.0,
        "zg_bottom_mode": "surface_pressure",
        "source_variable": "T_domain_avg",
        "native_time_resolution": "hourly",
    }
    mean = 280 + np.sin(np.arange(periods) / 12)
    data_vars = {
        name: ("time", np.ones(periods)) for name in cli.RESULT_REQUIRED_VARIABLES
    }
    data_vars["domain_volume"] = ("time", 1e12 + np.arange(periods) * 1e8)
    data_vars["T_domain_avg"] = ("time", mean, mean_attrs)
    budget = xr.Dataset(data_vars, coords={"time": times})
    tas_attrs = {
        "units": "K",
        "native_time_resolution": "daily",
        "lat_bounds": [40, 60],
        "lon_bounds": [-130, -110],
    }
    days = (times.normalize() - times[0].normalize()).days
    stage1 = xr.Dataset(
        {
            "T_mean": ("time", mean, mean_attrs),
            "tas_region": ("time", 285.0 + days, tas_attrs),
        },
        coords={"time": times},
        attrs={
            **domain,
            "pipeline_stage": "stage_1_harmonized_regional_timeseries",
            "stage1_contract_version": 2,
        },
    )
    root = tmp_path / "inputs" / "case"
    root.mkdir(parents=True)
    stage1.to_netcdf(root / "stage1.nc", engine="h5netcdf")
    climo = xr.Dataset(
        {
            "T_mean": (
                "climatology_time",
                275 + np.arange(periods) * 0.01,
                {**mean_attrs, "climatology_statistic": "mean"},
            )
        },
        coords={"climatology_time": [t.replace(year=2000) for t in times]},
        attrs={
            **domain,
            "pipeline_stage": "stage_1_regional_hourly_climatology",
            "climatology_matching": "calendar month, day, and UTC hour",
            "climatology_method": "arithmetic mean over source years",
            "climatology_event_exclusion": "none",
            "climatology_start_year": 1940,
            "climatology_end_year": 2024,
            "source_stage1_sha256": io.file_provenance(root / "stage1.nc")["sha256"],
        },
    )
    climo.to_netcdf(root / "climo.nc", engine="h5netcdf")
    surface_clim = np.arange(365)[None, :] * 0.01 + np.array([280, 281])[:, None]
    reference_attrs = {"units": "same as input", "window_years": 31, "window_days": 31}
    threshold = xr.Dataset(
        {
            "climatology": (("year", "dayofyear"), surface_clim, reference_attrs),
            "threshold": (
                ("year", "dayofyear"),
                surface_clim + 3,
                {
                    **reference_attrs,
                    "percentile": 90,
                    "doy_definition": "1..365, Feb-29 dropped",
                },
            ),
        },
        coords={"year": [2020, 2021], "dayofyear": np.arange(1, 366)},
        attrs={
            "region": "pnw_bartusek",
            "variable": "tas",
            "dataset": "ERA5",
            "method": "evolving",
            "quantile": 90,
        },
    )
    threshold.to_netcdf(root / "threshold.nc", engine="h5netcdf")
    run_info = {
        "cli_args": {"region": "pnw_bartusek"},
        "request": {
            "bbox": [40, 60, -130, -110],
            "zg_top_pressure": 70000.0,
            "zg_bottom": "surface_pressure",
        },
        "git": {"branch": "test", "commit": "a" * 40, "dirty": False},
    }
    (root / "run_info.json").write_text(json.dumps(run_info))
    budget.to_netcdf(root / "heat_budget.nc", engine="h5netcdf")
    request = PlotRequest(
        dataset="case",
        data_root=root.parent,
        plots_root=tmp_path / "plots",
        diagnostic_plots=False,
        benchmark_diagnostics="off",
        stage1_file=root / "stage1.nc",
        temperature_climatology_file=root / "climo.nc",
        tas_threshold_file=root / "threshold.nc",
    )
    return request, budget, run_info


@pytest.fixture
def campaign(tmp_path):
    return make_campaign(tmp_path)


def rewrite(path, mutate):
    with xr.open_dataset(path, engine="h5netcdf") as source:
        ds = source.load()
    result = mutate(ds)
    (ds if result is None else result).to_netcdf(path, engine="h5netcdf")


def refresh_stage1_checksum(request):
    rewrite(
        request.temperature_climatology_file,
        lambda ds: ds.attrs.update(
            source_stage1_sha256=io.file_provenance(request.stage1_file)["sha256"]
        ),
    )


def test_context_aligns_existing_statistics_and_preserves_inputs(campaign):
    request, budget, info = campaign
    original = budget.copy(deep=True)
    context = io.load_temperature_context(request, budget, info)
    assert context.series.time.equals(budget.time)
    np.testing.assert_array_equal(
        context.series.tas.values, np.repeat([285, 286, 287], 24)
    )
    np.testing.assert_allclose(
        context.series.tas_climatology.values, np.repeat([282.51, 282.52, 282.53], 24)
    )
    np.testing.assert_allclose(
        context.series.tas_q90 - context.series.tas_climatology, 3
    )
    np.testing.assert_allclose(
        context.series.T_mean_climatology, 275 + np.arange(72) * 0.01
    )
    assert context.provenance["T2m_reference"]["selected_years"] == [2021]
    xr.testing.assert_identical(budget, original)


def test_leap_year_uses_no_leap_threshold_but_calendar_hour_climatology(tmp_path):
    request, budget, info = make_campaign(tmp_path, start="2020-06-01")
    context = io.load_temperature_context(request, budget, info)
    assert context.series.tas_climatology.values[0] == pytest.approx(281.51)
    assert context.series.T_mean_climatology.values[0] == 275


def test_february_29_is_rejected(tmp_path):
    request, budget, info = make_campaign(tmp_path, start="2020-02-28")
    with pytest.raises(ValueError, match="February 29"):
        io.load_temperature_context(request, budget, info)


@pytest.mark.parametrize(
    "name", ["stage1_file", "temperature_climatology_file", "tas_threshold_file"]
)
def test_partial_sources_rejected(name, tmp_path):
    with pytest.raises(ValueError, match="all three"):
        PlotRequest(**{name: tmp_path / "missing.nc"})


def test_cli_reports_partial_sources_without_traceback(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--stage1-file", "missing.nc"])
    error = capsys.readouterr().err
    assert "all three" in error and "Traceback" not in error


@pytest.mark.parametrize("name", [".", "..", "../outside", "/absolute"])
def test_reject_traversal(name):
    with pytest.raises(ValueError, match="single folder"):
        io.dataset_paths(name)


@pytest.mark.parametrize(
    "field,change,match",
    [
        ("tas_threshold_file", lambda ds: ds.attrs.update(region="pnw_hotz"), "region"),
        (
            "temperature_climatology_file",
            lambda ds: ds.attrs.update(heat_budget_top_boundary="300hPa"),
            "boundary",
        ),
        (
            "temperature_climatology_file",
            lambda ds: ds.T_mean.attrs.update(units="degC"),
            "Kelvin",
        ),
        (
            "temperature_climatology_file",
            lambda ds: ds.attrs.update(source_stage1_sha256="wrong"),
            "sha256",
        ),
        ("tas_threshold_file", lambda ds: ds.attrs.update(quantile=95), "q90"),
        (
            "tas_threshold_file",
            lambda ds: ds.threshold.attrs.update(doy_definition="dayofyear"),
            "doy_definition",
        ),
        ("tas_threshold_file", lambda ds: ds.isel(year=[0]), "coverage"),
        (
            "temperature_climatology_file",
            lambda ds: ds.isel(climatology_time=slice(1, None)),
            "coverage",
        ),
        (
            "temperature_climatology_file",
            lambda ds: ds.drop_vars("T_mean"),
            "requires T_mean",
        ),
        (
            "temperature_climatology_file",
            lambda ds: ds.attrs.__delitem__("climatology_start_year"),
            "baseline years",
        ),
        (
            "tas_threshold_file",
            lambda ds: ds.threshold.values.__setitem__((1, 151), np.nan),
            "finite",
        ),
    ],
)
def test_invalid_references_fail(campaign, field, change, match):
    request, budget, info = campaign
    rewrite(getattr(request, field), change)
    with pytest.raises(ValueError, match=match):
        io.load_temperature_context(request, budget, info)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda ds: ds.tas_region.attrs.update(lat_bounds=[41, 60]), "bounds"),
        (lambda ds: ds.tas_region.attrs.update(units="degC"), "Kelvin"),
        (lambda ds: ds.tas_region.values.__setitem__(1, 286), "constant"),
        (lambda ds: ds.tas_region.values.__setitem__(1, np.nan), "finite"),
        (lambda ds: ds.isel(time=slice(1, None)), "coverage"),
        (lambda ds: ds.isel(time=[0, 0, *range(1, ds.sizes["time"])]), "unique"),
    ],
)
def test_invalid_stage1_fails(campaign, change, match):
    request, budget, info = campaign
    rewrite(request.stage1_file, change)
    refresh_stage1_checksum(request)
    with pytest.raises(ValueError, match=match):
        io.load_temperature_context(request, budget, info)


def test_legacy_units_require_verified_kelvin_reference(campaign):
    request, budget, info = campaign
    budget.T_domain_avg.attrs.update(
        units="1", long_name="Fractional layer volume above surface pressure"
    )
    context = io.load_temperature_context(request, budget, info)
    assert context.provenance["budget_temperature_units"]["resolved"] == "K"
    assert budget.T_domain_avg.attrs["units"] == "1"
    budget.T_domain_avg.values[0] += 1e-7
    with pytest.raises(ValueError, match="unverified temperature units"):
        io.load_temperature_context(request, budget, info)


def test_unknown_budget_units_fail(campaign):
    request, budget, info = campaign
    budget.T_domain_avg.attrs["units"] = "degC"
    with pytest.raises(ValueError, match="unverified temperature units"):
        io.load_temperature_context(request, budget, info)


def test_missing_run_info_and_budget_coverage_fail(campaign):
    request, budget, info = campaign
    with pytest.raises(ValueError, match="domain metadata"):
        io.load_temperature_context(request, budget, {})
    with pytest.raises(ValueError, match="requested start"):
        io.load_temperature_context(
            replace(request, time_start="2021-05-31"), budget, info
        )
    with pytest.raises(ValueError, match="contiguous hourly"):
        io.load_temperature_context(request, budget.isel(time=[0, 2]), info)
    with pytest.raises(ValueError, match="T_domain_avg"):
        io.load_temperature_context(request, budget.drop_vars("T_domain_avg"), info)


def test_invalid_context_fails_before_writing_any_figures(campaign):
    request, _, _ = campaign
    rewrite(request.tas_threshold_file, lambda ds: ds.attrs.update(region="pnw_hotz"))
    with pytest.raises(ValueError, match="region"):
        cli.plot_dataset(request)
    assert not request.plots_root.exists()


def test_dirty_calculation_provenance_is_not_published(campaign):
    request, _, info = campaign
    info["git"]["dirty"] = True
    (request.data_root / request.dataset / "run_info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="clean EHB"):
        cli.plot_dataset(request)
    assert not request.plots_root.exists()


@pytest.mark.parametrize("window", [1, 24])
def test_hourly_panel_four_curves_single_band_and_no_volume(campaign, window):
    request, budget, info = campaign
    context = io.load_temperature_context(request, budget, info)
    fig, ax = plt.subplots()
    try:
        plot_results._add_domain_state_panel(
            ax, budget, smoothing_window=window, temperature_context=context
        )
        assert len(fig.axes) == 1
        assert len(ax.lines) == 4 and len(ax.collections) == 1
        assert ax.get_ylabel() == "Temperature [K]"
        assert ax.get_title(loc="left") == "Temperature"
        assert [line.get_linestyle() for line in ax.lines] == ["-", "--", "-", "--"]
        expected = budget.T_domain_avg.rolling(
            time=window, center=True, min_periods=1
        ).mean()
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected.values)
        expected_clim = context.series.T_mean_climatology.rolling(
            time=window, center=True, min_periods=1
        ).mean()
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), expected_clim.values)
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), [285, 286, 287, 287])
        assert ax.lines[2].get_drawstyle() == "steps-post"
        vertices = ax.collections[0].get_paths()[0].vertices[:, 1]
        assert vertices.min() == pytest.approx(282.51)
        assert vertices.max() == pytest.approx(285.53)
    finally:
        plt.close(fig)


def test_daily_panel_averages_volume_temperature_only(campaign):
    request, budget, info = campaign
    context = io.load_temperature_context(request, budget, info)
    fig, ax = plt.subplots()
    try:
        plot_results._add_domain_state_panel(
            ax, budget, daily=True, temperature_context=context
        )
        expected = budget.T_domain_avg.resample(time="1D").mean().values
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), np.r_[expected, expected[-1]]
        )
        np.testing.assert_array_equal(ax.lines[2].get_ydata(), [285, 286, 287, 287])
        assert len(ax.lines) == 4 and len(ax.collections) == 1
        assert all(line.get_drawstyle() == "steps-post" for line in ax.lines)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("daily", [False, True])
def test_single_column_layout_fonts_export_and_restoration(
    monkeypatch, tmp_path, daily
):
    request, budget, info = make_campaign(tmp_path, periods=92 * 24)
    for variable, amplitude in (
        ("dT_dt", 2),
        ("advection_term", 5),
        ("adiabatic_term", 5),
        ("diabatic_term", 2),
    ):
        budget[variable] = (
            budget.domain_volume
            * amplitude
            * np.sin(np.arange(budget.sizes["time"]) / 100)
            / 86400
        )
    context = io.load_temperature_context(request, budget, info)
    original = budget.copy(deep=True)
    original_font = plt.rcParams["font.size"]
    original_legend = plt.rcParams["legend.fontsize"]
    captured = []
    monkeypatch.setattr(
        plot_style,
        "save_figure",
        lambda fig, path, **kwargs: captured.append((fig, kwargs)),
    )
    if daily:
        plot_results.plot_budget_terms_day_bin(
            budget, str(tmp_path), temperature_context=context
        )
    else:
        plot_results.plot_budget_terms_hourly(
            budget, 24, str(tmp_path), temperature_context=context
        )
    fig, export_options = captured[0]
    assert export_options == {"crop": False}
    assert len(fig.axes) == 3
    assert fig.axes[1].get_title(loc="left") == "Temperature tendency"
    np.testing.assert_allclose(fig.get_size_inches(), [6, 10])
    assert fig.axes[0].get_position().height / fig.axes[
        1
    ].get_position().height == pytest.approx(1.6)
    assert not fig.texts
    assert fig.axes[0].lines[0].get_color() == "#E60000"
    assert fig.axes[1].lines[0].get_color() == "#D55E00"
    assert [line.get_color() for line in fig.axes[2].lines[:3]] == [
        "#2636CA", "#0DAF43", "#D55E00"
    ]
    assert plt.rcParams["font.size"] == original_font
    assert plt.rcParams["legend.fontsize"] == original_legend
    xr.testing.assert_identical(budget, original)

    # Export after leaving the context: existing artists must retain their fonts.
    monkeypatch.undo()
    image_path = tmp_path / "single.png"
    plot_style.save_figure(fig, image_path, crop=False)
    with Image.open(image_path) as image:
        assert image.size == (1800, 3000)
    fig.canvas.draw()
    for ax in fig.axes:
        assert ax.yaxis.label.get_fontsize() == 18
        assert ax._left_title.get_fontsize() == 18
        assert all(label.get_fontsize() == 18 for label in ax.get_yticklabels())
        assert all(text.get_fontsize() == 10.5 for text in ax.get_legend().get_texts())
    for ax in fig.axes[1:]:
        labels = [label.get_text() for label in ax.get_yticklabels()]
        if daily:
            assert all("." not in label for label in labels)
            ticks = ax.get_yticks()
            np.testing.assert_array_equal(ticks, np.round(ticks))
        else:
            assert all("." in label for label in labels)
    renderer = fig.canvas.get_renderer()
    for artist in [*(ax.get_legend() for ax in fig.axes), *fig.texts]:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    for ax in fig.axes:
        artists = [ax.xaxis.label, ax.yaxis.label, ax._left_title]
        artists.extend(ax.get_xticklabels() + ax.get_yticklabels())
        for artist in artists:
            if artist.get_visible() and artist.get_text():
                box = artist.get_window_extent(renderer)
                assert box.x0 >= 0 and box.y0 >= 0, artist.get_text()
                assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
        assert ax.get_legend().get_window_extent(renderer).y0 > ax.bbox.y1
        assert (
            ax._left_title.get_window_extent(renderer).y0
            > ax.get_legend().get_window_extent(renderer).y1
        )
        gap_pt = (
            ax._left_title.get_window_extent(renderer).y0
            - ax.get_legend().get_window_extent(renderer).y1
        ) * 72 / fig.dpi
        assert gap_pt <= 6
    for upper, lower in zip(fig.axes, fig.axes[1:]):
        assert lower._left_title.get_window_extent(renderer).y1 < upper.bbox.y0

    # A later diagnostic figure still uses the original shared theme.
    diagnostic, ax = plt.subplots()
    try:
        line, = ax.plot([0, 1], [0, 1], label="Diagnostic")
        ax.set_ylabel("Value")
        plot_style.inside_legend(ax, [line], [line.get_label()])
        plot_style.save_figure(diagnostic, tmp_path / "diagnostic.png")
        assert ax.yaxis.label.get_fontsize() == original_font
        assert ax.get_legend().get_texts()[0].get_fontsize() == original_legend
    finally:
        plt.close(diagnostic)


def test_budget_arithmetic_unchanged(campaign):
    _, budget, _ = campaign
    original = budget.copy(deep=True)
    for window in (1, 24):
        terms = plot_results._hourly_budget_terms(budget, window)
        for result, variable in (
            ("temperature_tendency", "dT_dt"),
            ("advection", "advection_term"),
            ("adiabatic", "adiabatic_term"),
            ("diabatic", "diabatic_term"),
        ):
            expected = (
                (budget[variable] * (1 / budget.domain_volume) * 3600)
                .rolling(time=window, center=True, min_periods=1)
                .mean()
            )
            xr.testing.assert_equal(terms[result], expected)
    daily = plot_results._daily_budget_terms(budget)
    xr.testing.assert_equal(
        daily["temperature_tendency"],
        (budget.dT_dt * (1 / budget.domain_volume) * 3600).resample(time="1D").sum(),
    )
    xr.testing.assert_identical(budget, original)


def test_end_to_end_inventory_manifest_and_no_overwrite(campaign):
    request, _, _ = campaign
    paths = cli.plot_dataset(request)
    files = list(paths.results_dir.glob("*.png"))
    assert len(files) == 3 and all(p.stat().st_size > 0 for p in files)
    assert not paths.diagnostics_dir.exists()
    manifest = json.loads((paths.results_dir.parent / "plot_manifest.json").read_text())
    assert manifest["selection"]["hourly_samples"] == 72
    assert manifest["layout"]["width_inches"] == 6
    assert manifest["layout"]["height_inches"] == 10
    assert manifest["layout"]["font_size_pt"] == 18
    assert manifest["layout"]["legend_font_size_pt"] == 10.5
    assert manifest["layout"]["supplemental_footer"] is False
    assert manifest["layout"]["crop"] is False
    assert len(manifest["outputs"]) == 3
    for output in manifest["outputs"]:
        assert io.file_provenance(output["path"])["sha256"] == output["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        cli.plot_dataset(request)


def test_plain_invocations_remain_supported(campaign):
    request, budget, _ = campaign
    request = replace(
        request,
        stage1_file=None,
        temperature_climatology_file=None,
        tas_threshold_file=None,
    )
    assert io.load_temperature_context(request, budget, {}) is None
    fig, ax = plt.subplots()
    try:
        plot_results._add_domain_state_panel(ax, budget, smoothing_window=1)
        assert len(fig.axes) == 1 and len(ax.lines) == 1 and not ax.collections
    finally:
        plt.close(fig)
