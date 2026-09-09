"""Publication result plots for reduced heat-budget datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from . import plot_style as style
from .specs import TemperatureContext

RESULT_WIDTH = "full"
RESULT_ASPECT = style.RESULT_STACK_ASPECT
HOURLY_RATE_SCALE = 3600


def _rolling_mean(da: xr.DataArray, window: int) -> xr.DataArray:
    return da.rolling(time=window, center=True, min_periods=1).mean()


def _plot_series(
    ax,
    da: xr.DataArray,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    drawstyle: str | None = None,
    alpha: float = 1.0,
    linewidth: float = style.LINE_WIDTH_PT,
):
    (line,) = ax.plot(
        da["time"].values,
        da.values,
        label=label,
        color=color,
        linestyle=linestyle,
        drawstyle=drawstyle,
        alpha=alpha,
        linewidth=linewidth,
        solid_capstyle="round",
    )
    return line


def _format_result_axes(axes) -> None:
    for ax in axes:
        style.style_axis(ax)
    style.format_time_axis(axes[-1])
    for ax in axes[:-1]:
        ax.set_xlabel("")
    axes[-1].set_xlabel("Time")


def _daily_budget_terms(ds_budget: xr.Dataset) -> dict[str, xr.DataArray]:
    norm_factor = 1 / ds_budget["domain_volume"]
    return {
        "storage": (ds_budget["d_dt_T"] * norm_factor * HOURLY_RATE_SCALE)
        .resample(time="1D")
        .sum(),
        "volume_tendency": (
            ds_budget["T_domain_avg"]
            * norm_factor
            * ds_budget["dV_dt"]
            * HOURLY_RATE_SCALE
        )
        .resample(time="1D")
        .sum(),
        "temperature_tendency": (ds_budget["dT_dt"] * norm_factor * HOURLY_RATE_SCALE)
        .resample(time="1D")
        .sum(),
        "advection": (ds_budget["advection_term"] * norm_factor * HOURLY_RATE_SCALE)
        .resample(time="1D")
        .sum(),
        "adiabatic": (ds_budget["adiabatic_term"] * norm_factor * HOURLY_RATE_SCALE)
        .resample(time="1D")
        .sum(),
        "diabatic": (ds_budget["diabatic_term"] * norm_factor * HOURLY_RATE_SCALE)
        .resample(time="1D")
        .sum(),
    }


def _hourly_budget_terms(
    ds_budget: xr.Dataset,
    smoothing_window: int,
) -> dict[str, xr.DataArray]:
    norm_factor = 1 / ds_budget["domain_volume"]
    terms = {
        "storage": ds_budget["d_dt_T"] * norm_factor * HOURLY_RATE_SCALE,
        "volume_tendency": (
            ds_budget["T_domain_avg"]
            * norm_factor
            * ds_budget["dV_dt"]
            * HOURLY_RATE_SCALE
        ),
        "temperature_tendency": ds_budget["dT_dt"] * norm_factor * HOURLY_RATE_SCALE,
        "advection": ds_budget["advection_term"] * norm_factor * HOURLY_RATE_SCALE,
        "adiabatic": ds_budget["adiabatic_term"] * norm_factor * HOURLY_RATE_SCALE,
        "diabatic": ds_budget["diabatic_term"] * norm_factor * HOURLY_RATE_SCALE,
    }
    return {name: _rolling_mean(da, smoothing_window) for name, da in terms.items()}


def _add_domain_state_panel(
    ax,
    ds_budget: xr.Dataset,
    *,
    smoothing_window: int | None = None,
    daily: bool = False,
    temperature_context: TemperatureContext | None = None,
):
    if daily:
        t_domain_avg = _extend_daily_steps(
            ds_budget["T_domain_avg"].resample(time="1D").mean()
        )
        drawstyle = "steps-post"
    else:
        assert smoothing_window is not None
        t_domain_avg = _rolling_mean(ds_budget["T_domain_avg"], smoothing_window)
        drawstyle = None

    line_temperature = _plot_series(
        ax,
        t_domain_avg,
        label=r"$\langle T\rangle$",
        color=style.COLORS["temperature"],
        drawstyle=drawstyle,
    )
    ax.set_ylabel("Temperature [K]")
    style.use_default_numeric_formatter(ax.yaxis)
    ax.set_title("Temperature", loc="left", pad=4)
    handles = [line_temperature]
    if temperature_context is not None:
        context = temperature_context.series
        if not context.time.equals(ds_budget.time):
            raise ValueError(
                "Temperature context must match the selected budget time coordinate exactly."
            )
        t_clim = context["T_mean_climatology"]
        t_clim = (
            _extend_daily_steps(t_clim.resample(time="1D").mean())
            if daily
            else _rolling_mean(t_clim, smoothing_window)
        )
        handles.append(
            _plot_series(
                ax,
                t_clim,
                label=r"$\langle T\rangle$ climatology",
                color=style.COLORS["temperature"],
                linestyle="--",
                drawstyle=drawstyle,
            )
        )
        surface = {
            name: _extend_daily_steps(context[name].resample(time="1D").first())
            for name in ("tas", "tas_climatology", "tas_q90")
        }
        for name, label, linestyle in (
            ("tas", "T2m (daily)", "-"),
            ("tas_climatology", "T2m climatology", "--"),
        ):
            handles.append(
                _plot_series(
                    ax,
                    surface[name],
                    label=label,
                    color=style.COLORS["surface_temperature"],
                    linestyle=linestyle,
                    drawstyle="steps-post",
                )
            )
        band = ax.fill_between(
            surface["tas_climatology"].time.values,
            surface["tas_climatology"].values,
            surface["tas_q90"].values,
            step="post",
            color=style.COLORS["surface_temperature"],
            alpha=style.TEMPERATURE_BAND_ALPHA,
            linewidth=0,
            zorder=0,
            label="T2m mean to q90",
        )
        handles.append(band)
    style.pad_y_limits(ax, fraction=0.08)
    style.inside_legend(ax, handles, [handle.get_label() for handle in handles], ncol=2)


def _extend_daily_steps(da: xr.DataArray) -> xr.DataArray:
    """Include the last day's right edge so steps and shading cover that day."""
    edge = da.isel(time=[-1]).assign_coords(
        time=[da.time.values[-1] + np.timedelta64(1, "D")]
    )
    return xr.concat([da, edge], dim="time")


def _add_temperature_note(
    fig, context: TemperatureContext | None, *, daily: bool, window: int = 1
):
    if context is None:
        return
    baseline = context.provenance["T_mean_climatology"]
    years = context.provenance["T2m_reference"]["selected_years"]
    year_label = (
        str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]} (year-specific)"
    )
    resolution = (
        "daily means"
        if daily
        else ("hourly" if window == 1 else f"{window}-hour centered means")
    )
    style.add_result_note(
        fig,
        f"T2m: daily; evolving mean and q90 for {year_label}. "
        rf"$\langle T\rangle$: {resolution}; fixed {baseline['start_year']}-{baseline['end_year']} climatology.",
    )


def _add_tendency_panel(ax, terms: dict[str, xr.DataArray], *, units: str, daily: bool):
    drawstyle = "steps-post" if daily else None
    lines = [
        # _plot_series(
        #     ax,
        #     terms["storage"],
        #     label=r"$d/dt(\int T\,dV)$",
        #     color=style.COLORS["storage"],
        #     drawstyle=drawstyle,
        # ),
        # _plot_series(
        #     ax,
        #     terms["volume_tendency"],
        #     label=r"$\langle T\rangle/V \cdot dV/dt$",
        #     color=style.COLORS["volume_tendency"],
        #     drawstyle=drawstyle,
        # ),
        _plot_series(
            ax,
            terms["temperature_tendency"],
            label=r"$d\langle T\rangle/dt$",
            color=style.COLORS["temperature_tendency"],
            drawstyle=drawstyle,
        ),
    ]
    style.zero_line(ax)
    ax.set_ylabel(units)
    ax.set_title("Temperature tendency", loc="left", pad=4)
    style.inside_legend(
        ax,
        lines,
        [line.get_label() for line in lines],
        ncol=3,
    )


def _add_budget_terms_panel(
    ax,
    terms: dict[str, xr.DataArray],
    ds_budget: xr.Dataset,
    *,
    units: str,
    daily: bool,
    smoothing_window: int | None = None,
):
    drawstyle = "steps-post" if daily else None
    labels = {
        "advection": "Advection",
        "adiabatic": "Adiabatic",
        "diabatic": "Diabatic",
    }
    lines = []
    for name in ("advection", "adiabatic", "diabatic"):
        line = _plot_series(
            ax,
            terms[name],
            label=labels[name],
            color=style.COLORS[name],
            drawstyle=drawstyle,
        )
        lines.append(line)

        if "advection_error" in ds_budget.data_vars and name in {
            "advection",
            "diabatic",
        }:
            error = np.abs(ds_budget["advection_error"])
            error = error * (1 / ds_budget["domain_volume"]) * HOURLY_RATE_SCALE
            if daily:
                error = error.resample(time="1D").sum()  # type: ignore
            else:
                assert smoothing_window is not None
                error = _rolling_mean(error, smoothing_window)  # type: ignore
            ax.fill_between(
                terms[name]["time"].values,
                (terms[name] - error).values,
                (terms[name] + error).values,
                color=style.COLORS[name],
                alpha=0.12,
                linewidth=0,
                step="post" if daily else None,
                zorder=0,
            )

    style.zero_line(ax)
    ax.set_ylabel(units)
    ax.set_title("Heat-budget terms", loc="left", pad=4)
    style.inside_legend(
        ax,
        lines,
        [line.get_label() for line in lines],
        ncol=3,
    )


def _set_shared_symmetric_budget_limits(axes) -> None:
    ymax = max(
        abs(axes[1].get_ylim()[0]),
        abs(axes[1].get_ylim()[1]),
        abs(axes[2].get_ylim()[0]),
        abs(axes[2].get_ylim()[1]),
    )
    if np.isfinite(ymax) and ymax > 0:
        axes[1].set_ylim(-1.08 * ymax, 1.08 * ymax)
        axes[2].set_ylim(-1.08 * ymax, 1.08 * ymax)


def plot_budget_terms_hourly(
    ds_budget: xr.Dataset,
    smoothing_window: int,
    plot_dir: str,
    *,
    temperature_context: TemperatureContext | None = None,
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    terms = _hourly_budget_terms(ds_budget, smoothing_window)
    units = r"[K hr$^{-1}$]"

    fig, ax = plt.subplots(
        figsize=style.publication_figsize(RESULT_WIDTH, RESULT_ASPECT),
        nrows=3,
        sharex=True,
        gridspec_kw={"height_ratios": style.RESULT_HEIGHT_RATIOS},
    )

    _add_domain_state_panel(
        ax[0],
        ds_budget,
        smoothing_window=smoothing_window,
        temperature_context=temperature_context,
    )
    _add_tendency_panel(ax[1], terms, units=units, daily=False)
    _add_budget_terms_panel(
        ax[2],
        terms,
        ds_budget,
        units=units,
        daily=False,
        smoothing_window=smoothing_window,
    )
    _set_shared_symmetric_budget_limits(ax)
    _format_result_axes(ax)
    style.apply_stacked_layout(fig, 3, width=RESULT_WIDTH)
    if ds_budget.sizes["time"] > 1:
        ax[-1].set_xlim(ds_budget.time.values[0], ds_budget.time.values[-1])
    _add_temperature_note(
        fig, temperature_context, daily=False, window=smoothing_window
    )

    out_path = (
        plot_path
        / f"budget_terms_timeseries_hourly_smoothwindow_{smoothing_window}.png"
    )
    style.save_figure(fig, out_path)
    plt.close(fig)


def plot_budget_terms_day_bin(
    ds_budget: xr.Dataset,
    plot_dir: str,
    *,
    temperature_context: TemperatureContext | None = None,
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    terms = _daily_budget_terms(ds_budget)
    units = r"[K day$^{-1}$]"

    fig, ax = plt.subplots(
        figsize=style.publication_figsize(RESULT_WIDTH, RESULT_ASPECT),
        nrows=3,
        sharex=True,
        gridspec_kw={"height_ratios": style.RESULT_HEIGHT_RATIOS},
    )

    _add_domain_state_panel(
        ax[0], ds_budget, daily=True, temperature_context=temperature_context
    )
    _add_tendency_panel(ax[1], terms, units=units, daily=True)
    _add_budget_terms_panel(ax[2], terms, ds_budget, units=units, daily=True)
    _set_shared_symmetric_budget_limits(ax)
    _format_result_axes(ax)
    style.apply_stacked_layout(fig, 3, width=RESULT_WIDTH)
    _add_temperature_note(fig, temperature_context, daily=True)

    style.save_figure(fig, plot_path / "budget_terms_timeseries_daily.png")
    plt.close(fig)


def plot_constant_T_results(
    ds_budget: xr.Dataset,
    ds_test: xr.Dataset,
    plot_dir: str,
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    norm_factor = 1 / ds_budget["domain_volume"]
    time_rate_conversion = HOURLY_RATE_SCALE

    fig, ax = plt.subplots(
        figsize=style.publication_figsize("full", style.TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    line_error = _plot_series(
        ax[0],
        ds_budget["advection_error"] * norm_factor * time_rate_conversion,
        label=r"$\delta M T_{scale}$",
        color=style.COLORS["diabatic"],
    )
    line_test = _plot_series(
        ax[0],
        ds_test["advection_term"] * norm_factor * time_rate_conversion,
        label=r"$\mathcal{F}_{advection}$",
        color=style.COLORS["advection"],
    )
    style.inside_legend(
        ax[0],
        [line_error, line_test],
        [line_error.get_label(), line_test.get_label()],
        ncol=2,
    )

    dt = (ds_budget["time"][1] - ds_budget["time"][0]).values / np.timedelta64(1, "s")
    integrated_advection_error = (
        ds_budget["advection_error"] * norm_factor * dt
    ).cumsum(dim="time")
    integrated_net_heat_advection = (
        ds_test["advection_term"] * norm_factor * dt
    ).cumsum(dim="time")

    line_integrated_error = _plot_series(
        ax[1],
        integrated_advection_error,
        label=r"$T_{scale}\int \delta M\,dt$",
        color=style.COLORS["diabatic"],
    )
    line_integrated_test = _plot_series(
        ax[1],
        integrated_net_heat_advection,
        label=r"$\int \mathcal{F}_{advection}\,dt$",
        color=style.COLORS["advection"],
    )
    style.inside_legend(
        ax[1],
        [line_integrated_error, line_integrated_test],
        [line_integrated_error.get_label(), line_integrated_test.get_label()],
        ncol=2,
    )

    ax[0].set_title("Advection rate", loc="left", pad=4)
    ax[1].set_title("Integrated advection", loc="left", pad=4)
    ax[0].set_ylabel(r"[K hr$^{-1}$]")
    ax[1].set_ylabel("[K]")
    _format_result_axes(ax)
    style.apply_stacked_layout(fig, 2, width="full")
    style.save_figure(fig, plot_path / "constant_T_advection_comparison.png")
    plt.close(fig)
