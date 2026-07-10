"""Publication result plots for reduced heat-budget datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from . import plot_style as style

RESULT_WIDTH = "full"
RESULT_ASPECT = style.THREE_PANEL_STACK_ASPECT
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


def _format_twin_axis(ax) -> None:
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(False)
    ax.tick_params(axis="y", which="major", length=4, pad=3)
    ax.tick_params(axis="y", which="minor", length=2.5)
    ax.yaxis.get_offset_text().set_fontsize(style.PAPER_FONT_SIZE_PT)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_linewidth(0.9)


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
        "storage": (
            ds_budget["d_dt_T"] * norm_factor * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
        "volume_tendency": (
            ds_budget["T_domain_avg"]
            * norm_factor
            * ds_budget["dV_dt"]
            * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
        "temperature_tendency": (
            ds_budget["dT_dt"] * norm_factor * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
        "advection": (
            ds_budget["advection_term"] * norm_factor * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
        "adiabatic": (
            ds_budget["adiabatic_term"] * norm_factor * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
        "diabatic": (
            ds_budget["diabatic_term"] * norm_factor * HOURLY_RATE_SCALE
        ).resample(time="1D").sum(),
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
):
    if daily:
        domain_volume = ds_budget["domain_volume"].resample(time="1D").mean()
        t_domain_avg = ds_budget["T_domain_avg"].resample(time="1D").mean()
        drawstyle = "steps-post"
    else:
        assert smoothing_window is not None
        domain_volume = _rolling_mean(ds_budget["domain_volume"], smoothing_window)
        t_domain_avg = _rolling_mean(ds_budget["T_domain_avg"], smoothing_window)
        drawstyle = None

    line_temperature = _plot_series(
        ax,
        t_domain_avg,
        label=r"$\langle T\rangle$",
        color=style.COLORS["temperature"],
        drawstyle=drawstyle,
    )
    ax.set_ylabel(r"$\langle T\rangle$ [K]")
    style.use_default_numeric_formatter(ax.yaxis)
    ax.set_title("Domain state", loc="left", pad=4)

    ax_v = ax.twinx()
    line_volume = _plot_series(
        ax_v,
        domain_volume,
        label=r"$V$",
        color=style.COLORS["volume"],
        drawstyle=drawstyle,
    )
    ax_v.set_ylabel(r"$V$ [m$^2$ Pa]")
    _format_twin_axis(ax_v)
    style.pad_y_limits(ax, fraction=0.08)
    style.pad_y_limits(ax_v, fraction=0.08)

    style.inside_legend(
        ax,
        [line_temperature, line_volume],
        [r"$\langle T\rangle$", r"$V$"],
        ncol=2,
    )
    return ax_v


def _add_tendency_panel(ax, terms: dict[str, xr.DataArray], *, units: str, daily: bool):
    drawstyle = "steps-post" if daily else None
    lines = [
        _plot_series(
            ax,
            terms["storage"],
            label=r"$d/dt(\int T\,dV)$",
            color=style.COLORS["storage"],
            drawstyle=drawstyle,
        ),
        _plot_series(
            ax,
            terms["volume_tendency"],
            label=r"$\langle T\rangle/V \cdot dV/dt$",
            color=style.COLORS["volume_tendency"],
            drawstyle=drawstyle,
        ),
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
    ax.set_title("Storage decomposition", loc="left", pad=4)
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

        if "advection_error" in ds_budget.data_vars and name in {"advection", "diabatic"}:
            error = np.abs(ds_budget["advection_error"])
            error = error * (1 / ds_budget["domain_volume"]) * HOURLY_RATE_SCALE
            if daily:
                error = error.resample(time="1D").sum() #type: ignore
            else:
                assert smoothing_window is not None
                error = _rolling_mean(error, smoothing_window) #type: ignore
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
) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    terms = _hourly_budget_terms(ds_budget, smoothing_window)
    units = r"[K hr$^{-1}$]"

    fig, ax = plt.subplots(
        figsize=style.publication_figsize(RESULT_WIDTH, RESULT_ASPECT),
        nrows=3,
        sharex=True,
    )

    _add_domain_state_panel(ax[0], ds_budget, smoothing_window=smoothing_window)
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

    out_path = plot_path / f"budget_terms_timeseries_hourly_smoothwindow_{smoothing_window}.png"
    style.save_figure(fig, out_path)
    plt.close(fig)


def plot_budget_terms_day_bin(ds_budget: xr.Dataset, plot_dir: str) -> None:
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    terms = _daily_budget_terms(ds_budget)
    units = r"[K day$^{-1}$]"

    fig, ax = plt.subplots(
        figsize=style.publication_figsize(RESULT_WIDTH, RESULT_ASPECT),
        nrows=3,
        sharex=True,
    )

    _add_domain_state_panel(ax[0], ds_budget, daily=True)
    _add_tendency_panel(ax[1], terms, units=units, daily=True)
    _add_budget_terms_panel(ax[2], terms, ds_budget, units=units, daily=True)
    _set_shared_symmetric_budget_limits(ax)
    _format_result_axes(ax)
    style.apply_stacked_layout(fig, 3, width=RESULT_WIDTH)

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
