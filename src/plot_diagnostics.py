'''

'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter

from . import plot_style as style

SINGLE_COLUMN_WIDTH_IN = style.SINGLE_COLUMN_WIDTH_IN
FULL_TWO_COLUMN_WIDTH_IN = style.FULL_TWO_COLUMN_WIDTH_IN

PAPER_FONT_SIZE_PT = style.PAPER_FONT_SIZE_PT
LEGEND_FONT_SIZE_PT = style.LEGEND_FONT_SIZE_PT
LINE_WIDTH_PT = style.LINE_WIDTH_PT
SCATTER_SIZE_PT2 = style.SCATTER_SIZE_PT2
SCATTER_ALPHA = style.SCATTER_ALPHA

SINGLE_PANEL_ASPECT = style.SINGLE_PANEL_ASPECT
TWO_PANEL_STACK_ASPECT = 0.95
THREE_PANEL_STACK_ASPECT = 1.2
SQUARE_PANEL_ASPECT = style.SQUARE_PANEL_ASPECT
MASS_FLUX_UNITS = r"m$^2$ Pa s$^{-1}$"
HEAT_FLUX_UNITS = r"K m$^2$ Pa s$^{-1}$"
FIG5_MASS_FLUX_COLOR = style.COLORS["mass"]
FIG5_TOTAL_HEAT_FLUX_COLOR = "red"
FIG5_PRIME_HEAT_FLUX_COLOR = "black"
WALL_FACES = ("north", "south", "east", "west")
FIG5_PER_FACE_VARIABLES = tuple(
    f"{prefix}_{face}"
    for prefix in (
        "benchmark_mass_flux",
        "benchmark_heat_flux",
        "mass_flux_contribution",
        "flux_contribution",
    )
    for face in WALL_FACES
) + ("T_domain_avg",)
FIG5_NET_VARIABLES = (
    "benchmark_mass_flux_net",
    "calculated_mass_flux_net_lateral",
    "benchmark_heat_flux_net",
    "calculated_heat_flux_net_lateral_full",
    "calculated_heat_flux_net_lateral",
    "benchmark_heat_flux_net_lateral_prime",
)
BENCHMARK_MASS_CONTINUITY_VARIABLES = (
    "dV_dt",
    "dV_dt_true",
    "net_mass_advection",
    "benchmark_mass_flux_net",
)
FIG6_VARIABLES = (
    "adiabatic_term",
    "diabatic_term",
    "calculated_diabatic_term_physical",
    "residual_heat",
    "benchmark_adiabatic_term",
    "benchmark_diabatic_term",
    "benchmark_diabatic_term_physical",
    "benchmark_residual_heat",
)
FIG6_TERM_SPECS = (
    (
        "benchmark_adiabatic_term",
        "adiabatic_term",
        r"Adiabatic pressure work ($\mathcal{C}$)",
        style.COLORS["adiabatic"],
    ),
    (
        "benchmark_diabatic_term",
        "diabatic_term",
        r"Workflow diabatic residual ($\mathcal{D}_0$)",
        style.COLORS["diabatic"],
    ),
    (
        "benchmark_diabatic_term_physical",
        "calculated_diabatic_term_physical",
        r"Physical diabatic residual ($\mathcal{D}_{phys}$)",
        "#7B3294",
    ),
)
FIG7_VARIABLES = (
    "benchmark_heat_flux_divergence",
    "benchmark_heat_flux_divergence_from_walls",
    "benchmark_mass_flux_divergence",
    "benchmark_mass_flux_divergence_from_walls",
)
FIG8_TERM_SPECS = (
    (
        "benchmark_storage_term",
        "d_dt_T",
        r"Total storage ($d\int T\,dV/dt$)",
        style.COLORS["storage"],
    ),
    (
        "benchmark_mean_temperature_storage_term",
        "dT_dt",
        r"Mean-temperature storage ($V\,d\langle T\rangle/dt$)",
        "#1B9E77",
    ),
    (
        "benchmark_volume_change_storage_term",
        "volume_change_storage_term",
        r"Volume-change storage ($\langle T\rangle\,dV/dt$)",
        style.COLORS["residual"],
    ),
)
FIG8_VARIABLES = tuple(
    name
    for benchmark_name, calculated_name, _title, _color in FIG8_TERM_SPECS
    for name in (benchmark_name, calculated_name)
)
CORE_DIAGNOSTIC_VARIABLES = (
    "dV_dt",
    "domain_volume",
    "net_mass_advection",
    "advection_term",
    "advection_error",
    "d_dt_T",
    "dT_dt",
    "dT_dt_2",
)
ONE_TO_ONE_FIGURE_LAYOUT = style.ONE_TO_ONE_FIGURE_LAYOUT


def _publication_figsize(width: str = "single", aspect: float = SINGLE_PANEL_ASPECT):
    return style.publication_figsize(width, aspect)


def _date_locator_formatter():
    return style.date_locator_formatter()


def _apply_stacked_figure_layout(fig, nrows: int, width: str = "single"):
    style.apply_stacked_layout(fig, nrows, width=width)


def _format_time_axes(axes) -> None:
    for axis in axes:
        style.style_axis(axis)
    style.format_time_axis(axes[-1])
    for axis in axes[:-1]:
        axis.set_xlabel("")


def _format_cumulative_mass_tick(value: float, _position: int) -> str:
    if abs(value) < 5e-10:
        return "0"
    return f"{value:.2f}"


def _pad_upper_ylim(ax, fraction: float = 0.15):
    style.pad_y_limits(ax, fraction=fraction)


def _set_square_1_to_1_axis(ax):
    style.format_one_to_one_axis(ax)


def _set_square_xlabel(ax, label: str):
    ax.set_xlabel(label)


def _add_correlation_text(ax, text: str):
    anchored_text = AnchoredText(
        text,
        loc="upper left",
        prop={"size": LEGEND_FONT_SIZE_PT},
        frameon=True,
        borderpad=0.6,
        pad=0.2,
    )
    anchored_text.patch.set_facecolor("white")
    anchored_text.patch.set_edgecolor("#CFCFCF")
    anchored_text.patch.set_alpha(0.9)
    ax.add_artist(anchored_text)


def _prepare_square_figure_for_save(fig):
    fig.subplots_adjust(**ONE_TO_ONE_FIGURE_LAYOUT)
    fig.canvas.draw()


def _save_square_figure(fig, path: str):
    _prepare_square_figure_for_save(fig)
    style.save_figure(fig, path)


def _mass_flux_wall_label(var_name: str) -> str:
    prefix = "mass_flux_contribution_"
    if var_name.startswith(prefix):
        return var_name[len(prefix):].replace("_", " ").title()
    return var_name


def _require_dataset_variables(
    ds: xr.Dataset,
    variables: tuple[str, ...],
    context: str,
) -> None:
    required = tuple(dict.fromkeys(variables))
    missing = [name for name in required if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"{context} requires variables missing from heat_budget.nc: "
            f"{', '.join(missing)}"
        )


def _advection_terms_for_core_diagnostics(ds_budget: xr.Dataset) -> xr.Dataset:
    advection_variables = [
        name
        for name in ds_budget.data_vars
        if (
            name in {"net_mass_advection", "advection_term"}
            or name.startswith("mass_flux_contribution_") #type: ignore
            or name.startswith("flux_contribution_") #type: ignore
        )
    ]
    return ds_budget[advection_variables]


def plot_core_diagnostics(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Generate non-benchmark diagnostic figures from a reduced dataset."""
    _require_dataset_variables(
        ds_budget,
        CORE_DIAGNOSTIC_VARIABLES,
        "Core diagnostics",
    )
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    advection_terms = _advection_terms_for_core_diagnostics(ds_budget)
    plot_dir_str = str(plot_path)

    fig1_mass_continuity(
        ds_budget["dV_dt"],
        advection_terms,
        plot_dir_str,
    )
    fig2_mass_advection_residual_timeseries(
        advection_terms,
        ds_budget["dV_dt"],
        ds_budget["domain_volume"],
        plot_dir_str,
    )
    # fig3_advection_components_timeseries(
    #     advection_terms,
    #     ds_budget["dV_dt"],
    #     ds_budget["advection_error"],
    #     ds_budget["domain_volume"],
    #     plot_dir_str,
    # )
    # fig4_temperature_derivative_timeseries(
    #     ds_budget["d_dt_T"],
    #     ds_budget["dT_dt"],
    #     ds_budget["dT_dt_2"],
    #     ds_budget["domain_volume"],
    #     plot_dir_str,
    # )


def fig1_mass_continuity(dV_dt: xr.DataArray, advection_terms: xr.Dataset, plot_dir: str):
    fig,ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    x, y = style.finite_pair(dV_dt.values, advection_terms["net_mass_advection"].values)
    lower, upper = style.one_to_one_limits(x, y, symmetric=True)

    ax.scatter(
        x,
        y,
        s=SCATTER_SIZE_PT2,
        color=style.COLORS["mass"],
        alpha=SCATTER_ALPHA,
        edgecolors="none",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color=style.COLORS["advection"],
        linewidth=style.REFERENCE_LINE_WIDTH_PT,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    _set_square_1_to_1_axis(ax)

    _set_square_xlabel(ax, f"dV/dt [{MASS_FLUX_UNITS}]")
    ax.set_ylabel(f"Net mass-flux [{MASS_FLUX_UNITS}]")

    _save_square_figure(fig, plot_dir + '/fig1_mass_continuity.png')
    plt.close(fig)


def fig1_benchmark_mass_continuity(
    dV_dt: xr.DataArray,
    advection_terms: xr.Dataset,
    dV_dt_true: xr.DataArray,
    benchmark_mass_flux_net: xr.DataArray,
    plot_dir: str,
):
    def _finite_pair(x: xr.DataArray, y: xr.DataArray):
        return style.finite_pair(x.values, y.values)

    def _axis_limits(x: np.ndarray, y: np.ndarray):
        return style.one_to_one_limits(x, y)

    def _correlation_text(x: np.ndarray, y: np.ndarray):
        if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
            return f"Pearson r: n/a\nn = {x.size}"
        r = np.corrcoef(x, y)[0, 1]
        return f"Pearson r: {r:.3f}\nn = {x.size}"

    def _plot_benchmark_vs_calculated(
        benchmark: xr.DataArray,
        calculated: xr.DataArray,
        xlabel: str,
        ylabel: str,
        title: str,
        filename: str,
    ):
        benchmark_aligned, calculated_aligned = xr.align(
            benchmark,
            calculated,
            join="inner",
        )
        x, y = _finite_pair(benchmark_aligned, calculated_aligned)
        lower, upper = _axis_limits(x, y)

        fig, ax = plt.subplots(
            figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
        )
        ax.scatter(
            x,
            y,
            alpha=SCATTER_ALPHA,
            s=SCATTER_SIZE_PT2,
            color=style.COLORS["mass"],
            edgecolors="none",
        )
        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            color=style.COLORS["advection"],
            linewidth=style.REFERENCE_LINE_WIDTH_PT,
            label="1:1",
        )

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        _set_square_1_to_1_axis(ax)
        _set_square_xlabel(ax, xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)
        _add_correlation_text(ax, _correlation_text(x, y))
        ax.legend(loc="lower right", **style.legend_kwargs())

        _save_square_figure(fig, plot_dir + "/" + filename)
        plt.close(fig)

    eulerian_x, eulerian_y = xr.align(
        dV_dt,
        advection_terms["net_mass_advection"],
        join="inner",
    )
    benchmark_x, benchmark_y = xr.align(
        dV_dt_true,
        benchmark_mass_flux_net,
        join="inner",
    )

    x_e = eulerian_x.values.ravel()
    y_e = eulerian_y.values.ravel()
    x_b = benchmark_x.values.ravel()
    y_b = benchmark_y.values.ravel()

    mask_e = np.isfinite(x_e) & np.isfinite(y_e)
    mask_b = np.isfinite(x_b) & np.isfinite(y_b)

    lower, upper = style.one_to_one_limits(
        np.concatenate([x_e[mask_e], x_b[mask_b]]),
        np.concatenate([y_e[mask_e], y_b[mask_b]]),
        symmetric=True,
    )

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    ax.scatter(
        x_e[mask_e],
        y_e[mask_e],
        alpha=SCATTER_ALPHA,
        label="Eulerian",
        s=SCATTER_SIZE_PT2,
        color=style.COLORS["mass"],
        edgecolors="none",
    )
    ax.scatter(
        x_b[mask_b],
        y_b[mask_b],
        alpha=SCATTER_ALPHA,
        label="Benchmark",
        s=SCATTER_SIZE_PT2,
        color=style.COLORS["diabatic"],
        edgecolors="none",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color=style.COLORS["advection"],
        linewidth=style.REFERENCE_LINE_WIDTH_PT,
        label="1:1",
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    _set_square_1_to_1_axis(ax)
    _set_square_xlabel(ax, f"dV/dt [{MASS_FLUX_UNITS}]")
    ax.set_ylabel(f"Net mass-flux [{MASS_FLUX_UNITS}]")
    style.inside_legend(
        ax,
        [ax.collections[0], ax.collections[1], ax.lines[0]],
        ["Eulerian", "Benchmark", "1:1"],
        ncol=3,
    )

    _save_square_figure(fig, plot_dir + "/fig1_benchmark_mass_continuity.png")
    plt.close(fig)

    calculated_mass_flux_lateral = advection_terms["net_mass_advection"]
    for vertical_flux in (
        "mass_flux_contribution_top",
        "mass_flux_contribution_bottom",
    ):
        if vertical_flux in advection_terms:
            calculated_mass_flux_lateral = (
                calculated_mass_flux_lateral - advection_terms[vertical_flux]
            )

    _plot_benchmark_vs_calculated(
        benchmark_mass_flux_net,
        calculated_mass_flux_lateral,
        f"Benchmark [{MASS_FLUX_UNITS}]",
        f"Calculated [{MASS_FLUX_UNITS}]",
        "Net mass-flux",
        "fig1.1_benchmark_vs_calculated_mass_flux.png",
    )
    _plot_benchmark_vs_calculated(
        dV_dt_true,
        dV_dt,
        f"Benchmark [{MASS_FLUX_UNITS}]",
        f"Calculated [{MASS_FLUX_UNITS}]",
        "dV/dt",
        "fig1.2_benchmark_vs_calculated_dV_dt.png",
    )




def fig2_mass_advection_residual_timeseries(
    advection_terms: xr.Dataset,
    dV_dt: xr.DataArray,
    domain_volume: xr.DataArray,
    plot_dir: str,
):
    mean_norm_factor = 1 / np.nanmean(domain_volume)
    time_rate_conversion = 3600
    neg_dV_dt = -dV_dt
    delta_mass = advection_terms["net_mass_advection"] - dV_dt

    if not np.issubdtype(advection_terms["time"].dtype, np.datetime64):
        raise ValueError("Time coordinate is not in datetime format.")

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    line_residual = ax[0].plot(
        delta_mass["time"],
        delta_mass * mean_norm_factor * time_rate_conversion,
        label=r"$\delta M$",
        color=style.COLORS["residual"],
        linewidth=LINE_WIDTH_PT,
    )[0]
    line_net = ax[0].plot(
        delta_mass["time"],
        advection_terms["net_mass_advection"] * mean_norm_factor * time_rate_conversion,
        label="Net mass-flux",
        color=style.COLORS["mass"],
        alpha=0.75,
        linewidth=LINE_WIDTH_PT,
    )[0]
    line_dvdt = ax[0].plot(
        delta_mass["time"],
        neg_dV_dt * mean_norm_factor * time_rate_conversion,
        label=r"$-dV/dt$",
        color=style.COLORS["temperature"],
        alpha=0.75,
        linewidth=LINE_WIDTH_PT,
    )[0]
    style.zero_line(ax[0])

    dt = (advection_terms["time"][1] - advection_terms["time"][0]).values / np.timedelta64(1, "s")
    cumulative_residual = (delta_mass * dt).cumsum(dim="time")
    cumulative_net_adv = (advection_terms["net_mass_advection"] * dt).cumsum(dim="time")
    cumulative_dvdt = (neg_dV_dt * dt).cumsum(dim="time")

    ax[1].plot(
        advection_terms["time"],
        cumulative_residual * mean_norm_factor,
        color=style.COLORS["residual"],
        linewidth=LINE_WIDTH_PT,
    )
    ax[1].plot(
        advection_terms["time"],
        cumulative_net_adv * mean_norm_factor,
        color=style.COLORS["mass"],
        linewidth=LINE_WIDTH_PT,
    )
    ax[1].plot(
        advection_terms["time"],
        cumulative_dvdt * mean_norm_factor,
        color=style.COLORS["temperature"],
        linewidth=LINE_WIDTH_PT,
    )
    style.zero_line(ax[1])

    ax[0].set_ylabel(r"[hr$^{-1}$]")
    ax[1].set_ylabel("V fraction")
    ax[0].set_title("Mass residual terms - volume normalized", loc="left", pad=4)
    ax[1].set_title("Cumulative mass residual - volume normalized", loc="left", pad=4)
    style.inside_legend(
        ax[0],
        [line_residual, line_net, line_dvdt],
        [line_residual.get_label(), line_net.get_label(), line_dvdt.get_label()],
        ncol=3,
    )
    _format_time_axes(ax)
    ax[1].set_xlabel("Time")
    style.pad_y_limits(ax[0], fraction=0.08, symmetric=True)
    style.pad_y_limits(ax[1], fraction=0.08)
    ax[1].yaxis.set_major_formatter(FuncFormatter(_format_cumulative_mass_tick))
    _apply_stacked_figure_layout(fig, 2)
    style.save_figure(fig, plot_dir + "/fig2_mass_residual_time_series.png")
    plt.close(fig)

    mass_vars = [
        str(component)
        for component in advection_terms.data_vars
        if (
            "mass" in str(component)
            and "net" not in str(component)
            and "abs" not in str(component)
        )
    ]

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    advection_terms_smoothed = advection_terms.rolling(
        time=24,
        center=True,
        min_periods=1,
    ).mean()
    delta_mass_smoothed = delta_mass.rolling(
        time=24,
        center=True,
        min_periods=1,
    ).mean()

    net_zonal_mass_advection = xr.zeros_like(advection_terms_smoothed["net_mass_advection"])
    net_meridional_mass_advection = xr.zeros_like(advection_terms_smoothed["net_mass_advection"])
    net_vertical_mass_advection = xr.zeros_like(advection_terms_smoothed["net_mass_advection"])
    component_lines = []
    component_labels = []

    for var in mass_vars:
        face = var.split("_")[-1]
        contribution = (
            advection_terms_smoothed[var] * mean_norm_factor * time_rate_conversion
        )
        component_lines.append(
            ax[0].plot(
                advection_terms_smoothed["time"],
                contribution,
                label=_mass_flux_wall_label(var),
                color=style.FACE_COLORS.get(face, style.COLORS["mass"]),
                linewidth=LINE_WIDTH_PT,
            )[0]
        )
        component_labels.append(_mass_flux_wall_label(var))

        if face in ["east", "west"]:
            net_zonal_mass_advection = net_zonal_mass_advection + contribution
        elif face in ["north", "south"]:
            net_meridional_mass_advection = net_meridional_mass_advection + contribution
        elif face in ["top", "bottom"]:
            net_vertical_mass_advection = net_vertical_mass_advection + contribution

    net_horizontal_mass_advection = net_zonal_mass_advection + net_meridional_mass_advection

    line_delta = ax[1].plot(
        advection_terms_smoothed["time"],
        delta_mass_smoothed * mean_norm_factor * time_rate_conversion,
        label=r"$\delta M$",
        color=style.COLORS["residual"],
        linewidth=LINE_WIDTH_PT,
    )[0]
    line_horizontal = ax[1].plot(
        advection_terms_smoothed["time"],
        net_horizontal_mass_advection,
        label="Horizontal",
        color=style.COLORS["mass"],
        linewidth=LINE_WIDTH_PT,
    )[0]
    line_vertical = ax[1].plot(
        advection_terms_smoothed["time"],
        net_vertical_mass_advection,
        label="Vertical",
        color=style.COLORS["temperature"],
        linewidth=LINE_WIDTH_PT,
    )[0]
    style.zero_line(ax[0])
    style.zero_line(ax[1])

    ax[0].set_title("Mass-flux components - volume normalized", loc="left", pad=4)
    ax[1].set_title("Mass-flux aggregate - volume normalized", loc="left", pad=4)
    ax[0].set_ylabel(r"[hr$^{-1}$]")
    ax[1].set_ylabel(r"[hr$^{-1}$]")

    style.inside_legend(ax[0], component_lines, component_labels, ncol=len(component_labels))
    style.inside_legend(
        ax[1],
        [line_delta, line_horizontal, line_vertical],
        [line_delta.get_label(), line_horizontal.get_label(), line_vertical.get_label()],
        ncol=3,
    )
    _format_time_axes(ax)
    ax[1].set_xlabel("Time")
    style.pad_y_limits(ax[0], fraction=0.08, symmetric=True)
    style.pad_y_limits(ax[1], fraction=0.08, symmetric=True)
    _apply_stacked_figure_layout(fig, 2)
    style.save_figure(fig, plot_dir + "/fig2.2_mass_advection_terms_timeseries.png")
    plt.close(fig)
    
    


def fig3_advection_components_timeseries(advection_terms: xr.Dataset, dV_dt: xr.DataArray, delta_heat: xr.DataArray, domain_volume:xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']


    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    mass_vars = []
    heat_vars = []
    # Plot each component of advection terms
    for component in advection_terms.data_vars:
        if "mass" in str(component) or "net" in str(component):
            mass_vars.append(str(component))
            pass
        else:
            heat_vars.append(str(component))
            ax[0].plot(advection_terms['time'], advection_terms[component] * norm_factor * time_rate_conversion, label=component, linewidth=LINE_WIDTH_PT)

    ax[0].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net heat advection', linewidth=LINE_WIDTH_PT, color=style.COLORS["advection"])

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Heat-advection components", loc="left", pad=4)

    ax[1].plot(advection_terms['time'], advection_terms['advection_term'] * norm_factor * time_rate_conversion, label='Net heat advection', linewidth=LINE_WIDTH_PT, color=style.COLORS["advection"])
    ax[1].plot(advection_terms['time'], delta_heat * norm_factor * time_rate_conversion, label='Expected Residual', linewidth=LINE_WIDTH_PT, color=style.COLORS["diabatic"])

    ax[1].set_title("Net heat advection", loc="left", pad=4)
    ax[1].set_ylabel('[K / hr]')

    style.inside_legend(ax[0], ax[0].lines, [line.get_label() for line in ax[0].lines])
    style.inside_legend(ax[1], ax[1].lines, [line.get_label() for line in ax[1].lines])
    _format_time_axes(ax)
    ax[1].set_xlabel("Time")
    _apply_stacked_figure_layout(fig, 2)
    style.save_figure(fig, plot_dir + '/fig3_advection_components_timeseries.png')
    plt.close()




def fig4_temperature_derivative_timeseries(d_dt_T: xr.DataArray, dT_dt_1:xr.DataArray, dT_dt_2:xr.DataArray, domain_volume: xr.DataArray, plot_dir: str):

    norm_factor = 1 / domain_volume
    time_rate_conversion = 3600 # convert from per second to per hour

    # delta_mass = dV_dt + advection_terms['net_mass_advection']

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    ax[0].plot(d_dt_T['time'], d_dt_T * norm_factor * time_rate_conversion, label=r'd/dt $\int TdV$', linewidth=LINE_WIDTH_PT, color=style.COLORS["storage"])

    ax[0].set_ylabel("[K / hr]")
    ax[0].set_title("Storage term", loc="left", pad=4)

    ax[1].plot(dT_dt_2['time'], dT_dt_2 * norm_factor * time_rate_conversion, label=r'd/dt$\int TdV$-$\langle T \rangle $dV/dt', linewidth=LINE_WIDTH_PT, color=style.COLORS["volume_tendency"])
    ax[1].plot(dT_dt_1['time'], dT_dt_1 * norm_factor * time_rate_conversion, label=r'd$\langle T \rangle $/dt', linewidth=LINE_WIDTH_PT, color=style.COLORS["temperature_tendency"])
    
    ax[1].set_title("Temperature tendency", loc="left", pad=4)
    ax[1].set_ylabel('[K / hr]')

    style.inside_legend(ax[0], ax[0].lines, [line.get_label() for line in ax[0].lines])
    style.inside_legend(ax[1], ax[1].lines, [line.get_label() for line in ax[1].lines])
    _format_time_axes(ax)
    ax[1].set_xlabel("Time")
    _apply_stacked_figure_layout(fig, 2)
    style.save_figure(fig, plot_dir + '/fig4_temperature_derivative_timeseries.png')
    plt.close()



def fig5_benchmark_comparison(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Plot per-face benchmark/calculated flux comparisons for Figure 5."""
    _require_dataset_variables(
        ds_budget,
        FIG5_PER_FACE_VARIABLES
        + (
            "benchmark_mass_flux_net",
            "calculated_mass_flux_net_lateral",
            "benchmark_heat_flux_net",
            "calculated_heat_flux_net_lateral_full",
        ),
        "Figure 5",
    )
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    colors = style.FACE_COLORS

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", TWO_PANEL_STACK_ASPECT),
        nrows=2,
        sharex=True,
    )

    for face in WALL_FACES:
        ax[0].plot(
            ds_budget["time"],
            ds_budget[f"benchmark_mass_flux_{face}"],
            linestyle="--",
            color=colors[face],
            alpha=0.75,
            linewidth=LINE_WIDTH_PT,
        )
        ax[0].plot(
            ds_budget["time"],
            ds_budget[f"mass_flux_contribution_{face}"],
            linestyle="-",
            color=colors[face],
            linewidth=LINE_WIDTH_PT,
        )

        calculated_full_heat_flux = (
            ds_budget[f"flux_contribution_{face}"]
            + ds_budget[f"mass_flux_contribution_{face}"]
            * ds_budget["T_domain_avg"]
        )
        ax[1].plot(
            ds_budget["time"],
            ds_budget[f"benchmark_heat_flux_{face}"],
            linestyle="--",
            color=colors[face],
            alpha=0.75,
            linewidth=LINE_WIDTH_PT,
        )
        ax[1].plot(
            ds_budget["time"],
            calculated_full_heat_flux,
            linestyle="-",
            color=colors[face],
            linewidth=LINE_WIDTH_PT,
        )

    ax[0].plot(
        ds_budget["time"],
        ds_budget["benchmark_mass_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color=style.COLORS["benchmark"],
    )
    ax[0].plot(
        ds_budget["time"],
        ds_budget["calculated_mass_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color=style.COLORS["calculated"],
    )

    (benchmark_heat_net_line,) = ax[1].plot(
        ds_budget["time"],
        ds_budget["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color=style.COLORS["benchmark"],
        label="Benchmark net",
    )
    (calculated_heat_net_line,) = ax[1].plot(
        ds_budget["time"],
        ds_budget["calculated_heat_flux_net_lateral_full"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color=style.COLORS["calculated"],
        label="Calculated net",
    )

    _format_time_axes(ax)
    style.zero_line(ax[0])
    style.zero_line(ax[1])

    ax[0].set_ylabel(rf"[{MASS_FLUX_UNITS}]")
    ax[0].set_title("Mass-flux comparison", loc="left", pad=4)

    face_handles = [
        Line2D([0], [0], color=colors["north"], lw=LINE_WIDTH_PT, label="North"),
        Line2D([0], [0], color=colors["south"], lw=LINE_WIDTH_PT, label="South"),
        Line2D([0], [0], color=colors["east"], lw=LINE_WIDTH_PT, label="East"),
        Line2D([0], [0], color=colors["west"], lw=LINE_WIDTH_PT, label="West"),
        Line2D([0], [0], color=style.COLORS["calculated"], lw=LINE_WIDTH_PT, label="Net"),
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color=style.COLORS["benchmark"],
            lw=LINE_WIDTH_PT,
            linestyle="--",
            label="Benchmark",
        ),
        Line2D(
            [0],
            [0],
            color=style.COLORS["calculated"],
            lw=LINE_WIDTH_PT,
            linestyle="-",
            label="Calculated",
        ),
    ]

    leg1 = ax[0].legend(
        handles=face_handles,
        loc="upper center",
        ncol=5,
        **style.legend_kwargs(
            columnspacing=0.75,
            handlelength=1.5,
            borderpad=0.30,
        ),
    )
    ax[0].add_artist(leg1)
    ax[0].legend(
        handles=style_handles,
        loc="lower center",
        ncol=2,
        **style.legend_kwargs(
            columnspacing=0.9,
            handlelength=1.8,
            borderpad=0.30,
        ),
    )

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
    ax[1].set_title("Heat-flux comparison", loc="left", pad=4)
    style.inside_legend(
        ax[1],
        [benchmark_heat_net_line, calculated_heat_net_line],
        ["Benchmark net", "Calculated net"],
        ncol=2,
    )
    style.pad_y_limits(ax[0], fraction=0.08)
    style.pad_y_limits(ax[1], fraction=0.08)

    _apply_stacked_figure_layout(fig, 2)
    style.save_figure(fig, plot_path / "fig5_benchmark_comparison.png")
    plt.close()


def fig5_net_benchmark_comparison(
    ds_budget: xr.Dataset,
    plot_dir: str | Path,
) -> None:
    """Plot the net benchmark/calculated comparison used for Figure 5.1."""
    _require_dataset_variables(ds_budget, FIG5_NET_VARIABLES, "Figure 5.1")
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", THREE_PANEL_STACK_ASPECT),
        nrows=3,
        sharex=True,
    )

    ax[0].plot(
        ds_budget["time"],
        ds_budget["benchmark_mass_flux_net"],
        linestyle="--",
        color=FIG5_MASS_FLUX_COLOR,
        linewidth=LINE_WIDTH_PT,
        label=r"$F_{mass,bench}$",
        alpha=0.5
    )
    ax[0].plot(
        ds_budget["time"],
        ds_budget["calculated_mass_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color=FIG5_MASS_FLUX_COLOR,
        label=r"$F_{mass,calc}$",
        alpha=0.5
    )

    ax[1].plot(
        ds_budget["time"],
        ds_budget["benchmark_heat_flux_net"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color=FIG5_TOTAL_HEAT_FLUX_COLOR,
        label=r"$\mathcal{H}_{bench}$",
        alpha=0.5
    )
    ax[1].plot(
        ds_budget["time"],
        ds_budget["calculated_heat_flux_net_lateral_full"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color=FIG5_TOTAL_HEAT_FLUX_COLOR,
        label=r"$\mathcal{H}_{calc}$",
        alpha=0.5
    )

    ax[2].plot(
        ds_budget["time"],
        ds_budget["benchmark_heat_flux_net_lateral_prime"],
        linestyle="--",
        linewidth=LINE_WIDTH_PT,
        color=FIG5_PRIME_HEAT_FLUX_COLOR,
        label=r"$\mathcal{H}'_{bench}$",
        alpha=0.5
    )
    ax[2].plot(
        ds_budget["time"],
        ds_budget["calculated_heat_flux_net_lateral"],
        linestyle="-",
        linewidth=LINE_WIDTH_PT,
        color=FIG5_PRIME_HEAT_FLUX_COLOR,
        label=r"$\mathcal{H}'_{calc}$",
        alpha=0.5
    )

    _format_time_axes(ax)
    for axis in ax:
        style.zero_line(axis)

    ax[0].set_ylabel(rf"[{MASS_FLUX_UNITS}]")
    ax[0].set_title(r"Mass-flux comparison - $F_{mass} = \iint\vec (U \cdot \hat n)dA$", loc="left", pad=4)

    ax[0].legend(
        loc="upper center",
        ncol=2,
        **style.legend_kwargs(
            columnspacing=0.9,
            handlelength=1.8,
            borderpad=0.30,
        ),
    )

    ax[1].set_ylabel(r"[K m$^2$ Pa s$^{-1}$]")
    ax[1].set_title(r"Total heat-flux ($\mathcal{H}$)", loc="left", pad=4)
    ax[1].legend(
        loc="upper center",
        ncol=2,
        **style.legend_kwargs(
            columnspacing=0.8,
            handlelength=1.6,
            borderpad=0.30,
        ),
    )
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel(r"[K m$^2$ Pa s$^{-1}$]")
    ax[2].set_title(r"T' heat-flux ($\mathcal{H}'$)", loc="left", pad=4)
    style.inside_legend(ax[2], ax[2].lines[:2], [line.get_label() for line in ax[2].lines[:2]], ncol=2)
    for axis in ax:
        style.pad_y_limits(axis, fraction=0.08)

    _apply_stacked_figure_layout(fig, 3)
    style.save_figure(fig, plot_path / "fig5.1_net_benchmark_comparison.png")
    plt.close()


def _plot_heat_flux_correlation(
    benchmark: xr.DataArray,
    calculated: xr.DataArray,
    title: str,
    path: Path,
    *,
    point_color: str,
    benchmark_label: str = "Benchmark",
    calculated_label: str = "Calculated",
    units: str = HEAT_FLUX_UNITS,
) -> None:
    benchmark_aligned, calculated_aligned = xr.align(
        benchmark,
        calculated,
        join="inner",
    )
    x = benchmark_aligned.values.ravel()
    y = calculated_aligned.values.ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    lower, upper = style.one_to_one_limits(x, y)

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        correlation_label = f"Pearson r: n/a\nn = {x.size}"
    else:
        correlation_label = (
            f"Pearson r: {np.corrcoef(x, y)[0, 1]:.3f}\nn = {x.size}"
        )

    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SQUARE_PANEL_ASPECT),
    )
    ax.scatter(
        x,
        y,
        alpha=SCATTER_ALPHA,
        color=point_color,
        s=SCATTER_SIZE_PT2,
        edgecolors="none",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        color=style.COLORS["advection"],
        linewidth=style.REFERENCE_LINE_WIDTH_PT,
        label="1:1",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    _set_square_1_to_1_axis(ax)
    _set_square_xlabel(ax, f"{benchmark_label} [{units}]")
    ax.set_ylabel(f"{calculated_label} [{units}]")
    ax.set_title(title, pad=4)
    _add_correlation_text(ax, correlation_label)
    ax.legend(loc="lower right", **style.legend_kwargs())

    _save_square_figure(fig, str(path))
    plt.close()


def fig5_heat_flux_correlations(
    ds_budget: xr.Dataset,
    plot_dir: str | Path,
) -> None:
    """Plot the heat-flux correlation panels for Figures 5.2 and 5.3."""
    _require_dataset_variables(
        ds_budget,
        (
            "benchmark_heat_flux_net_lateral_prime",
            "calculated_heat_flux_net_lateral",
            "benchmark_heat_flux_net",
            "calculated_heat_flux_net_lateral_full",
        ),
        "Figures 5.2 and 5.3",
    )
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    _plot_heat_flux_correlation(
        ds_budget["benchmark_heat_flux_net_lateral_prime"],
        ds_budget["calculated_heat_flux_net_lateral"],
        r"T' heat-flux ($\mathcal{H}'$)",
        plot_path / "fig5.2_benchmark_vs_calculated_heat_flux_lateral_prime.png",
        point_color=FIG5_PRIME_HEAT_FLUX_COLOR,
    )
    _plot_heat_flux_correlation(
        ds_budget["benchmark_heat_flux_net"],
        ds_budget["calculated_heat_flux_net_lateral_full"],
        r"Total heat-flux ($\mathcal{H}$)",
        plot_path / "fig5.3_benchmark_vs_calculated_heat_flux_total.png",
        point_color=FIG5_TOTAL_HEAT_FLUX_COLOR,
    )


def plot_figure5_diagnostics(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Generate benchmark diagnostics from one reduced dataset."""
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)
    _require_dataset_variables(
        ds_budget,
        BENCHMARK_MASS_CONTINUITY_VARIABLES
        + FIG5_PER_FACE_VARIABLES
        + FIG5_NET_VARIABLES,
        "Benchmark diagnostics",
    )
    fig1_benchmark_mass_continuity(
        ds_budget["dV_dt"],
        _advection_terms_for_core_diagnostics(ds_budget),
        ds_budget["dV_dt_true"],
        ds_budget["benchmark_mass_flux_net"],
        str(plot_path),
    )
    fig5_benchmark_comparison(ds_budget, plot_path)
    fig5_net_benchmark_comparison(ds_budget, plot_path)
    fig5_heat_flux_correlations(ds_budget, plot_path)


def fig6_benchmark_heating_comparison(
    ds_budget: xr.Dataset,
    plot_dir: str | Path,
) -> None:
    """Plot aligned ERA5 and calculated full-column heating terms."""
    _require_dataset_variables(ds_budget, FIG6_VARIABLES, "Figure 6")
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        figsize=_publication_figsize("single", THREE_PANEL_STACK_ASPECT),
        nrows=3,
        sharex=True,
    )

    for axis, (benchmark_name, calculated_name, title, color) in zip(
        axes,
        FIG6_TERM_SPECS,
    ):
        benchmark, calculated = xr.align(
            ds_budget[benchmark_name],
            ds_budget[calculated_name],
            join="inner",
        )
        axis.plot(
            benchmark["time"],
            benchmark,
            linestyle="--",
            color=color,
            linewidth=LINE_WIDTH_PT,
            alpha=0.7,
            label="ERA5 benchmark",
        )
        axis.plot(
            calculated["time"],
            calculated,
            linestyle="-",
            color=color,
            linewidth=LINE_WIDTH_PT,
            label="Pressure-level calculation",
        )
        style.style_axis(axis)
        style.zero_line(axis)
        style.pad_y_limits(axis, fraction=0.08)
        axis.set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
        axis.set_title(title, loc="left", pad=4)
        style.inside_legend(
            axis,
            axis.lines[:2],
            [line.get_label() for line in axis.lines[:2]],
            ncol=2,
        )

    style.format_time_axis(axes[-1])
    axes[-1].set_xlabel("Time")
    for axis in axes[:-1]:
        axis.set_xlabel("")

    _apply_stacked_figure_layout(fig, 3)
    style.save_figure(
        fig,
        plot_path / "fig6_benchmark_heating_comparison.png",
    )
    plt.close()


def fig6_heating_correlations(
    ds_budget: xr.Dataset,
    plot_dir: str | Path,
) -> None:
    """Plot one-to-one panels for heating terms and mass-closure heating."""
    _require_dataset_variables(ds_budget, FIG6_VARIABLES, "Figures 6.1 to 6.4")
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    correlation_specs = (
        (
            *FIG6_TERM_SPECS[0][:3],
            "fig6.1_benchmark_vs_calculated_adiabatic.png",
            FIG6_TERM_SPECS[0][3],
        ),
        (
            *FIG6_TERM_SPECS[1][:3],
            "fig6.2_benchmark_vs_calculated_diabatic_workflow.png",
            FIG6_TERM_SPECS[1][3],
        ),
        (
            *FIG6_TERM_SPECS[2][:3],
            "fig6.3_benchmark_vs_calculated_diabatic_physical.png",
            FIG6_TERM_SPECS[2][3],
        ),
        (
            "benchmark_residual_heat",
            "residual_heat",
            r"Mass-closure heating ($\langle T\rangle\delta\mathcal{M}$)",
            "fig6.4_benchmark_vs_calculated_residual_heat.png",
            style.COLORS["residual"],
        ),
    )

    for (
        benchmark_name,
        calculated_name,
        title,
        filename,
        color,
    ) in correlation_specs:
        _plot_heat_flux_correlation(
            ds_budget[benchmark_name],
            ds_budget[calculated_name],
            title,
            plot_path / filename,
            point_color=color,
        )


def plot_figure6_diagnostics(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Generate the full-column adiabatic and diabatic Figure 6 bundle."""
    _require_dataset_variables(
        ds_budget,
        FIG6_VARIABLES,
        "Figure 6 benchmark diagnostics",
    )
    fig6_benchmark_heating_comparison(ds_budget, plot_dir)
    fig6_heating_correlations(ds_budget, plot_dir)


def plot_figure7_diagnostics(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Compare ERA5 wall-integrated mass and heat fluxes with divergences."""
    _require_dataset_variables(ds_budget, FIG7_VARIABLES, "Figure 7")
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    vithed, walls = xr.align(
        ds_budget["benchmark_heat_flux_divergence"],
        ds_budget["benchmark_heat_flux_divergence_from_walls"],
        join="inner",
    )
    color = style.COLORS["advection"]
    fig, ax = plt.subplots(
        figsize=_publication_figsize("single", SINGLE_PANEL_ASPECT),
    )
    ax.plot(
        vithed["time"],
        vithed,
        linestyle="--",
        color=color,
        linewidth=LINE_WIDTH_PT,
        alpha=0.75,
        label="ERA5 vithed",
    )
    ax.plot(
        walls["time"],
        walls,
        linestyle="-",
        color=color,
        linewidth=LINE_WIDTH_PT,
        label="ERA5 wall-flux sum",
    )
    style.style_axis(ax)
    style.zero_line(ax)
    style.pad_y_limits(ax, fraction=0.08)
    style.format_time_axis(ax)
    ax.set_xlabel("Time")
    ax.set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
    ax.set_title("Horizontal thermal-energy flux divergence", loc="left", pad=4)
    style.inside_legend(
        ax,
        ax.lines[:2],
        [line.get_label() for line in ax.lines[:2]],
        ncol=2,
    )
    style.save_figure(
        fig,
        plot_path / "fig7_vithed_vs_wall_flux_divergence.png",
    )
    plt.close(fig)

    _plot_heat_flux_correlation(
        vithed,
        walls,
        "Horizontal thermal-energy flux divergence",
        plot_path / "fig7.1_vithed_vs_wall_flux_divergence_correlation.png",
        point_color=color,
        benchmark_label="ERA5 vithed",
        calculated_label="ERA5 wall-flux sum",
    )

    _plot_heat_flux_correlation(
        ds_budget["benchmark_mass_flux_divergence"],
        ds_budget["benchmark_mass_flux_divergence_from_walls"],
        "Horizontal mass-flux divergence",
        plot_path / "fig7.2_vimad_vs_wall_mass_flux_divergence_correlation.png",
        point_color=style.COLORS["mass"],
        benchmark_label="ERA5 vimad",
        calculated_label="ERA5 wall-flux sum",
        units=r"m$^2$ Pa s$^{-1}$",
    )


def plot_figure8_diagnostics(ds_budget: xr.Dataset, plot_dir: str | Path) -> None:
    """Compare total thermal storage and its volume-average decomposition."""
    _require_dataset_variables(ds_budget, FIG8_VARIABLES, "Figure 8")
    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        figsize=_publication_figsize("single", THREE_PANEL_STACK_ASPECT),
        nrows=3,
        sharex=True,
    )
    for axis, (benchmark_name, calculated_name, title, color) in zip(
        axes,
        FIG8_TERM_SPECS,
    ):
        benchmark, calculated = xr.align(
            ds_budget[benchmark_name],
            ds_budget[calculated_name],
            join="inner",
        )
        axis.plot(
            benchmark["time"],
            benchmark,
            linestyle="--",
            color=color,
            linewidth=LINE_WIDTH_PT,
            alpha=0.75,
            label="ERA5 vithe",
        )
        axis.plot(
            calculated["time"],
            calculated,
            linestyle="-",
            color=color,
            linewidth=LINE_WIDTH_PT,
            label="Pressure-level calculation",
        )
        style.style_axis(axis)
        style.zero_line(axis)
        style.pad_y_limits(axis, fraction=0.08)
        axis.set_ylabel(rf"[{HEAT_FLUX_UNITS}]")
        axis.set_title(title, loc="left", pad=4)
        style.inside_legend(
            axis,
            axis.lines[:2],
            [line.get_label() for line in axis.lines[:2]],
            ncol=2,
        )

    style.format_time_axis(axes[-1])
    axes[-1].set_xlabel("Time")
    for axis in axes[:-1]:
        axis.set_xlabel("")
    _apply_stacked_figure_layout(fig, 3)
    style.save_figure(
        fig,
        plot_path / "fig8_benchmark_storage_comparison.png",
    )
    plt.close(fig)

    filenames = (
        "fig8.1_benchmark_vs_calculated_total_storage.png",
        "fig8.2_benchmark_vs_calculated_mean_temperature_storage.png",
        "fig8.3_benchmark_vs_calculated_volume_change_storage.png",
    )
    for (
        benchmark_name,
        calculated_name,
        title,
        color,
    ), filename in zip(FIG8_TERM_SPECS, filenames):
        _plot_heat_flux_correlation(
            ds_budget[benchmark_name],
            ds_budget[calculated_name],
            title,
            plot_path / filename,
            point_color=color,
        )
