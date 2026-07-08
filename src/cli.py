"""Command-line entry point for static heat-budget publication plots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import xarray as xr

from . import config, io, plot_diagnostics, plot_results
from .specs import BenchmarkDiagnostics, DatasetPaths, PlotRequest

RESULT_REQUIRED_VARIABLES = (
    "domain_volume",
    "T_domain_avg",
    "d_dt_T",
    "dV_dt",
    "dT_dt",
    "advection_term",
    "adiabatic_term",
    "diabatic_term",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create static publication plots from reduced heat-budget data."
    )
    parser.add_argument(
        "--dataset",
        default=config.DEFAULT_DATASET,
        help=f"Dataset folder under data/. Default: {config.DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--start",
        dest="time_start",
        default=None,
        help="Inclusive start time for plotting, e.g. 1941-05-01T00:00:00.",
    )
    parser.add_argument(
        "--end",
        dest="time_end",
        default=None,
        help="Inclusive end time for plotting, e.g. 1941-09-01T00:00:00.",
    )
    parser.add_argument(
        "--plots-root",
        type=Path,
        default=config.PLOTS_ROOT,
        help=f"Root output directory. Default: {config.PLOTS_ROOT}",
    )
    parser.add_argument(
        "--benchmark-diagnostics",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Whether to create full-atmosphere benchmark diagnostics. "
            "auto enables them for dataset names ending in "
            f"{config.FULL_ATMOSPHERE_SUFFIX!r}."
        ),
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets that contain data/{dataset}/heat_budget.nc and exit.",
    )
    return parser


def should_run_benchmark_diagnostics(
    dataset: str,
    mode: BenchmarkDiagnostics,
) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return dataset.endswith(config.FULL_ATMOSPHERE_SUFFIX)


def require_variables(
    ds: xr.Dataset,
    variables: Sequence[str],
    context: str,
) -> None:
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"{context} requires variables missing from heat_budget.nc: "
            f"{', '.join(missing)}"
        )


def plot_result_figures(ds: xr.Dataset, results_dir: Path) -> None:
    require_variables(ds, RESULT_REQUIRED_VARIABLES, "Result plotting")
    results_dir.mkdir(parents=True, exist_ok=True)

    for smoothing_window in config.RESULT_SMOOTHING_WINDOWS:
        plot_results.plot_budget_terms_hourly(
            ds,
            smoothing_window=smoothing_window,
            plot_dir=str(results_dir),
        )
    plot_results.plot_budget_terms_day_bin(ds, plot_dir=str(results_dir))


def plot_dataset(request: PlotRequest) -> DatasetPaths:
    paths = io.dataset_paths(request.dataset, plots_root=request.plots_root)
    ds = io.load_heat_budget(
        request.dataset,
        start=request.time_start,
        end=request.time_end,
    )
    try:
        plot_result_figures(ds, paths.results_dir)

        if should_run_benchmark_diagnostics(
            request.dataset,
            request.benchmark_diagnostics,
        ):
            paths.diagnostics_dir.mkdir(parents=True, exist_ok=True)
            plot_diagnostics.plot_figure5_diagnostics(
                ds,
                plot_dir=str(paths.diagnostics_dir),
            )
    finally:
        ds.close()

    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        for dataset in io.available_datasets():
            print(dataset)
        return 0

    request = PlotRequest(
        dataset=args.dataset,
        time_start=args.time_start,
        time_end=args.time_end,
        benchmark_diagnostics=args.benchmark_diagnostics,
        plots_root=args.plots_root,
    )

    try:
        paths = plot_dataset(request)
    except Exception as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")

    print(f"Wrote result plots to {paths.results_dir}")
    if should_run_benchmark_diagnostics(
        request.dataset,
        request.benchmark_diagnostics,
    ):
        print(f"Wrote diagnostic plots to {paths.diagnostics_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
