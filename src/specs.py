"""Small contracts used by the plotting-only workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import xarray as xr

from . import config

BenchmarkDiagnostics = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class PlotRequest:
    dataset: str = config.DEFAULT_DATASET
    data_root: Path | str = config.DATA_ROOT
    time_start: str | None = None
    time_end: str | None = None
    diagnostic_plots: bool = True
    benchmark_diagnostics: BenchmarkDiagnostics = "auto"
    plots_root: Path | str = config.PLOTS_ROOT
    stage1_file: Path | str | None = None
    temperature_climatology_file: Path | str | None = None
    tas_threshold_file: Path | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))
        object.__setattr__(self, "plots_root", Path(self.plots_root))
        names = ("stage1_file", "temperature_climatology_file", "tas_threshold_file")
        supplied = [getattr(self, name) is not None for name in names]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "Temperature context requires all three auxiliary files together."
            )
        for name in names:
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True)
class TemperatureContext:
    """Validated temperature references aligned exactly to selected budget hours."""

    series: xr.Dataset
    provenance: dict[str, Any]


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    heat_budget_path: Path
    run_info_path: Path
    results_dir: Path
    diagnostics_dir: Path
