"""Small contracts used by the plotting-only workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import config

BenchmarkDiagnostics = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class PlotRequest:
    dataset: str = config.DEFAULT_DATASET
    time_start: str | None = None
    time_end: str | None = None
    benchmark_diagnostics: BenchmarkDiagnostics = "auto"
    plots_root: Path | str = config.PLOTS_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "plots_root", Path(self.plots_root))


@dataclass(frozen=True)
class DatasetPaths:
    data_dir: Path
    heat_budget_path: Path
    run_info_path: Path
    results_dir: Path
    diagnostics_dir: Path
