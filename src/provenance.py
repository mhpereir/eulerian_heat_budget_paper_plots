"""Generated, checksummed provenance for temperature-context publication runs."""

import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from . import io, plot_style


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize provenance value {type(value).__name__}")


def build_manifest(request, paths, budget, run_info, context):
    root = Path(__file__).resolve().parents[1]
    calculation_git = run_info.get("git", {})
    if (
        not isinstance(calculation_git, dict)
        or not re.fullmatch(r"[0-9a-f]{40}", str(calculation_git.get("commit", "")))
        or not calculation_git.get("branch")
        or calculation_git.get("dirty") is not False
    ):
        raise ValueError(
            "Temperature-context publication requires clean EHB branch and commit provenance."
        )

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "schema_version": 1,
        "execution": {
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "host": socket.getfqdn(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
        "request": asdict(request),
        "heat_budget": io.file_provenance(paths.heat_budget_path),
        "run_info": io.file_provenance(paths.run_info_path),
        "calculation": {
            "git": run_info.get("git"),
            "domain": run_info.get("request"),
            "run_root": run_info.get("root_dir"),
        },
        "renderer": {
            "root": str(root),
            "branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        },
        "environment": {
            "python_executable": sys.executable,
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "xarray",
                    "matplotlib",
                    "h5netcdf",
                    "seaborn",
                )
            },
        },
        "selection": {
            "start": str(budget.time.values[0]),
            "end": str(budget.time.values[-1]),
            "hourly_samples": budget.sizes["time"],
        },
        "temperature_context": context.provenance,
        "layout": {
            "width": plot_style.RESULT_WIDTH,
            "width_inches": plot_style.result_figsize()[0],
            "height_inches": plot_style.result_figsize()[1],
            "panel_height_ratios": plot_style.RESULT_HEIGHT_RATIOS,
            "font_size_pt": plot_style.SINGLE_COLUMN_FONT_SIZE_PT,
            "legend_font_size_pt": plot_style.SINGLE_COLUMN_LEGEND_FONT_SIZE_PT,
            "dpi": plot_style.DPI,
            "crop": False,
            "colors": {
                name: plot_style.COLORS[name]
                for name in (
                    "temperature",
                    "surface_temperature",
                    "temperature_tendency",
                    "advection",
                    "adiabatic",
                    "diabatic",
                )
            },
        },
    }


def write_manifest(manifest, paths):
    outputs = sorted(paths.results_dir.glob("*.png"))
    if paths.diagnostics_dir.is_dir():
        outputs.extend(sorted(paths.diagnostics_dir.glob("*.png")))
    manifest["outputs"] = [io.file_provenance(path) for path in outputs]
    path = paths.results_dir.parent / "plot_manifest.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=_json_value, allow_nan=False)
        handle.write("\n")
    return path
