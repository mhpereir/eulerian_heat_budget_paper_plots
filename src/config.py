"""Project-level constants for static publication plotting."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
PLOTS_ROOT = PROJECT_ROOT / "plots"

DEFAULT_DATASET = "ocean_test_full_atmosphere"
RESULT_SMOOTHING_WINDOWS = (1, 24)
FULL_ATMOSPHERE_SUFFIX = "_full_atmosphere"
