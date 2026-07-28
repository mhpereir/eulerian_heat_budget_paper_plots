#!/bin/bash
#PBS -N ehb_paper_plots
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/mhpereir/eulerian_heat_budget_paper_plots}"
JOB_ID="${PBS_JOBID:-manual}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
LOGFILE="${LOG_DIR}/${JOB_ID}_EHB_paper_plots.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOGFILE}") 2>&1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/mhpereir/miniconda3}"
source "${MAMBA_ROOT_PREFIX}/etc/profile.d/mamba.sh"
mamba activate "${EHB_PLOTS_CONDA_ENV:-dev_env}"

ehb_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET="${DATASET:-pnw_bartusek_surface_700hPa}"
TIME_START="${TIME_START:-1941-06-01T00:00:00}"
TIME_END="${TIME_END:-1941-09-01T00:00:00}"
PLOTS_ROOT="${PLOTS_ROOT:-${PROJECT_ROOT}/plots}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
BENCHMARK_DIAGNOSTICS="${BENCHMARK_DIAGNOSTICS:-off}"

PLOT_ARGS=(
  --dataset "${DATASET}"
  --start "${TIME_START}"
  --end "${TIME_END}"
  --plots-root "${PLOTS_ROOT}"
  --benchmark-diagnostics "${BENCHMARK_DIAGNOSTICS}"
)

if ehb_bool_enabled "${ENABLE_DIAGNOSTIC_PLOTS}"; then
  PLOT_ARGS+=(--diagnostic-plots)
else
  PLOT_ARGS+=(--no-diagnostic-plots)
fi

cd "${PROJECT_ROOT}"

echo "[info] $(date -Is) starting publication plotting on host $(hostname)"
echo "[info] repo root: ${PROJECT_ROOT}"
echo "[info] log file: ${LOGFILE}"
echo "[info] dataset: ${DATASET}"
echo "[info] time range: ${TIME_START} to ${TIME_END}"
echo "[info] plots root: ${PLOTS_ROOT}"
echo "[info] diagnostic plots: ${ENABLE_DIAGNOSTIC_PLOTS}"
echo "[info] benchmark diagnostics: ${BENCHMARK_DIAGNOSTICS}"

/usr/bin/time -v python scripts/make_plots.py "${PLOT_ARGS[@]}"

echo "[info] $(date -Is) done"
