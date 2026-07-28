#!/bin/bash
#PBS -N ehb_paper_plots
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -o /dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be supplied by the submission workflow}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT must be supplied by the submission workflow}"
PBS_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is not set}"

PROJECT_ROOT=$(realpath -e -- "${PROJECT_ROOT}")
PBS_WORKDIR=$(realpath -e -- "${PBS_WORKDIR}")
if [[ "${PBS_WORKDIR}" != "${PROJECT_ROOT}" ]]; then
    echo "[error] PBS_O_WORKDIR must be the project Git root: ${PROJECT_ROOT}" >&2
    exit 2
fi

ACTUAL_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "[error] checkout commit ${ACTUAL_COMMIT} does not match ${EXPECTED_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "[error] runtime checkout is dirty: ${PROJECT_ROOT}" >&2
    exit 2
fi

JOB_ID="${PBS_JOBID:-manual}"
EHB_CAMPAIGN_DATA_ROOT="${EHB_CAMPAIGN_DATA_ROOT:-${HOME:?HOME must be set}/eulerian-heat-budget/campaign-data}"
LOG_DIR="${LOG_DIR:-${EHB_CAMPAIGN_DATA_ROOT}/logs/paper-plots}"
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
PYTHON_EXECUTABLE=$(command -v python)

ehb_bool_enabled() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

DATASET="${DATASET:-pnw_bartusek_surface_700hPa}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT must point to the parent of the selected dataset folder}"
TIME_START="${TIME_START:-1941-06-01T00:00:00}"
TIME_END="${TIME_END:-1941-08-31T23:00:00}"
PLOTS_ROOT="${PLOTS_ROOT:-${EHB_CAMPAIGN_DATA_ROOT}/paper-plots}"
ENABLE_DIAGNOSTIC_PLOTS="${ENABLE_DIAGNOSTIC_PLOTS:-1}"
BENCHMARK_DIAGNOSTICS="${BENCHMARK_DIAGNOSTICS:-off}"

PLOT_ARGS=(
  --dataset "${DATASET}"
  --data-root "${DATA_ROOT}"
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
echo "[info] expected commit: ${EXPECTED_COMMIT}"
echo "[info] Python executable: ${PYTHON_EXECUTABLE}"
echo "[info] log file: ${LOGFILE}"
echo "[info] dataset: ${DATASET}"
echo "[info] data root: ${DATA_ROOT}"
echo "[info] time range: ${TIME_START} to ${TIME_END}"
echo "[info] plots root: ${PLOTS_ROOT}"
echo "[info] diagnostic plots: ${ENABLE_DIAGNOSTIC_PLOTS}"
echo "[info] benchmark diagnostics: ${BENCHMARK_DIAGNOSTICS}"

/usr/bin/time -v "${PYTHON_EXECUTABLE}" scripts/make_plots.py "${PLOT_ARGS[@]}"

echo "[info] $(date -Is) done"
