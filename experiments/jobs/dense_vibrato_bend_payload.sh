#!/bin/bash
# Payload for the dense vibrato/bend preflight, sweep array, and aggregation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Slurm copies batch scripts into its spool, so BASH_SOURCE no longer locates
# the checkout inside a job.  sbatch records the caller's repository cwd in
# SLURM_SUBMIT_DIR; the explicit override also keeps local invocation useful.
if [[ -n "${DENSE_KS_PROJECT_ROOT:-}" ]]; then
    PROJECT_ROOT="$DENSE_KS_PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
MODE="${1:?expected preflight, sweep, or aggregate}"
OUTPUT="${DENSE_KS_OUTPUT:-${PROJECT_ROOT}/experiments/outputs/dense-vibrato-bend-ot}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUTPUT"
export PYTHONPATH="$PROJECT_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "mode=${MODE} job=${SLURM_JOB_ID:-local} array=${SLURM_ARRAY_TASK_ID:-none}"
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date --iso-8601=seconds)"

case "$MODE" in
    preflight)
        nvidia-smi
        # Build on the sm_80 node, then reinstall the compatibility pin after
        # PhilTorch so its compiled extension cannot be displaced by resolution.
        pixi install -e cuda
        pixi run -e cuda python -m pip install \
            --force-reinstall --no-deps --no-build-isolation torchlpc==0.7.2
        pixi run -e cuda python experiments/dense_vibrato_bend.py \
            --output "$OUTPUT" preflight
        ;;
    sweep)
        : "${SLURM_ARRAY_TASK_ID:?sweep mode requires a Slurm array task id}"
        pixi run -e cuda python experiments/dense_vibrato_bend.py \
            --output "$OUTPUT" sweep --task-index "$SLURM_ARRAY_TASK_ID" --device cuda
        ;;
    aggregate)
        pixi run -e cuda python experiments/dense_vibrato_bend.py \
            --output "$OUTPUT" aggregate
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

echo "finish=$(date --iso-8601=seconds)"
