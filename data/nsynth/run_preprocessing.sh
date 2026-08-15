#!/bin/bash
# Submit NSynth guitar-acoustic preprocessing to SLURM.
#
#   data/nsynth/run_preprocessing.sh
#
# Runs onset, f0, and loudness detection over the downloaded splits and writes
# preprocessed/ inside each of them. Set NSYNTH_DIR in the root .env first; see
# .env.example.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../experiments/jobs" && pwd)/common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    require_slurm
    mkdir -p "${PROJECT_ROOT}/logs"
    build_slurm_args preprocess

    sbatch --parsable \
        --job-name=sm-preprocess \
        --output="${PROJECT_ROOT}/logs/sm-preprocess-%j.out" \
        --error="${PROJECT_ROOT}/logs/sm-preprocess-%j.err" \
        --export=ALL \
        "${SLURM_ARGS[@]}" \
        "$0" "$@"
    exit 0
fi

echo "=== Preprocessing ==="
job_setup
echo "NSynth:    ${NSYNTH_DIR:-${PROJECT_ROOT}/data/nsynth}"
echo ""

pixi run -e "${SM_PIXI_ENV:-cuda}" python data/nsynth/preprocess_subset.py "$@"

echo ""
echo "=== Done: $(date) ==="
