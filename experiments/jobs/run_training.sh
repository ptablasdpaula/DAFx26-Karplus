#!/bin/bash
# Submit one training run to SLURM.
#
#   experiments/jobs/run_training.sh data=synth detector=end_to_end training=combined model=ksa_freq
#
# Arguments are Hydra overrides, passed straight through to experiments/train.py.
# Site settings come from the root .env — see .env.example.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    require_slurm
    mkdir -p "${PROJECT_ROOT}/logs"
    build_slurm_args train

    sbatch --parsable \
        --job-name=sm-train \
        --output="${PROJECT_ROOT}/logs/sm-train-%j.out" \
        --error="${PROJECT_ROOT}/logs/sm-train-%j.err" \
        --export=ALL \
        "${SLURM_ARGS[@]}" \
        "$0" "$@"
    exit 0
fi

echo "=== Training ==="
job_setup

pixi run -e "${SM_PIXI_ENV:-cuda}" python experiments/train.py "$@"

echo ""
echo "=== Done: $(date) ==="
