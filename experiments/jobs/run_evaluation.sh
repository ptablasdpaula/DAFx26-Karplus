#!/bin/bash
# Submit one evaluation run to SLURM.
#
#   experiments/jobs/run_evaluation.sh --mode all
#
# Arguments are passed straight through to experiments/evaluate.py.
# Site settings come from the root .env — see .env.example.
#
# Requires checkpoints in experiments/checkpoints/. Fetch them with:
#   scripts/fetch_checkpoints.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    require_slurm
    mkdir -p "${PROJECT_ROOT}/logs"
    build_slurm_args eval

    sbatch --parsable \
        --job-name=sm-eval \
        --output="${PROJECT_ROOT}/logs/sm-eval-%j.out" \
        --error="${PROJECT_ROOT}/logs/sm-eval-%j.err" \
        --export=ALL \
        "${SLURM_ARGS[@]}" \
        "$0" "$@"
    exit 0
fi

echo "=== Evaluation ==="
job_setup

pixi run -e "${SM_PIXI_ENV:-cuda}" python experiments/evaluate.py "$@"

echo ""
echo "=== Done: $(date) ==="
