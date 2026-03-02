#!/bin/bash
# Launch all gradient analysis jobs with proper dependencies.
# Usage: bash experiments/jobs/grad/launch_all.sh [extra hydra overrides...]
#
# Example:
#   bash experiments/jobs/grad/launch_all.sh grid.n_points=1000

cd "$(dirname "$0")/../../.." || exit 1

mkdir -p experiments/logs/grad experiments/outputs/grad

echo "Submitting 4-stage gradient analysis array…"
ARRAY_ID=$(sbatch --parsable experiments/jobs/grad/run_gradient_array.sh "$@")
echo "  Array job: $ARRAY_ID  (tasks 0-3)"

echo "Submitting combine job (will run after array completes)…"
COMBINE_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_ID} \
    experiments/jobs/grad/run_combine.sh "$ARRAY_ID")
echo "  Combine job: $COMBINE_ID"

echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f experiments/logs/grad/${ARRAY_ID}_*.out"
echo ""
echo "Results will be in:"
echo "  experiments/outputs/grad/${ARRAY_ID}/"