#!/bin/bash
# Run from the root DAFx26-Karplus directory: ./experiments/jobs/run_all_evaluation.sh

set -e

JOB_SCRIPT="experiments/jobs/run_evaluation.sh"

echo "🔬 Submitting Evaluation Jobs to SLURM..."

sbatch $JOB_SCRIPT --mode synthetic
sbatch $JOB_SCRIPT --mode nsynth

echo "✅ Both evaluation jobs submitted!"