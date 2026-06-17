#!/bin/bash
#SBATCH --job-name=sm-pilot-eval
#SBATCH --output=logs/sm-pilot-eval-%j.out
#SBATCH --error=logs/sm-pilot-eval-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00

set -euo pipefail

echo "=== PILOT Evaluation: synthetic mode ==="
echo "Job ID:    ${SLURM_JOB_ID:-NA}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""

PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
export PROJECT_ROOT
mkdir -p logs

export WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

# Synthetic eval evaluates the fixed tag list and picks the LATEST checkpoint per
# tag, so the newly-trained pilot tKSA_E2E_synth row reflects the pilot weights.
# Accepted baselines are preserved in experiments/evaluation/*.ACCEPTED.csv.
"$PY" experiments/evaluate.py --mode synthetic

echo ""
echo "=== Done: $(date) ==="
