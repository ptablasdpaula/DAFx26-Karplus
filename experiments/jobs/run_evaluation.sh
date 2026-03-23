#!/bin/bash
#SBATCH --job-name=sm-eval
#SBATCH --output=logs/sm-eval-%j.out
#SBATCH --error=logs/sm-eval-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00

set -euo pipefail

echo "=== Evaluation ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""

mkdir -p logs
cd /data/home/YOUR_USERNAME/DAFx26-Karplus

export PROJECT_ROOT=/data/home/YOUR_USERNAME/DAFx26-Karplus

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

pixi run -e cuda python experiments/evaluate.py "$@"

echo ""
echo "=== Done: $(date) ==="