#!/bin/bash
#SBATCH --job-name=sm-train
#SBATCH --output=logs/sm-train-%j.out
#SBATCH --error=logs/sm-train-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "=== Training! ==="
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

export WANDB_API_KEY="***REDACTED_WANDB_KEY***"

pixi run -e cuda python experiments/train.py "$@"

echo ""
echo "=== Done: $(date) ==="