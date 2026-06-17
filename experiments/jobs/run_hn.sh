#!/bin/bash
# Harmonics-plus-Noise baseline retraining with the Torres SOT+MSS weighting
# (wSOT=1, wMSS=0.05) used by all audio-trained models in the paper. External
# detectors (CREPE f0 + A-weighted loudness), Real data, spectral-only loss.
#
# Usage:  sbatch experiments/jobs/run_hn.sh <hn|hn_tcn>
#   sbatch experiments/jobs/run_hn.sh hn       # HpN  (original DDSP encoder)
#   sbatch experiments/jobs/run_hn.sh hn_tcn   # HpN+ (our TCN encoder)
#SBATCH --job-name=sm-hn
#SBATCH --output=logs/sm-hn-%x-%j.out
#SBATCH --error=logs/sm-hn-%x-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail
MODEL="${1:?usage: run_hn.sh <hn|hn_tcn>}"
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "=== HpN baseline | model=$MODEL data=real detector=external | wSOT=1 wMSS=0.05 ==="
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname) | $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

"$PY" experiments/train.py \
    data=real \
    detector=external \
    training=spectral_only \
    model="$MODEL" \
    training.w_sot=1.0 \
    training.w_mss=0.05

echo "=== Done: $(date) ==="
