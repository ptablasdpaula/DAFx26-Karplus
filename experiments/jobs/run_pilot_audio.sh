#!/bin/bash
# Audio-Only (spectral_only, external detectors) pilot run with the Torres audio
# weighting (wSOT=1, wMSS=0.05; no P-loss). Parameterized by model and data.
#
# Usage:  sbatch experiments/jobs/run_pilot_audio.sh <model> <data>
#   <model> = ksa_time | ksa_freq
#   <data>  = synth | real
# Examples:
#   sbatch experiments/jobs/run_pilot_audio.sh ksa_time synth
#   sbatch experiments/jobs/run_pilot_audio.sh ksa_freq real
#SBATCH --job-name=sm-pilot-audio
#SBATCH --output=logs/sm-pilot-%x-%j.out
#SBATCH --error=logs/sm-pilot-%x-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail
MODEL="${1:?usage: run_pilot_audio.sh <ksa_time|ksa_freq> <synth|real>}"
DATA="${2:?usage: run_pilot_audio.sh <ksa_time|ksa_freq> <synth|real>}"

PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "=== PILOT Audio-Only | model=$MODEL data=$DATA | wSOT=1 wMSS=0.05 (spectral_only) ==="
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname) | $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

"$PY" experiments/train.py \
    data="$DATA" \
    detector=external \
    training=spectral_only \
    model="$MODEL" \
    training.w_sot=1.0 \
    training.w_mss=0.05

echo "=== Done: $(date) ==="
