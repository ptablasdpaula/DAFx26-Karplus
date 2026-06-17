#!/bin/bash
# P+Audio (combined) pilot run with Torres SOT+MSS weighting (wSOT=1, wMSS=0.05,
# wP=1). Parameterized by model and data so all four variants share one script.
#
# Usage:  sbatch experiments/jobs/run_pilot.sh <model> <data>
#   <model> = ksa_time | ksa_freq
#   <data>  = synth | mix
# Examples:
#   sbatch experiments/jobs/run_pilot.sh ksa_time synth
#   sbatch experiments/jobs/run_pilot.sh ksa_freq mix
#SBATCH --job-name=sm-pilot
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
MODEL="${1:?usage: run_pilot.sh <ksa_time|ksa_freq> <synth|mix>}"
DATA="${2:?usage: run_pilot.sh <ksa_time|ksa_freq> <synth|mix>}"

PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE=offline                 # no committed W&B key; log locally
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "=== PILOT P+Audio | model=$MODEL data=$DATA | wSOT=1 wMSS=0.05 wP=1 ==="
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname) | $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

"$PY" experiments/train.py \
    data="$DATA" \
    detector=end_to_end \
    training=combined \
    model="$MODEL" \
    training.w_sot=1.0 \
    training.w_mss=0.05 \
    training.w_param=1.0

echo "=== Done: $(date) ==="
