#!/bin/bash
# P+Audio (combined, Torres ratio: wSOT=1, wMSS=0.05, wP=1) trained from scratch
# exactly like the joint pilots, but with the AUDIO loss's gradient stopped to the
# onset and/or f0 heads (P-loss still trains them). Tests whether restricting the
# audio loss to the continuous timbre params helps.
#
# Usage:  sbatch experiments/jobs/run_ablate.sh <model> <data> <stopgrad>
#   <model>    = ksa_time | ksa_freq
#   <data>     = synth | mix
#   <stopgrad> = onset | f0 | both
#SBATCH --job-name=sm-ablate
#SBATCH --output=logs/sm-ablate-%x-%j.out
#SBATCH --error=logs/sm-ablate-%x-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300

set -euo pipefail
MODEL="${1:?usage: run_ablate.sh <ksa_time|ksa_freq> <synth|mix> <onset|f0|both>}"
DATA="${2:?usage: run_ablate.sh <ksa_time|ksa_freq> <synth|mix> <onset|f0|both>}"
SG="${3:?usage: run_ablate.sh <ksa_time|ksa_freq> <synth|mix> <onset|f0|both>}"

PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "=== ABLATE P+Audio | model=$MODEL data=$DATA stopgrad=$SG | wSOT=1 wMSS=0.05 wP=1 ==="
echo "Job ${SLURM_JOB_ID:-NA} on $(hostname) | $(date)"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

"$PY" experiments/train.py \
    data="$DATA" \
    detector=end_to_end \
    training=combined \
    model="$MODEL" \
    training.w_sot=1.0 \
    training.w_mss=0.05 \
    training.w_param=1.0 \
    +stopgrad="$SG"

echo "=== Done: $(date) ==="
