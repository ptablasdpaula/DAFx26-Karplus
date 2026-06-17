#!/bin/bash
# Preprocess the NSynth guitar/acoustic subset (CREPE f0 + onsets + loudness) for
# all splits. Data lives on /gpfs/scratch via the data/nsynth/{split} symlinks.
#SBATCH --job-name=nsynth-preprocess
#SBATCH --output=logs/nsynth-preprocess-%j.out
#SBATCH --error=logs/nsynth-preprocess-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

echo "=== NSynth preprocess on $(hostname) | $(date) ==="
"$PY" -c "import torch,torchcrepe; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

"$PY" data/nsynth/preprocess_subset.py

echo "=== Done: $(date) ==="
