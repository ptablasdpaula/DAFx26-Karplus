#!/bin/bash
# Render the stop-gradient ablation prediction audio for the online supplement.
# Uses --render_only (skips the perceptual KAD models), so on a GPU node with the
# compiled torchlpc CUDA kernel the recursive tKSA renders in minutes.
#   sbatch experiments/jobs/run_render_ablations.sh
#SBATCH --job-name=render-abl
#SBATCH --output=logs/render-abl-%j.out
#SBATCH --error=logs/render-abl-%j.err
#SBATCH --account=pilot
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:55:00

set -uo pipefail
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT" WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"

SYN="tKSA_E2E_synth_sgOnset,tKSA_E2E_synth_sgF0,tKSA_E2E_synth_sgBoth,fKSA_E2E_synth_sgOnset,fKSA_E2E_synth_sgF0,fKSA_E2E_synth_sgBoth"
NS="tKSA_E2E_mix_sgOnset,tKSA_E2E_mix_sgF0,tKSA_E2E_mix_sgBoth,fKSA_E2E_mix_sgOnset,fKSA_E2E_mix_sgF0,fKSA_E2E_mix_sgBoth"

echo "=== render ablations on $(hostname) | $(date) ==="
"$PY" -c "import torch,torchlpc; print('cuda', torch.cuda.is_available(), '| torchlpc', 'compiled' if getattr(torchlpc,'EXTENSION_LOADED',False) else 'numba')"
"$PY" experiments/evaluate.py --mode synthetic --tags "$SYN" --render_only --out_csv /tmp/r_syn.csv
"$PY" experiments/evaluate.py --mode nsynth    --tags "$NS"  --render_only --out_csv /tmp/r_ns.csv
echo "=== done $(date) ==="
