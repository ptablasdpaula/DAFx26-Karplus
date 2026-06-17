#!/bin/bash
# Chunked evaluation for gpushort (<=1h): evaluate a subset of tags for one mode
# into a distinct CSV, so several chunks can run in parallel without clobbering
# each other. Uses the already-installed (numba-fallback) torchlpc; no rebuild.
#   sbatch experiments/jobs/run_eval_chunk.sh <synthetic|nsynth> <tag1,tag2,...> <out_csv>
#SBATCH --job-name=evalchunk
#SBATCH --output=logs/evalchunk-%x-%j.out
#SBATCH --error=logs/evalchunk-%x-%j.err
#SBATCH --account=pilot
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:55:00

set -uo pipefail
MODE="${1:?usage: run_eval_chunk.sh <synthetic|nsynth> <tags> <out_csv>}"
TAGS="${2:?tags}"
OUTCSV="${3:?out_csv}"
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT PYTHONPATH="$PROJECT_ROOT" WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"
echo "=== evalchunk mode=$MODE on $(hostname) | $(date) ==="
echo "tags=$TAGS -> $OUTCSV"
"$PY" experiments/evaluate.py --mode "$MODE" --tags "$TAGS" --out_csv "$OUTCSV"
echo "=== evalchunk done $(date) ==="
