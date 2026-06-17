#!/bin/bash
# Runs after the 4 pilot trainings finish: (1) upgrade the main env's torchlpc to
# the optimised compiled CUDA kernels, (2) run the full evaluation (synthetic +
# nsynth). Submit chained: sbatch --dependency=afterany:<jids> finalize_and_eval.sh
#SBATCH --job-name=finalize-eval
#SBATCH --output=logs/finalize-eval-%j.out
#SBATCH --error=logs/finalize-eval-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE=offline
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"
echo "=== finalize+eval on $(hostname) | $(date) ==="

# ── 1. Optimised CUDA (best-effort): try to ensure the compiled scan kernel, but
#       fall back to the numba implementation if the build fails. The eval runs
#       fine either way; the compiled kernel is only ~5% faster, so it must not
#       block the evaluation. ──
TLSO=$(ls "$PROJECT_ROOT"/.pixi/envs/cuda/lib/python*/site-packages/torchlpc/_C*.so 2>/dev/null | head -1 || true)
if [ -z "${TLSO:-}" ] || ! nm -D "$TLSO" 2>/dev/null | grep -q scan_cuda_wrapper; then
  echo "--- attempting torchlpc compiled CUDA rebuild (best-effort) ---"
  export CUDA_HOME="$PROJECT_ROOT/.pixi/envs/cuda"
  export TORCH_CUDA_ARCH_LIST="8.0"
  export MAX_JOBS=2
  "$PY" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir torchlpc==0.7.2 \
    || echo "WARN: torchlpc compiled rebuild failed; falling back to the numba implementation."
fi
"$PY" -c "import torchlpc; print('torchlpc:', 'compiled-CUDA' if getattr(torchlpc,'EXTENSION_LOADED',False) else 'numba-fallback')" || true

# ── 2. Back up the accepted result CSVs (eval overwrites them) ──
for m in synthetic nsynth; do
  f="experiments/evaluation/${m}_results.csv"
  [ -f "$f" ] && [ ! -f "experiments/evaluation/${m}_results.ACCEPTED.csv" ] && \
    cp "$f" "experiments/evaluation/${m}_results.ACCEPTED.csv" || true
done

# ── 3. Full evaluation (nsynth + synthetic) ──
echo "--- evaluate.py --mode all ---"
"$PY" experiments/evaluate.py --mode all

echo "=== Done: $(date) ==="
echo "Results: experiments/evaluation/{synthetic,nsynth}_results.csv"
