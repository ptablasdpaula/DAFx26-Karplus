#!/bin/bash
# Canonical environment build for the Apocrita GPU cluster.
#
# Why a job (not `pixi install` on the login node)?
#   * Login nodes cap per-user memory and kill the C++ compiler while building
#     philtorch's template-heavy csrc/*.cpp ("Killed signal terminated cc1plus").
#   * The pixi/rattler cache corrupts on the shared gpfs filesystem (incomplete
#     extraction -> "could not open source file"); we put it on node-local /tmp.
#   * A GPU must be present so philtorch/torchlpc compile their CUDA extensions
#     and so the smoke test can validate forward/backward on CUDA.
#
# Usage:  sbatch experiments/jobs/build_env.sh
#SBATCH --job-name=karplus-build-env
#SBATCH --output=logs/build-env-%j.out
#SBATCH --error=logs/build-env-%j.err
#SBATCH --account=pilot_andrena
#SBATCH --partition=andrena
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00

set -euo pipefail
PROJECT_ROOT=/data/home/acw794/DAFx26-Karplus
cd "$PROJECT_ROOT"
mkdir -p logs
echo "=== Build cuda env on $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true

# Node-local cache + compute-node memory + capped parallel compiles.
export PIXI_CACHE_DIR="/tmp/pixi-cache-${SLURM_JOB_ID:-$$}"
export CONDA_OVERRIDE_CUDA="12.9"
export MAX_JOBS=2
mkdir -p "$PIXI_CACHE_DIR"

echo "--- pixi install -e cuda ---"
pixi install -e cuda

# torchlpc 0.7.2 ships compiled CUDA kernels (csrc/cuda/{linear_recurrence,lpc}.cu)
# that register torchlpc::scan/lpc for the CUDA backend, but setup.py only compiles
# them when `use_cuda = torch.cuda.is_available() and CUDA_HOME is not None`. pixi's
# build subprocess leaves CUDA_HOME unset, so the wheel pixi builds is CPU-only ->
# torchlpc::scan has no CUDA backend and the time-domain synth crashes on GPU.
# Rebuild torchlpc from source with CUDA_HOME set so the optimised compiled CUDA
# scan/lpc kernels are included (this is the fast path; ~5% faster than the numba
# fallback and the intended implementation).
echo "--- rebuilding torchlpc with compiled CUDA kernels ---"
export CUDA_HOME="$PROJECT_ROOT/.pixi/envs/cuda"
export TORCH_CUDA_ARCH_LIST="8.0"   # A100 = sm_80
PY="$PROJECT_ROOT/.pixi/envs/cuda/bin/python"
"$PY" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir torchlpc==0.7.2
echo "--- verify compiled CUDA scan kernel is present ---"
TLSO=$(ls "$PROJECT_ROOT"/.pixi/envs/cuda/lib/python*/site-packages/torchlpc/_C*.so | head -1)
nm -D "$TLSO" | grep -q scan_cuda_wrapper || { echo "ERROR: torchlpc _C lacks scan_cuda kernel"; exit 1; }
"$PY" -c "from torchlpc import EXTENSION_LOADED; assert EXTENSION_LOADED is True; print('torchlpc EXTENSION_LOADED = True (compiled CUDA path) OK')"

echo "--- imports ---"
"$PY" -c "import torch,philtorch,philtorch.lpv,torchlpc,sot,lightning,hydra,torchaudio,transformers; print('philtorch', philtorch.__version__, '| torch', torch.__version__, torch.version.cuda)"

echo "--- GPU smoke test (tKSA + fKSA fwd/bwd on CUDA) ---"
PYTHONPATH="$PROJECT_ROOT" .pixi/envs/cuda/bin/python experiments/jobs/gpu_smoke.py

echo "=== Done: $(date) ==="
