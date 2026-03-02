#!/bin/bash
#SBATCH --job-name=grad_stage
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=experiments/logs/grad/%A_%a.out
#SBATCH --error=experiments/logs/grad/%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.tablasdepaula@qmul.ac.uk

# ── Stage mapping ──
STAGES=(cga_pluck cga_ks fga_pluck fga_ks)
STAGE=${STAGES[$SLURM_ARRAY_TASK_ID]}

# ── Shared results directory (all 4 tasks write here) ──
RESULTS_DIR="experiments/outputs/grad/${SLURM_ARRAY_JOB_ID}"

echo "═══════════════════════════════════════════"
echo "Job ID:     $SLURM_JOB_ID  (array $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID)"
echo "Stage:      $STAGE"
echo "Results:    $RESULTS_DIR"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Started:    $(date)"
echo "═══════════════════════════════════════════"

# ── Environment ──
PROJECT_DIR="$HOME/DAFx26-Karplus"
cd "$PROJECT_DIR" || exit 1
eval "$(pixi shell-hook)"

# Project root on PYTHONPATH → `import paths` and `from synths...` just work
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RESULTS_DIR" experiments/logs/grad

export HYDRA_FULL_ERROR=1

# Verify CUDA
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# ── Run stage ──
python experiments/scripts/grad/gradient_analysis_stage.py \
    +stage="$STAGE" \
    +results_dir="$RESULTS_DIR" \
    compute.device=cuda \
    hydra.run.dir="$RESULTS_DIR/hydra_${STAGE}" \
    "$@"

echo "═══════════════════════════════════════════"
echo "Stage $STAGE finished: $(date)"
echo "═══════════════════════════════════════════"