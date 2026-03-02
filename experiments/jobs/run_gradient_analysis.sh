#!/bin/bash
#SBATCH --job-name=grad_analysis
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12          # 12 cores per GPU (Andrena requirement)
#SBATCH --mem-per-cpu=7500          # 7.5G per core = 90G total
#SBATCH --gres=gpu:1                # 1× A100
#SBATCH --time=04:00:00             # 4 hours (generous for n_points=50)
#SBATCH --output=experiments/logs/grad_%j.out
#SBATCH --error=experiments/logs/grad_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.tablasdepaula@qmul.ac.uk

# ── Print job info ──
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $CUDA_VISIBLE_DEVICES"
echo "Started:    $(date)"
echo "─────────────────────────────────"

# ── Environment ──
PROJECT_DIR="$HOME/DAFx26-Karplus"
cd "$PROJECT_DIR" || exit 1

# Activate pixi environment
eval "$(pixi shell-hook)"

# Verify CUDA is visible
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: CUDA not available, falling back to CPU')
"

# ── Run ──
# Hydra outputs go to experiments/outputs/YYYY-MM-DD/HH-MM-SS/
SCRATCH="/gpfs/scratch/$USER/gradient_analysis"
mkdir -p "$SCRATCH" experiments/logs

python experiments/scripts/gradient_analysis.py \
    compute.device=cuda \
    "$@"

# ── Copy outputs to scratch for easy access ──
LATEST_OUTPUT=$(find experiments/outputs -maxdepth 2 -mindepth 2 -type d | sort | tail -1)
if [ -n "$LATEST_OUTPUT" ]; then
    cp -r "$LATEST_OUTPUT" "$SCRATCH/job_${SLURM_JOB_ID}"
    echo "Results copied to: $SCRATCH/job_${SLURM_JOB_ID}"
fi

echo "─────────────────────────────────"
echo "Finished:   $(date)"