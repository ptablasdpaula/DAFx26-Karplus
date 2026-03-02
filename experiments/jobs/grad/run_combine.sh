#!/bin/bash
#SBATCH --job-name=grad_combine
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH -n 12
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/logs/grad/combine_%j.out
#SBATCH --error=experiments/logs/grad/combine_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=p.tablasdepaula@qmul.ac.uk

# ── Usage ──
# Submit AFTER the array job completes:
#   ARRAY_ID=$(sbatch --parsable experiments/jobs/grad/run_gradient_array.sh)
#   sbatch --dependency=afterok:$ARRAY_ID experiments/jobs/grad/run_combine.sh $ARRAY_ID
#
# Or use the launch helper:
#   bash experiments/jobs/grad/launch_all.sh

if [ -z "$1" ]; then
    echo "ERROR: pass the array job ID as first argument"
    echo "Usage: sbatch run_combine.sh <ARRAY_JOB_ID>"
    exit 1
fi

ARRAY_JOB_ID="$1"
RESULTS_DIR="experiments/outputs/grad/${ARRAY_JOB_ID}"

echo "═══════════════════════════════════════════"
echo "Combine job: $SLURM_JOB_ID"
echo "Array ID:    $ARRAY_JOB_ID"
echo "Results:     $RESULTS_DIR"
echo "Started:     $(date)"
echo "═══════════════════════════════════════════"

PROJECT_DIR="$HOME/DAFx26-Karplus"
cd "$PROJECT_DIR" || exit 1
eval "$(pixi shell-hook)"

# Project root on PYTHONPATH → `import paths` and `from synths...` just work
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ── Check all 4 .npz files exist ──
MISSING=0
for f in cga_pluck.npz cga_ks.npz fga_pluck.npz fga_ks.npz; do
    if [ ! -f "$RESULTS_DIR/$f" ]; then
        echo "ERROR: missing $RESULTS_DIR/$f"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    echo "Some stages did not complete. Check array job logs."
    exit 1
fi

# ── Combine ──
python experiments/scripts/grad/combine_gradient_results.py \
    "$RESULTS_DIR" \
    --output-dir "$RESULTS_DIR"

# ── Copy to scratch ──
SCRATCH="/gpfs/scratch/$USER/gradient_analysis"
mkdir -p "$SCRATCH"
cp -r "$RESULTS_DIR" "$SCRATCH/job_${ARRAY_JOB_ID}"
echo "Results copied to: $SCRATCH/job_${ARRAY_JOB_ID}"

echo "═══════════════════════════════════════════"
echo "Combine finished: $(date)"
echo "═══════════════════════════════════════════"