#!/bin/bash
# Submit or run one SLURM training payload.
#
# Submit one training job:
#   experiments/jobs/run_training.sh data=synth detector=end_to_end training=combined model=ksa_freq
#
# Cluster-specific settings are read from experiments/jobs/.env if present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

append_if_set() {
    local -n target_args="$1"
    local flag="$2"
    local value="$3"
    if [[ -n "$value" ]]; then
        target_args+=("$flag" "$value")
    fi
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "sbatch was not found. This job wrapper is SLURM-only." >&2
        exit 1
    fi

    mkdir -p "${PROJECT_ROOT}/logs"

    args=()
    append_if_set args "--account" "${SM_SLURM_ACCOUNT:-}"
    append_if_set args "--partition" "${SM_SLURM_PARTITION:-}"
    append_if_set args "--qos" "${SM_SLURM_QOS:-}"
    append_if_set args "--gres" "${SM_SLURM_GRES:-gpu:1}"
    append_if_set args "--gpus" "${SM_SLURM_GPUS:-}"
    append_if_set args "--cpus-per-task" "${SM_SLURM_CPUS_PER_TASK:-4}"
    append_if_set args "--ntasks" "${SM_SLURM_NTASKS:-1}"
    append_if_set args "--mem" "${SM_SLURM_TRAIN_MEM:-32G}"
    append_if_set args "--time" "${SM_SLURM_TRAIN_TIME:-24:00:00}"
    append_if_set args "--signal" "${SM_SLURM_TRAIN_SIGNAL:-B:TERM@300}"
    if [[ -n "${SM_SLURM_TRAIN_EXTRA_ARGS:-}" ]]; then
        read -r -a extra_args <<< "$SM_SLURM_TRAIN_EXTRA_ARGS"
        args+=("${extra_args[@]}")
    fi

    sbatch --parsable \
        --job-name=sm-train \
        --output="${PROJECT_ROOT}/logs/sm-train-%j.out" \
        --error="${PROJECT_ROOT}/logs/sm-train-%j.err" \
        "${args[@]}" \
        "$0" "$@"
    exit 0
fi

echo "=== Training! ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:     $(date)"
echo ""

mkdir -p logs
cd "$PROJECT_ROOT"

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi

PIXI_ENV="${SM_PIXI_ENV:-cuda}"
pixi run -e "$PIXI_ENV" python experiments/train.py "$@"

echo ""
echo "=== Done: $(date) ==="
