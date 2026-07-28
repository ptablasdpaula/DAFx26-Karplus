#!/bin/bash
# run_all.sh — Submit training, evaluation, or both to SLURM.
#
# Usage:
#   ./experiments/jobs/run_all.sh              # train + eval (eval chains after training)
#   ./experiments/jobs/run_all.sh --train      # training only
#   ./experiments/jobs/run_all.sh --eval       # evaluation only
#
# Cluster-specific settings are read from experiments/jobs/.env if present.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_training.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluation.sh"
EXPERIMENTS="${SCRIPT_DIR}/experiments.conf"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

if ! command -v sbatch >/dev/null 2>&1; then
    echo "sbatch was not found. These job wrappers are SLURM-only." >&2
    echo "Install/use SLURM, or run experiments/train.py and experiments/evaluate.py directly." >&2
    exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

MODE="all"
case "${1:-}" in
    --train) MODE="train" ;;
    --eval)  MODE="eval"  ;;
esac

read_experiments() {
    grep -v '^\s*#' "$EXPERIMENTS" | grep -v '^\s*$'
}

append_if_set() {
    local -n target_args="$1"
    local flag="$2"
    local value="$3"
    if [[ -n "$value" ]]; then
        target_args+=("$flag" "$value")
    fi
}

slurm_common_args() {
    local array_name="$1"
    append_if_set "$array_name" "--account" "${SM_SLURM_ACCOUNT:-}"
    append_if_set "$array_name" "--partition" "${SM_SLURM_PARTITION:-}"
    append_if_set "$array_name" "--qos" "${SM_SLURM_QOS:-}"
    append_if_set "$array_name" "--gres" "${SM_SLURM_GRES:-gpu:1}"
    append_if_set "$array_name" "--gpus" "${SM_SLURM_GPUS:-}"
    append_if_set "$array_name" "--cpus-per-task" "${SM_SLURM_CPUS_PER_TASK:-4}"
    append_if_set "$array_name" "--ntasks" "${SM_SLURM_NTASKS:-1}"
}

slurm_extra_args() {
    local -n target_args="$1"
    local extra="$2"
    if [[ -n "$extra" ]]; then
        read -r -a parsed_extra <<< "$extra"
        target_args+=("${parsed_extra[@]}")
    fi
}

submit_training() {
    echo "Submitting training experiments with SLURM..." >&2
    local args=()
    slurm_common_args args
    append_if_set args "--mem" "${SM_SLURM_TRAIN_MEM:-32G}"
    append_if_set args "--time" "${SM_SLURM_TRAIN_TIME:-24:00:00}"
    append_if_set args "--signal" "${SM_SLURM_TRAIN_SIGNAL:-B:TERM@300}"
    slurm_extra_args args "${SM_SLURM_TRAIN_EXTRA_ARGS:-}"

    local count=0
    while IFS= read -r overrides; do
        jid=$(sbatch --parsable \
            --job-name=sm-train \
            --output="${PROJECT_ROOT}/logs/sm-train-%j.out" \
            --error="${PROJECT_ROOT}/logs/sm-train-%j.err" \
            "${args[@]}" \
            --export=ALL \
            "$TRAIN_SCRIPT" $overrides)
        echo "  [$jid] $overrides" >&2
        echo "$jid"
        ((++count))
    done < <(read_experiments)
    echo "${count} training jobs submitted." >&2
}

submit_eval() {
    echo "Submitting evaluation jobs with SLURM..." >&2
    local dep_flag=()
    if [[ -n "${1:-}" ]]; then
        dep_flag=(--dependency="afterany:${1}")
    fi

    local args=()
    slurm_common_args args
    append_if_set args "--mem" "${SM_SLURM_EVAL_MEM:-64G}"
    append_if_set args "--time" "${SM_SLURM_EVAL_TIME:-6:00:00}"
    slurm_extra_args args "${SM_SLURM_EVAL_EXTRA_ARGS:-}"

    for eval_mode in synthetic nsynth; do
        jid=$(sbatch --parsable \
            "${dep_flag[@]}" \
            --job-name=sm-eval \
            --output="${PROJECT_ROOT}/logs/sm-eval-%j.out" \
            --error="${PROJECT_ROOT}/logs/sm-eval-%j.err" \
            "${args[@]}" \
            --export=ALL \
            "$EVAL_SCRIPT" --mode "$eval_mode")
        echo "  [$jid] --mode $eval_mode" >&2
    done
    echo "Evaluation jobs submitted." >&2
}

case "$MODE" in
    train)
        submit_training > /dev/null
        ;;
    eval)
        submit_eval
        ;;
    all)
        mapfile -t JIDS < <(submit_training)
        DEP_STR=$(IFS=:; echo "${JIDS[*]}")
        echo "" >&2
        submit_eval "$DEP_STR"
        ;;
esac

echo "" >&2
echo "Monitor with: squeue -u \$USER" >&2
