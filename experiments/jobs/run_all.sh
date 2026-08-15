#!/bin/bash
# Submit the full experiment matrix to SLURM.
#
#   experiments/jobs/run_all.sh            # train, then evaluate once training finishes
#   experiments/jobs/run_all.sh --train    # training only
#   experiments/jobs/run_all.sh --eval     # evaluation only
#
# The matrix lives in experiments/jobs/experiments.conf — one line of Hydra
# overrides per run. Site settings come from the root .env — see .env.example.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

TRAIN_SCRIPT="${SCRIPT_DIR}/run_training.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluation.sh"
EXPERIMENTS="${SCRIPT_DIR}/experiments.conf"

require_slurm
mkdir -p "${PROJECT_ROOT}/logs"

MODE="all"
case "${1:-}" in
    --train) MODE="train" ;;
    --eval)  MODE="eval"  ;;
esac

read_experiments() {
    grep -v '^\s*#' "$EXPERIMENTS" | grep -v '^\s*$'
}

submit_training() {
    echo "Submitting the training matrix..." >&2
    build_slurm_args train

    local count=0 jid
    while IFS= read -r overrides; do
        jid=$(sbatch --parsable \
            --job-name=sm-train \
            --output="${PROJECT_ROOT}/logs/sm-train-%j.out" \
            --error="${PROJECT_ROOT}/logs/sm-train-%j.err" \
            --export=ALL \
            "${SLURM_ARGS[@]}" \
            "$TRAIN_SCRIPT" $overrides)
        echo "  [$jid] $overrides" >&2
        echo "$jid"
        ((++count))
    done < <(read_experiments)
    echo "${count} training jobs submitted." >&2
}

submit_eval() {
    echo "Submitting evaluation jobs..." >&2
    local dep_flag=()
    if [[ -n "${1:-}" ]]; then
        dep_flag=(--dependency="afterany:${1}")
    fi
    build_slurm_args eval

    local jid
    for eval_mode in synthetic nsynth; do
        jid=$(sbatch --parsable \
            "${dep_flag[@]}" \
            --job-name=sm-eval \
            --output="${PROJECT_ROOT}/logs/sm-eval-%j.out" \
            --error="${PROJECT_ROOT}/logs/sm-eval-%j.err" \
            --export=ALL \
            "${SLURM_ARGS[@]}" \
            "$EVAL_SCRIPT" --mode "$eval_mode")
        echo "  [$jid] --mode $eval_mode" >&2
    done
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
