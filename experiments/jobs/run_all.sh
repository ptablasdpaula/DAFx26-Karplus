#!/bin/bash
# run_all.sh — Submit training, evaluation, or both.
#
# Usage:
#   ./experiments/jobs/run_all.sh              # train + eval (eval chains after training)
#   ./experiments/jobs/run_all.sh --train      # training only
#   ./experiments/jobs/run_all.sh --eval       # evaluation only
#
# Run from the root DAFx26-Karplus directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_training.sh"
EVAL_SCRIPT="${SCRIPT_DIR}/run_evaluation.sh"
EXPERIMENTS="${SCRIPT_DIR}/experiments.conf"

# ── Parse mode ──
MODE="all"
case "${1:-}" in
    --train) MODE="train" ;;
    --eval)  MODE="eval"  ;;
esac

# ── Read experiment definitions (skip comments and blanks) ──
read_experiments() {
    grep -v '^\s*#' "$EXPERIMENTS" | grep -v '^\s*$'
}

# ── Submit training jobs ──
# Prints status to stderr, echoes bare job IDs to stdout.
submit_training() {
    echo "🚀 Submitting training experiments..." >&2
    local count=0
    while IFS= read -r overrides; do
        jid=$(sbatch --parsable "$TRAIN_SCRIPT" $overrides)
        echo "  [$jid] $overrides" >&2
        echo "$jid"   # stdout — captured by mapfile
        ((++count))
    done < <(read_experiments)
    echo "✅ ${count} training jobs submitted." >&2
}

# ── Submit evaluation jobs ──
submit_eval() {
    local dep_flag=""
    if [[ -n "${1:-}" ]]; then
        dep_flag="--dependency=afterany:${1}"
    fi
    echo "🔬 Submitting evaluation jobs...${dep_flag:+ (chained)}" >&2
    for eval_mode in synthetic nsynth; do
        jid=$(sbatch --parsable $dep_flag "$EVAL_SCRIPT" --mode "$eval_mode")
        echo "  [$jid] --mode $eval_mode" >&2
    done
    echo "✅ Evaluation jobs submitted." >&2
}

# ── Orchestrate ──
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