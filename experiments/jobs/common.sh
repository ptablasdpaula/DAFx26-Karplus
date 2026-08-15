#!/bin/bash
# Shared plumbing for the job wrappers. Sourced, never run directly.
#
# Everything site-specific comes from the root .env — see .env.example.
# Real environment variables win over .env, so a value exported in your shell
# (or inherited through `sbatch --export=ALL`) is never clobbered by the file.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"

if [[ -f "$ENV_FILE" ]]; then
    while IFS= read -r _line; do
        [[ "$_line" =~ ^[[:space:]]*# ]] && continue
        [[ "$_line" =~ ^[[:space:]]*$ ]] && continue
        [[ "$_line" != *=* ]] && continue
        _key="${_line%%=*}"
        _key="${_key#"${_key%%[![:space:]]*}"}"
        _key="${_key%"${_key##*[![:space:]]}"}"
        # Only set what the shell has not already defined.
        if [[ -z "${!_key:-}" ]]; then
            set -a
            eval "$_line"
            set +a
        fi
    done < "$ENV_FILE"
    unset _line _key
fi

require_slurm() {
    if ! command -v sbatch >/dev/null 2>&1; then
        cat >&2 <<'EOF'
sbatch was not found. These job wrappers are SLURM-only.

To run without a scheduler, invoke the entry points directly on a CUDA host:
  pixi run -e cuda python experiments/train.py data=synth model=ksa_time
  pixi run -e cuda python experiments/evaluate.py --mode all
EOF
        exit 1
    fi
}

# Builds SLURM_ARGS for one job kind: "train", "eval", or "preprocess".
# Blank .env values are skipped entirely, so an unused flag is never passed.
build_slurm_args() {
    local kind="$1"
    SLURM_ARGS=()

    _add() { [[ -n "$2" ]] && SLURM_ARGS+=("$1" "$2"); }

    _add --account       "${SM_SLURM_ACCOUNT:-}"
    _add --partition     "${SM_SLURM_PARTITION:-}"
    _add --qos           "${SM_SLURM_QOS:-}"
    _add --gres          "${SM_SLURM_GRES-gpu:1}"
    _add --gpus          "${SM_SLURM_GPUS:-}"
    _add --nodes         "${SM_SLURM_NODES:-}"
    _add --constraint    "${SM_SLURM_CONSTRAINT:-}"
    _add --ntasks        "${SM_SLURM_NTASKS:-1}"
    _add --cpus-per-task "${SM_SLURM_CPUS_PER_TASK:-4}"

    local extra=""
    case "$kind" in
        train)
            _add --mem    "${SM_SLURM_TRAIN_MEM:-32G}"
            _add --time   "${SM_SLURM_TRAIN_TIME:-24:00:00}"
            _add --signal "${SM_SLURM_TRAIN_SIGNAL:-B:TERM@300}"
            extra="${SM_SLURM_TRAIN_EXTRA_ARGS:-}"
            ;;
        eval|preprocess)
            _add --mem  "${SM_SLURM_EVAL_MEM:-64G}"
            _add --time "${SM_SLURM_EVAL_TIME:-6:00:00}"
            extra="${SM_SLURM_EVAL_EXTRA_ARGS:-}"
            ;;
    esac

    if [[ -n "$extra" ]]; then
        local parsed
        read -r -a parsed <<< "$extra"
        SLURM_ARGS+=("${parsed[@]}")
    fi
}

# Runs inside the allocation, before python starts.
job_setup() {
    cd "$PROJECT_ROOT"
    mkdir -p logs

    export PROJECT_ROOT
    export PYTHONPATH="$PROJECT_ROOT"
    export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-${SM_SCRATCH_DIR:-/tmp}/pixi-$USER}"

    # Site setup from .env, e.g. SM_SLURM_PREAMBLE='module load cuda/12.9'
    if [[ -n "${SM_SLURM_PREAMBLE:-}" ]]; then
        echo "Preamble:  ${SM_SLURM_PREAMBLE}"
        eval "${SM_SLURM_PREAMBLE}"
    fi

    echo "Job ID:    ${SLURM_JOB_ID:-none}"
    echo "Node:      $(hostname)"
    echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "W&B mode:  ${WANDB_MODE:-online}"
    echo "Start:     $(date)"
    echo ""
}
