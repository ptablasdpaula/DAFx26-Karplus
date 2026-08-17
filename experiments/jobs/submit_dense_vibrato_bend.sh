#!/bin/bash
# Submit CUDA preflight -> nine arm/lambda tasks -> dependent aggregation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PAYLOAD="${SCRIPT_DIR}/dense_vibrato_bend_payload.sh"
BRANCH="experiment/dense-vibrato-bend-ot"

cd "$PROJECT_ROOT"
if [[ "$(git branch --show-current)" != "$BRANCH" ]]; then
    echo "checkout $BRANCH before submitting" >&2
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "tracked source changes must be committed before submission" >&2
    exit 1
fi
if ! git merge-base --is-ancestor HEAD "origin/${BRANCH}" 2>/dev/null; then
    echo "push HEAD to origin/${BRANCH} before submission" >&2
    exit 1
fi

mkdir -p logs
COMMON=(
    --account=pilot_andrena
    --partition=andrena
    --qos=normal
    --gres=gpu:nvidia_a100-pcie-40gb:1
    --cpus-per-task=12
    --mem=64G
)

PREFLIGHT_JID=$(sbatch --parsable \
    "${COMMON[@]}" \
    --time=01:00:00 \
    --job-name=dense-ks-preflight \
    --output="${PROJECT_ROOT}/logs/dense-ks-preflight-%j.out" \
    --error="${PROJECT_ROOT}/logs/dense-ks-preflight-%j.err" \
    "$PAYLOAD" preflight)

ARRAY_JID=$(sbatch --parsable \
    "${COMMON[@]}" \
    --time=2-00:00:00 \
    --signal=TERM@300 \
    --dependency="afterok:${PREFLIGHT_JID}" \
    --array=0-8 \
    --job-name=dense-ks-sweep \
    --output="${PROJECT_ROOT}/logs/dense-ks-sweep-%A_%a.out" \
    --error="${PROJECT_ROOT}/logs/dense-ks-sweep-%A_%a.err" \
    "$PAYLOAD" sweep)

AGGREGATE_JID=$(sbatch --parsable \
    --account=pilot \
    --partition=compute \
    --qos=normal \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --dependency="afterany:${ARRAY_JID}" \
    --job-name=dense-ks-aggregate \
    --output="${PROJECT_ROOT}/logs/dense-ks-aggregate-%j.out" \
    --error="${PROJECT_ROOT}/logs/dense-ks-aggregate-%j.err" \
    "$PAYLOAD" aggregate)

{
    echo "commit=$(git rev-parse HEAD)"
    echo "preflight=${PREFLIGHT_JID}"
    echo "array=${ARRAY_JID}"
    echo "aggregate=${AGGREGATE_JID}"
} > .dense_vibrato_bend_jids

echo "preflight=${PREFLIGHT_JID}"
echo "array=${ARRAY_JID}"
echo "aggregate=${AGGREGATE_JID}"
echo "monitor: squeue -j ${PREFLIGHT_JID},${ARRAY_JID},${AGGREGATE_JID}"
