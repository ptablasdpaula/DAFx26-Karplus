#!/bin/bash
# Fetch the paper's trained checkpoints into experiments/checkpoints/.
#
#   scripts/fetch_checkpoints.sh
#
# The weights live on the `checkpoints` branch rather than on main, so a clone
# of the code stays small. They are Git LFS objects, downloaded on demand here.

set -euo pipefail

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs is required to fetch the checkpoints." >&2
    echo "  macOS:  brew install git-lfs && git lfs install" >&2
    echo "  Debian: sudo apt install git-lfs && git lfs install" >&2
    exit 1
fi

cd "$(git rev-parse --show-toplevel)"

REMOTE="${CHECKPOINTS_REMOTE:-origin}"
BRANCH="${CHECKPOINTS_BRANCH:-checkpoints}"

echo "Fetching ${REMOTE}/${BRANCH}..."
# Explicit refspec: a --single-branch clone only tracks its own branch, so a
# plain `git fetch origin checkpoints` leaves no ref to check out from.
git fetch --no-tags "$REMOTE" \
    "refs/heads/${BRANCH}:refs/remotes/${REMOTE}/${BRANCH}"

echo "Checking out experiments/checkpoints/..."
git checkout "refs/remotes/${REMOTE}/${BRANCH}" -- experiments/checkpoints
git restore --staged experiments/checkpoints

count=$(find experiments/checkpoints -name '*.ckpt' | wc -l | tr -d ' ')
smallest=$(find experiments/checkpoints -name '*.ckpt' -exec stat -f%z {} \; 2>/dev/null \
    || find experiments/checkpoints -name '*.ckpt' -exec stat -c%s {} \;)
if [[ -n "$smallest" ]] && [[ $(echo "$smallest" | sort -n | head -1) -lt 100000 ]]; then
    echo "" >&2
    echo "Warning: some .ckpt files are only a few hundred bytes — those are LFS" >&2
    echo "pointers, not weights. Run 'git lfs install' and try again." >&2
    exit 1
fi

echo "Fetched ${count} checkpoints into experiments/checkpoints/."
