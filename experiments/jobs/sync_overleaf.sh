#!/bin/bash
# Push the repo's Latex/ folder to the GitHub repo that Overleaf is linked to
# (ptablasdpaula/DAFx26-Karplus-draft). Overleaf then syncs that repo -> PDF.
#
# One-time GitHub auth on the cluster (you do this; needs a Personal Access Token
# with `repo` scope from https://github.com/settings/tokens):
#   ! git config --global credential.helper store
#   ! git clone https://github.com/ptablasdpaula/DAFx26-Karplus-draft.git /data/home/acw794/overleaf-DAFx26-draft
#       Username: <your github user>   Password: <the PAT>
#   (credential.helper store saves it so future pushes are non-interactive)
#
# Then sync anytime:
#   bash experiments/jobs/sync_overleaf.sh "edited results section"
#
# Mirrors Latex/ INTO the draft repo root. Non-destructive unless DELETE=1.
set -euo pipefail

REPO=/data/home/acw794/DAFx26-Karplus
DRAFT_URL="${DRAFT_URL:-https://github.com/ptablasdpaula/DAFx26-Karplus-draft.git}"
WORK="${DRAFT_DIR:-/data/home/acw794/overleaf-DAFx26-draft}"
MSG="${1:-sync Latex/ from cluster $(date -u +%F\ %T)}"
RSYNC_DELETE=""; [ "${DELETE:-0}" = "1" ] && RSYNC_DELETE="--delete --exclude=README.md"

if [ ! -d "$WORK/.git" ]; then
  echo "ERROR: $WORK not cloned yet. Do the one-time clone first (see header)." >&2
  exit 1
fi

cd "$WORK"
echo "Pulling latest from the draft repo (in case Overleaf pushed edits)..."
git pull --no-rebase || true

echo "Mirroring $REPO/Latex/ -> $WORK $RSYNC_DELETE"
# Never push working material to Overleaf: the attached reference PDFs live at the
# Latex/ root (figures/*.pdf still sync), the reviewer feedback, and the README.
rsync -a $RSYNC_DELETE --exclude='.git/' --exclude='README.md' \
      --exclude='/*.pdf' --exclude='feedbacK' \
      "$REPO/Latex/" "$WORK/"

git add -A
if git diff --cached --quiet; then echo "Nothing changed; nothing to push."; exit 0; fi
git commit -m "$MSG"
git push
echo "Pushed to draft repo. In Overleaf: Menu -> GitHub -> pull (if not automatic) to update the PDF."
