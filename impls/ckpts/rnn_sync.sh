#!/bin/bash
# One-way, additive-only sync of this ckpts/ dir to rnn.
#
# Safety guarantees:
#   - Never deletes anything on rnn (no --delete, ever).
#   - Never overwrites an existing remote file (--ignore-existing): if rnn
#     already has a file at a given path, it is left untouched, even if the
#     local copy differs. This protects extra data that may already live in
#     remote run folders.
#   - Only files that don't yet exist on rnn get copied over.
#
# Usage:
#   ./rnn_sync.sh              # perform the sync
#   ./rnn_sync.sh --dry-run    # show what would be copied, without copying

set -euo pipefail

REMOTE_HOST="rnn"
REMOTE_DIR="/nas/ucb/ishirgarg/ogbench/impls/ckpts"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RSYNC_FLAGS=(
  -avz
  --progress
  --ignore-existing
  --exclude='.DS_Store'
  --exclude='_staging_noisy'
  --exclude='__pycache__'
)

if [[ "${1:-}" == "--dry-run" ]]; then
  RSYNC_FLAGS+=(--dry-run)
  echo "[dry run] previewing sync from $LOCAL_DIR/ -> $REMOTE_HOST:$REMOTE_DIR/"
elif [[ -n "${1:-}" ]]; then
  echo "Unknown argument: $1" >&2
  echo "Usage: $0 [--dry-run]" >&2
  exit 1
else
  echo "Syncing $LOCAL_DIR/ -> $REMOTE_HOST:$REMOTE_DIR/ (additive only, nothing on rnn will be deleted or overwritten)"
fi

rsync "${RSYNC_FLAGS[@]}" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
