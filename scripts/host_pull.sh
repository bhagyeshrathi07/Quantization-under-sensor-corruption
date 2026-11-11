#!/usr/bin/env bash
# Bring the GPU host's checkout up to date without losing anything it produced.
#
#   bash scripts/host_pull.sh              # in /workspace/qflip
#   HOST_BACKUP=/workspace/host_backup_1 bash scripts/host_pull.sh   # must not exist yet
#
# A checkout that predates this script (the host sits at 7dfd138) takes it from
# the fetched commit before running it:
#   git fetch origin main && git show origin/main:scripts/host_pull.sh > /workspace/host_pull.sh
#   bash /workspace/host_pull.sh
#
# Why not `git checkout -- . && git pull`: the host holds untracked copies of files
# the incoming commits add (the frozen files written by freeze_config.py, the
# profile and stability results, scp'd scripts and manifests). Git refuses to
# overwrite untracked files even when they are byte-identical, so the pull fails.
# This script moves every colliding untracked path and copies every modified
# tracked path into a backup directory, restores the tracked tree, fast-forwards,
# and then compares each backed-up file with what was pulled: a difference is
# reported and exits non-zero (nothing is deleted; the backup stays). Untracked
# paths the incoming commits do not touch (logs/, results/scratch/) are left alone.
# The backup directory is new for every invocation: an existing one is refused
# before anything moves, so an earlier run's host copies can never be overwritten.
set -euo pipefail

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"
BACKUP="${HOST_BACKUP:-$(pwd)/../host_backup_$(date +%Y%m%d_%H%M%S)}"

git rev-parse --is-inside-work-tree >/dev/null
if [ -e "$BACKUP" ]; then
  echo "host_pull.sh: backup directory $BACKUP already exists; refusing to reuse it (an earlier run's host copies would be overwritten). Give a new HOST_BACKUP or omit it." >&2
  exit 2
fi
git fetch "$REMOTE" "$BRANCH"
mkdir -p "$BACKUP"
BACKUP="$(cd "$BACKUP" && pwd)"
moved=0

# untracked paths that the incoming commits add or change: move aside
while IFS= read -r f; do
  [ -n "$f" ] || continue
  if [ -e "$f" ] && ! git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
    mkdir -p "$BACKUP/$(dirname "$f")"
    mv "$f" "$BACKUP/$f"
    moved=$((moved + 1))
  fi
done < <(git diff --name-only HEAD "$REMOTE/$BRANCH")

# tracked paths the host modified (serve manifests, scp'd files): copy, then restore
while IFS= read -r f; do
  [ -n "$f" ] || continue
  mkdir -p "$BACKUP/$(dirname "$f")"
  cp -p "$f" "$BACKUP/$f"
  moved=$((moved + 1))
done < <(git diff --name-only)
git checkout -- .

git merge --ff-only "$REMOTE/$BRANCH"

# every backed-up file must equal what was pulled; host-only files stay in the backup
status=0
while IFS= read -r -d '' b; do
  rel="${b#"$BACKUP"/}"
  if [ -e "$rel" ]; then
    cmp -s "$b" "$rel" || { echo "DIFFERS: $rel (host copy kept at $b)"; status=1; }
  else
    echo "host-only, kept in backup: $rel"
  fi
done < <(find "$BACKUP" -type f -print0)

echo "backed up $moved path(s) to $BACKUP; now at $(git rev-parse --short HEAD)"
if [ "$status" -ne 0 ]; then
  echo "pull done, but the host copies above differ from the pulled files: look before deleting the backup" >&2
fi
exit "$status"
