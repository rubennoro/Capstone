#!/bin/bash
# Laptop-side entry point: copy new raw data to the cluster (if --raw-dir is given), update the cluster's repo
# clone, and submit jobs with scripts/cluster/submit.sh. Takes the same arguments as submit.sh, e.g.:
#
#   scripts/run_pipeline.sh all --task block_raise --dataset oct12-single --sessions oct12 \
#       --raw-dir data/raw/2026-10-12 --smoke-first --note "second recording session"
#   scripts/run_pipeline.sh train --task block_raise --dataset firstreal-single --note "longer run" -- training.num_epochs=200
#
# Finished runs are pushed to GitHub by the training job itself: `git pull` to get them.
set -eo pipefail
LOGIN=noroian.r@login.explorer.northeastern.edu
XFER=noroian.r@xfer.discovery.neu.edu
cd "$(dirname "$0")/.."

[ $# -gt 0 ] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

# Find --raw-dir among the arguments (before any `--` hydra overrides)
RAW_DIR=
args=("$@")
for i in "${!args[@]}"; do
  [ "${args[$i]}" = "--" ] && break
  [ "${args[$i]}" = "--raw-dir" ] && RAW_DIR=${args[$((i + 1))]%/}
done

if [ -n "$RAW_DIR" ]; then
  [ -d "$RAW_DIR" ] || { echo "raw dir not found locally: $RAW_DIR" >&2; exit 1; }
  echo "== Copying $RAW_DIR to the cluster (through the transfer node)"
  rsync -a --partial --info=progress2 -e "ssh -o ServerAliveInterval=30" \
    "$RAW_DIR/" "$XFER:CapstoneProject/Capstone/$RAW_DIR/"
fi

# The cluster pulls from GitHub, so local commits it should run with must be pushed first
git fetch -q origin
if [ -n "$(git log origin/main..HEAD --oneline 2>/dev/null)" ]; then
  echo "WARNING: you have local commits not pushed to GitHub; the cluster will run without them." >&2
fi

echo "== Updating the cluster clone and submitting"
ssh "$LOGIN" "cd ~/CapstoneProject/Capstone && git pull -q --rebase --autostash && bash scripts/cluster/submit.sh $(printf '%q ' "$@")"
