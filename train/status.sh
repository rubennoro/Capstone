#!/bin/bash
# Read-only status of a training run on the cluster (Slurm state, log tail, checkpoints so far).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/status.sh FOLDER [RUN] [--wait] [--dry-run]

  FOLDER     local task folder data/FOLDER/
  RUN        run name, e.g. run_003_2026-11-13; default: the newest data/FOLDER/policy/*/job.json
  --wait     check every 5 minutes until the run is finished (ok, failed or timeout)
  --dry-run  print the ssh command, run nothing

example: train/status.sh block_raise run_003_2026-11-13 --wait
EOF
}

WAIT=0 ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --wait) WAIT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} -ge 1 ] && [ ${#ARGS[@]} -le 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} RUN=${ARGS[1]:-}
check_name folder "$FOLDER"

load_config
need_var CLUSTER_LOGIN CLUSTER_TOOLS LOCAL_DATA_ROOT
if [ -z "$RUN" ]; then
  newest=$(ls -1t "$LOCAL_DATA_ROOT/$FOLDER"/policy/*/job.json 2>/dev/null | head -1 || true)
  [ -n "$newest" ] || die "no data/$FOLDER/policy/*/job.json; give the run name"
  RUN=$(basename "$(dirname "$newest")")
  info "run: $RUN (newest job.json)"
fi
check_name run "$RUN"
LABEL=$(run_label "$FOLDER" "$RUN")
check_label "$LABEL"

POLL_SEC=${TRAIN_STATUS_POLL_SEC:-300}   # test hook
CMD="bash $(rq "$CLUSTER_TOOLS/submit_job.sh") status $(qargs --label "$LABEL" --run "$RUN")"

if [ "$DRY_RUN" = 1 ]; then
  login_ssh "$CMD"
  exit 0
fi

while :; do
  if out=$(login_ssh "$CMD"); then
    st=$(remote_kv "$out" STATUS)
  else
    st="" ; [ "$WAIT" = 1 ] || { printf '%s\n' "$out"; die "status check failed (see above)"; }
    warn "status check failed; trying again in $((POLL_SEC / 60)) min"
  fi
  case $st in ok|failed|timeout) break ;; esac
  [ "$WAIT" = 1 ] || break
  info "$(date +%H:%M) $RUN: STATUS=${st:-?}"
  sleep "$POLL_SEC"
done
printf '%s\n' "$out"
case $st in
  ok|failed|timeout) info "next: train/fetch.sh $FOLDER $RUN" ;;
esac
