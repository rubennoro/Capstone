#!/bin/bash
# Read-only status of a training run on the cluster (Slurm state, log tail, checkpoints so far), or with --eval of the
# newest offline-eval job of one of its checkpoints (train/analysis/eval.sh).
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/status.sh FOLDER [RUN] [--eval topN] [--wait] [--dry-run]

  FOLDER       local task folder data/FOLDER/
  RUN          run name, e.g. run_003_2026-11-13; default: the newest data/FOLDER/policy/*/job.json
  --eval topN  the eval job of that checkpoint (submitted with train/analysis/eval.sh) instead of the training run
  --wait       check every 5 minutes until it is finished (ok, failed or timeout); a lost connection is retried
               24 times in a row (2 h), an error from the cluster ends the wait at once
  --dry-run    print the ssh command, run nothing

The first output line is STATUS=<queued|running|ok|failed|timeout>.
examples: train/status.sh block_raise run_003_2026-11-13 --wait
          train/status.sh block_raise run_003_2026-11-13 --eval top1 --wait
EOF
}

WAIT=0 EVAL="" ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --wait) WAIT=1; shift ;;
    --eval) EVAL=${2:?--eval needs top1, top2, ...}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} -ge 1 ] && [ ${#ARGS[@]} -le 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} RUN=${ARGS[1]:-}
check_name folder "$FOLDER"
[ -z "$EVAL" ] || [[ $EVAL =~ ^top[1-9][0-9]*$ ]] || die "--eval must be top1, top2, ... (got '$EVAL')"

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
# --wait retries only when ssh cannot connect (exit 255: network, laptop asleep), at most MAX_FAILS times in a row
MAX_FAILS=${TRAIN_STATUS_MAX_FAILS:-24}
if [ -n "$EVAL" ]; then
  CMD="bash $(rq "$CLUSTER_TOOLS/submit_job.sh") eval-status $(qargs --label "$LABEL" --run "$RUN" --ckpt "$EVAL")"
  WHAT="$RUN eval $EVAL"
else
  CMD="bash $(rq "$CLUSTER_TOOLS/submit_job.sh") status $(qargs --label "$LABEL" --run "$RUN")"
  WHAT=$RUN
fi

if [ "$DRY_RUN" = 1 ]; then
  login_ssh "$CMD"
  exit 0
fi

fails=0
while :; do
  if out=$(login_ssh "$CMD"); then
    st=$(remote_kv "$out" STATUS) fails=0
  else
    rc=$? st=""
    [ "$WAIT" = 1 ] || { printf '%s\n' "$out"; die "status check failed (see above)"; }
    # anything but a connection failure is an answer from the cluster (run not found, ...): waiting will not fix it
    [ "$rc" = 255 ] || { printf '%s\n' "$out"; die "status check failed on the cluster (exit $rc, see above); not retrying"; }
    fails=$((fails + 1))
    [ "$fails" -lt "$MAX_FAILS" ] ||
      die "status check failed $fails times in a row (ssh exit 255: cannot connect to $CLUSTER_LOGIN); giving up"
    warn "status check failed (ssh exit 255, $fails of $MAX_FAILS); trying again in $((POLL_SEC / 60)) min"
  fi
  case $st in ok|failed|timeout) break ;; esac
  [ "$WAIT" = 1 ] || break
  info "$(date +%H:%M) $WHAT: STATUS=${st:-?}"
  sleep "$POLL_SEC"
done
printf '%s\n' "$out"
case $st in
  ok|failed|timeout)
    if [ -n "$EVAL" ]; then info "next: train/fetch.sh $FOLDER $RUN --no-checkpoints   (results in policy/$RUN/eval/$EVAL/)"
    else info "next: train/fetch.sh $FOLDER $RUN"; fi ;;
esac
