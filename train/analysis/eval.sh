#!/bin/bash
# Laptop -> cluster: submit an offline-eval job (franka_deploy offline_eval, optionally attention videos) for one
# checkpoint of a finished run. Results land in <run>/eval/<ckpt>/ on the cluster; get them with train/fetch.sh.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

usage() {
  cat <<'EOF'
usage: train/analysis/eval.sh FOLDER RUN [--ckpt top1|top2|top3] [--per-demo N] [--attention] [--dry-run]

  FOLDER, RUN    a run submitted with train/submit.sh (its data/FOLDER/policy/RUN/job.json names the dataset)
  --ckpt topN    which checkpoint to evaluate (default top1)
  --per-demo N   passed to offline_eval (frames evaluated per demo)
  --attention    also render the attention rollout videos
  --dry-run      print the ssh/rsync commands, run nothing

The cluster clone (CLUSTER_REPO) must be on main >= 4be9982 (it needs controlSW/franka_deploy).
example: train/analysis/eval.sh block_raise run_003_2026-11-13 --ckpt top1 --attention
EOF
}

CKPT=top1 PER_DEMO="" ATTENTION=0 ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --ckpt) CKPT=${2:?--ckpt needs top1, top2, ...}; shift 2 ;;
    --per-demo) PER_DEMO=${2:?--per-demo needs a number}; shift 2 ;;
    --attention) ATTENTION=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} RUN=${ARGS[1]}
check_name folder "$FOLDER"
check_name run "$RUN"
[[ $CKPT =~ ^top[1-9][0-9]*$ ]] || die "--ckpt must be top1, top2, ... (got '$CKPT')"
[ -z "$PER_DEMO" ] || check_posint --per-demo "$PER_DEMO"

load_config
need_var CLUSTER_LOGIN CLUSTER_TOOLS LOCAL_DATA_ROOT
DIR=$LOCAL_DATA_ROOT/$FOLDER/policy/$RUN
META=""
for f in "$DIR/job.json" "$DIR/train_meta.json"; do [ -f "$f" ] && { META=$f; break; }; done
[ -n "$META" ] || die "no $DIR/job.json (or fetched train_meta.json): cannot tell the run's label and dataset"
LABEL=$(json_get "$META" label)
DATASET=$(json_get "$META" dataset)
check_name label "$LABEL"
check_name dataset "$DATASET"

sync_tools
RCMD="bash $(rq "$CLUSTER_TOOLS/submit_job.sh") eval $(qargs --label "$LABEL" --run "$RUN" --dataset "$DATASET" --ckpt "$CKPT")"
[ -z "$PER_DEMO" ] || RCMD+=" $(qargs --per-demo "$PER_DEMO")"
[ "$ATTENTION" = 0 ] || RCMD+=" --attention"
step "Submitting eval of $RUN $CKPT"
if [ "$DRY_RUN" = 1 ]; then
  login_ssh "$RCMD"
  exit 0
fi
out=$(login_ssh "$RCMD") || { printf '%s\n' "$out" >&2; die "eval submission failed (see above)"; }
printf '%s\n' "$out" >&2
JOB=$(remote_kv "$out" JOB)
[ -n "$JOB" ] || die "submit_job.sh printed no JOB= line (see above)"
echo "JOB=$JOB"
info "when it finished: train/fetch.sh $FOLDER $RUN --no-checkpoints   (results in policy/$RUN/eval/$CKPT/)"
