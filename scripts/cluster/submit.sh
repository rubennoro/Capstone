#!/bin/bash
# One entry point for processing and training jobs. Run on the cluster login node from anywhere:
#
#   submit.sh process --task T --dataset D --sessions s1,s2 --raw-dir data/raw/<date> [--sides single]
#   submit.sh train   --task T --dataset D [--smoke] [--note "..."] [--after JOBID] [-- hydra overrides...]
#   submit.sh all     --task T --dataset D --sessions s1,s2 --raw-dir data/raw/<date> [--smoke-first] [--note "..."] [-- hydra overrides...]
#
# Datasets live at ~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip.
# Each training job gets the next results/<task>/run_NNN_<date>[_smoke]/ folder, which the job fills in and pushes.
# `all` chains processing -> (smoke ->) full training; each step only starts if the previous one succeeded.
set -eo pipefail
REPO=~/CapstoneProject/Capstone
SCRIPTS=$REPO/scripts/cluster

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

COMMAND=${1:-}; shift || true
TASK= DATASET= SESSIONS= RAW_DIR= SIDES=single SMOKE=0 SMOKE_FIRST=0 NOTE= AFTER=
OVERRIDES=()
while [ $# -gt 0 ]; do
  case $1 in
    --task) TASK=$2; shift 2 ;;
    --dataset) DATASET=$2; shift 2 ;;
    --sessions) SESSIONS=$2; shift 2 ;;
    --raw-dir) RAW_DIR=${2%/}; shift 2 ;;
    --sides) SIDES=$2; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --smoke-first) SMOKE_FIRST=1; shift ;;
    --note) NOTE=$2; shift 2 ;;
    --after) AFTER=$2; shift 2 ;;
    --) shift; OVERRIDES=("$@"); break ;;
    *) echo "unknown option: $1" >&2; usage ;;
  esac
done
[ -n "$TASK" ] && [ -n "$DATASET" ] || usage
mkdir -p ~/CapstoneProject/logs

# Next free results/<task>/run_NNN_<date>[_smoke]; mkdir is atomic, so simultaneous submissions can't collide
new_run_name() {
  local suffix=$1 task_dir=$REPO/results/$TASK last name
  mkdir -p "$task_dir"
  if [ ! -f "$task_dir/README.md" ]; then
    printf '# %s\n\nDescribe the task here (what the robot should do, objects, success criteria).\n\nEach `run_NNN_<date>/` folder is one training job; its README says which dataset and settings it used.\n' "$TASK" > "$task_dir/README.md"
  fi
  while :; do
    last=$(ls -d "$task_dir"/run_[0-9][0-9][0-9]_* 2>/dev/null | sed 's|.*/run_\([0-9]\{3\}\)_.*|\1|' | sort -n | tail -1)
    name=$(printf 'run_%03d_%s%s' $((10#${last:-0} + 1)) "$(date +%F)" "$suffix")
    mkdir "$task_dir/$name" 2>/dev/null && { echo "$name"; return; }
  done
}

submit_process() {
  [ -n "$SESSIONS" ] && [ -n "$RAW_DIR" ] || { echo "process needs --sessions and --raw-dir" >&2; exit 1; }
  [ -d "$REPO/$RAW_DIR" ] || { echo "raw dir $REPO/$RAW_DIR not found (copy it to the cluster first)" >&2; exit 1; }
  sbatch --parsable "$SCRIPTS/process_dataset.sbatch" "$TASK" "$DATASET" "$SESSIONS" "$RAW_DIR" "$SIDES"
}

submit_train() {  # $1 = full|smoke, $2 = dependency job id (optional)
  local mode=$1 after=$2 suffix= args=()
  [ "$mode" = smoke ] && { suffix=_smoke; args+=(--partition=gpu-short --time=02:00:00); }
  if [ -n "$after" ]; then
    args+=(--dependency=afterok:$after)
  elif [ ! -f ~/CapstoneProject/datasets/$TASK/$DATASET/$DATASET.zarr.zip ]; then
    echo "dataset not found: ~/CapstoneProject/datasets/$TASK/$DATASET/$DATASET.zarr.zip" >&2; exit 1
  fi
  local run_name; run_name=$(new_run_name "$suffix")
  local job; job=$(sbatch --parsable "${args[@]}" "$SCRIPTS/train.sbatch" "$TASK" "$DATASET" "$run_name" "$mode" "$NOTE" "${OVERRIDES[@]}")
  echo "  $mode training: job $job -> results/$TASK/$run_name" >&2
  echo "$job"
}

case $COMMAND in
  process)
    job=$(submit_process); echo "processing: job $job -> ~/CapstoneProject/datasets/$TASK/$DATASET/" ;;
  train)
    if [ "$SMOKE" = 1 ]; then submit_train smoke "$AFTER" >/dev/null; else submit_train full "$AFTER" >/dev/null; fi ;;
  all)
    job=$(submit_process); echo "processing: job $job -> ~/CapstoneProject/datasets/$TASK/$DATASET/"
    if [ "$SMOKE_FIRST" = 1 ]; then job=$(submit_train smoke "$job"); fi
    submit_train full "$job" >/dev/null ;;
  *) usage ;;
esac
echo "Check with: squeue -u $USER   ·   logs in ~/CapstoneProject/logs/"
