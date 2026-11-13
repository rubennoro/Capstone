#!/bin/bash
# Cluster-side dispatcher for the train/ workflow. Runs on the Explorer login node; the laptop scripts
# (train/submit.sh, train/status.sh, train/analysis/eval.sh) sync this folder + config.env to CLUSTER_TOOLS and call it
# over ssh. Paths in config.env are relative to $HOME (absolute paths are used as they are).
#
#   submit_job.sh train  --folder F --label L --dataset D --mode smoke|full --epochs N --topk K --note TEXT
#                        [--after JOBID] [-- hydra overrides...]
#   submit_job.sh eval   --label L --run R --dataset D --ckpt topN [--per-demo N] [--attention]
#   submit_job.sh status --label L --run R
#
# Machine-readable output lines (everything else is human text):
#   train  -> RUN=<run name>  and  JOB=<slurm job id>
#   eval   -> JOB=<slurm job id>
#   status -> STATUS=<queued|running|ok|failed|timeout>  and  JOB=<slurm job id>
#
# --epochs defaults to DEFAULT_EPOCHS (3 for smoke); --topk to TOPK (1 if unset: one checkpoint per run).
# train: reserves $HOME/CLUSTER_RUNS/L/run_NNN_<date>[_smoke]/ (atomic mkdir, retried on collision), writes
# train_meta.json + STATUS=queued there, and submits train.sbatch with every resource on the sbatch command line
# (command-line options win over the #SBATCH defaults in the script).
#
# Hydra overrides after `--` are appended AFTER the job's fixed overrides. Verified with hydra-core 1.2.0: a key given
# twice as `key=value` is applied in order, so the last one wins and user overrides can replace fixed keys such as
# training.num_epochs or checkpoint.save_train_loss_ckpt_every. The exceptions are refused here: hydra.run.dir,
# logging.dir and task.dataset.dataset_path decide where the job reads and writes (packaging would look in the wrong
# place; use --dataset), and checkpoint.topk.k must equal the number of checkpoints packaged (use --topk). (`+logging.dir=` twice would also be a Hydra error: "Could not append".)
set -euo pipefail

TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
die() { echo "ERROR: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit "${1:-1}"; }
case ${1:-} in -h|--help) usage 0 ;; "") usage 1 ;; esac

[ -f "$TOOLS/config.env" ] || die "$TOOLS/config.env not found (train/submit.sh syncs it here)"
# shellcheck disable=SC2034  # read by config.env (LOCAL_DATA_ROOT default)
TRAIN_DIR=$TOOLS
set +u
# shellcheck source=/dev/null
source "$TOOLS/config.env"
# shellcheck source=/dev/null
[ -f "$TOOLS/config.local.env" ] && source "$TOOLS/config.local.env"
set -u

# Required cluster-side keys: non-empty and not a "<...>" placeholder. SLURM_ACCOUNT may be empty (= no --account).
need() {
  local var
  for var in "$@"; do
    [ -n "${!var:-}" ] || die "config: $var is empty (set it in train/config.env)"
    case ${!var} in *'<'*) die "config: $var is still a placeholder (${!var}); set it in train/config.env" ;; esac
  done
}
need CLUSTER_REPO CLUSTER_DATASETS CLUSTER_RUNS CLUSTER_LOGS CLUSTER_CONDA_MODULE CLUSTER_CONDA_ENV \
  SLURM_PARTITION SLURM_GRES SLURM_TIME SLURM_CPUS SLURM_MEM SMOKE_PARTITION SMOKE_TIME EVAL_PARTITION EVAL_TIME \
  DEFAULT_EPOCHS
case ${SLURM_ACCOUNT:-} in *'<'*) die "config: SLURM_ACCOUNT is still a placeholder; set it or make it \"\"" ;; esac

TOPK=${TOPK:-1}  # one checkpoint per run by default (~2.5 GB slim), like the old pipeline's checkpoint.topk.k=1

# shellcheck disable=SC2088  # a literal '~/' prefix in a config value, not a tilde to expand
abs() { case $1 in /*) echo "$1" ;; '~/'*) echo "$HOME/${1#\~/}" ;; *) echo "$HOME/$1" ;; esac; }
REPO=$(abs "$CLUSTER_REPO")
DATASETS=$(abs "$CLUSTER_DATASETS")
RUNS=$(abs "$CLUSTER_RUNS")
LOGS=$(abs "$CLUSTER_LOGS")

safe_name() {  # used in paths and job names
  [[ $2 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "--$1 must be letters, digits, '.', '_' or '-' (got '$2')"
}

json_str() {  # JSON string literal for arbitrary text
  local s=$1
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\r'/\\r}; s=${s//$'\t'/\\t}
  printf '"%s"' "$(printf '%s' "$s" | tr -d '\000-\010\013\014\016-\037')"
}
json_list() {
  local first=1 x
  printf '['
  for x in "$@"; do [ $first = 1 ] || printf ', '; first=0; json_str "$x"; done
  printf ']'
}

# Resources shared by every job; partition/time differ per job type.
resource_args() {  # $1 partition, $2 time
  SB=(--partition="$1" --time="$2" --gres="$SLURM_GRES" --cpus-per-task="$SLURM_CPUS" --mem="$SLURM_MEM"
      --output="$LOGS/%x_%j.out")
  [ -n "${SLURM_ACCOUNT:-}" ] && SB+=(--account="$SLURM_ACCOUNT")
  return 0
}

# Accepts both "Submitted batch job N" and --parsable output ("N" or "N;cluster").
run_sbatch() {
  local out id
  out=$(sbatch "$@") || die "sbatch failed: $out"
  id=$(printf '%s\n' "$out" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -1)
  [ -n "$id" ] || id=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\{0,1\}$/\1/p' | tail -1)
  [ -n "$id" ] || die "could not read the job id from sbatch output: $out"
  echo "$id"
}

check_clone() {
  [ -f "$REPO/behavior_prompting/behavior_prompting/train_network/train.py" ] && [ -f "$REPO/scripts/package_run.py" ] ||
    die "repo clone not found or incomplete at $REPO (CLUSTER_REPO); clone the repo there first"
}

# Next free run_NNN_<date>[_smoke] under $RUNS/<label>. mkdir is atomic: a concurrent submitter that took the same
# name makes mkdir fail, and the next pass sees its folder and moves on to NNN+1.
# Jobs read this frozen copy, not CLUSTER_TOOLS/config.env: a later submit re-syncs the tools dir, and a queued job
# (e.g. a full run waiting behind its smoke run) must still use the paths/env it was submitted with.
snapshot_config() {  # $1 destination file
  mkdir -p "$(dirname "$1")"
  {
    echo "# config.env (+ config.local.env) as of submission $(date -Iseconds); the job sources this copy"
    cat "$TOOLS/config.env"
    if [ -f "$TOOLS/config.local.env" ]; then echo; echo "# ---- config.local.env"; cat "$TOOLS/config.local.env"; fi
  } > "$1"
}

new_run_name() {  # $1 label dir, $2 suffix
  local dir=$1 suffix=$2 last name attempt
  mkdir -p "$dir"
  for attempt in $(seq 1 50); do
    last=$(ls -d "$dir"/run_[0-9][0-9][0-9]_* 2>/dev/null | sed 's|.*/run_\([0-9]\{3\}\)_.*|\1|' | sort -n | tail -1 || true)
    name=$(printf 'run_%03d_%s%s' $((10#${last:-0} + 1)) "$(date +%F)" "$suffix")
    if mkdir "$dir/$name" 2>/dev/null; then echo "$name"; return 0; fi
    [ -e "$dir/$name" ] || die "cannot create $dir/$name (permissions? quota?)"
    echo "run name $name was just taken; retrying (attempt $attempt)" >&2
  done
  die "could not reserve a run folder in $dir after 50 attempts"
}

# ------------------------------------------------------------------ train
cmd_train() {
  local folder='' label='' dataset='' mode='' epochs='' topk='' note='' after='' overrides=()
  while [ $# -gt 0 ]; do
    case $1 in
      --folder) folder=${2?}; shift 2 ;;
      --label) label=${2?}; shift 2 ;;
      --dataset) dataset=${2?}; shift 2 ;;
      --mode) mode=${2?}; shift 2 ;;
      --epochs) epochs=${2?}; shift 2 ;;
      --topk) topk=${2?}; shift 2 ;;
      --note) note=${2?}; shift 2 ;;
      --after) after=${2?}; shift 2 ;;
      --) shift; overrides=("$@"); break ;;
      -h|--help) usage 0 ;;
      *) die "train: unknown option $1" ;;
    esac
  done
  [ -n "$folder" ] && [ -n "$label" ] && [ -n "$dataset" ] && [ -n "$mode" ] || die "train needs --folder --label --dataset --mode"
  safe_name folder "$folder"; safe_name label "$label"; safe_name dataset "$dataset"
  case $mode in smoke|full) ;; *) die "--mode must be smoke or full" ;; esac
  [ -n "$epochs" ] || { [ "$mode" = smoke ] && epochs=3 || epochs=$DEFAULT_EPOCHS; }
  [ -n "$topk" ] || topk=$TOPK
  [[ $epochs =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
  [[ $topk =~ ^[1-9][0-9]*$ ]] || die "--topk must be a positive integer"
  [ -z "$after" ] || [[ $after =~ ^[0-9]+$ ]] || die "--after must be a job id"
  local o key
  for o in ${overrides[@]+"${overrides[@]}"}; do
    key=${o%%=*}; key=${key#\~}; key=${key#+}; key=${key#+}
    case $key in
      hydra.run.dir|logging.dir|task.dataset.dataset_path)
        die "override '$o' is not allowed: the job sets $key itself (use --dataset to pick the dataset)" ;;
      checkpoint.topk.k)
        die "override '$o' is not allowed: use --topk K (the job saves and packages exactly K checkpoints)" ;;
    esac
  done

  check_clone
  local zip="$DATASETS/$label/$dataset/$dataset.zarr.zip"
  [ -f "$zip" ] || die "dataset not found: $zip (upload it with train/upload.sh)"
  mkdir -p "$LOGS"

  local suffix=; [ "$mode" = smoke ] && suffix=_smoke
  local run run_dir commit submitted
  run=$(new_run_name "$RUNS/$label" "$suffix")
  run_dir="$RUNS/$label/$run"
  commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
  submitted=$(date -Iseconds)
  write_meta() {  # $1 = job id ("" before submission)
    {
      printf '{\n'
      printf '  "folder": %s,\n  "label": %s,\n  "dataset": %s,\n  "run": %s,\n  "mode": %s,\n' \
        "$(json_str "$folder")" "$(json_str "$label")" "$(json_str "$dataset")" "$(json_str "$run")" "$(json_str "$mode")"
      printf '  "epochs": %s,\n  "topk": %s,\n  "note": %s,\n  "overrides": %s,\n' \
        "$epochs" "$topk" "$(json_str "$note")" "$(json_list ${overrides[@]+"${overrides[@]}"})"
      printf '  "submitted": %s,\n  "clone_git_commit": %s,\n  "after": %s,\n  "job_id": %s,\n  "dataset_zip": %s\n}\n' \
        "$(json_str "$submitted")" "$(json_str "$commit")" "$(json_str "$after")" "$(json_str "$1")" "$(json_str "$zip")"
    } > "$run_dir/train_meta.json"
  }
  write_meta ""
  echo queued > "$run_dir/STATUS"
  snapshot_config "$run_dir/run_config/submit_config.env"

  local partition=$SLURM_PARTITION time=$SLURM_TIME
  [ "$mode" = smoke ] && { partition=$SMOKE_PARTITION; time=$SMOKE_TIME; }
  resource_args "$partition" "$time"
  SB+=(--job-name="bp_train_$label" --signal=B:USR1@600)
  [ -n "$after" ] && SB+=(--dependency="afterok:$after")
  local job
  # train.sbatch args: <run dir> <label> <dataset> <run> <mode> <epochs> <topk> <note> [overrides...]
  # (paths are passed because Slurm runs a spooled copy of the script; the job sources <run dir>/run_config/submit_config.env)
  if ! job=$(run_sbatch "${SB[@]}" "$TOOLS/train.sbatch" "$run_dir" "$label" "$dataset" "$run" "$mode" "$epochs" "$topk" \
      "$note" ${overrides[@]+"${overrides[@]}"}); then
    rm -rf "$run_dir/run_config"; rm -f "$run_dir/train_meta.json" "$run_dir/STATUS"; rmdir "$run_dir" 2>/dev/null || true
    die "submission failed; released $run_dir"
  fi
  write_meta "$job"
  echo "$mode training: job $job -> $run_dir (${epochs} epochs, top-$topk, $partition, $time${after:+, after job $after})"
  echo "RUN=$run"
  echo "JOB=$job"
}

# ------------------------------------------------------------------ eval
cmd_eval() {
  local label='' run='' dataset='' ckpt=top1 per_demo=0 attention=0
  while [ $# -gt 0 ]; do
    case $1 in
      --label) label=${2?}; shift 2 ;;
      --run) run=${2?}; shift 2 ;;
      --dataset) dataset=${2?}; shift 2 ;;
      --ckpt) ckpt=${2?}; shift 2 ;;
      --per-demo) per_demo=${2?}; shift 2 ;;
      --attention) attention=1; shift ;;
      -h|--help) usage 0 ;;
      *) die "eval: unknown option $1" ;;
    esac
  done
  [ -n "$label" ] && [ -n "$run" ] && [ -n "$dataset" ] || die "eval needs --label --run --dataset"
  safe_name label "$label"; safe_name run "$run"; safe_name dataset "$dataset"
  [[ $ckpt =~ ^top[1-9][0-9]*$ ]] || die "--ckpt must be top1, top2, ..."
  [[ $per_demo =~ ^[0-9]+$ ]] || die "--per-demo must be a non-negative integer"
  check_clone
  [ -d "$REPO/controlSW/franka_deploy/franka_deploy" ] ||
    die "controlSW/franka_deploy missing in the clone: git -C ~/CapstoneProject/Capstone pull (clone is older than 4be9982)"
  local final="$RUNS/$label/$run" zip="$DATASETS/$label/$dataset/$dataset.zarr.zip"
  [ -d "$final" ] || die "run not found: $final"
  [ -f "$zip" ] || die "dataset not found: $zip"
  local files=("$final"/checkpoints/"${ckpt}"_*.ckpt)
  [ -f "${files[0]}" ] || die "no checkpoints/${ckpt}_*.ckpt in $final (STATUS=$(cat "$final/STATUS" 2>/dev/null || echo ?); training not packaged yet?)"
  [ ${#files[@]} -eq 1 ] || die "more than one checkpoints/${ckpt}_*.ckpt in $final"
  mkdir -p "$LOGS"
  resource_args "$EVAL_PARTITION" "$EVAL_TIME"
  SB+=(--job-name="bp_eval_$label" --signal=B:USR1@300)
  local job
  # eval.sbatch args: <eval dir> <label> <run> <dataset> <ckpt tag> <per-demo N> <attention 0|1>
  # (the job sources <eval dir>/submit_config.env, frozen here)
  snapshot_config "$final/eval/$ckpt/submit_config.env"
  job=$(run_sbatch "${SB[@]}" "$TOOLS/eval.sbatch" "$final/eval/$ckpt" "$label" "$run" "$dataset" "$ckpt" "$per_demo" "$attention")
  echo "eval of $label/$run/$ckpt: job $job -> $final/eval/$ckpt/ (per-demo $per_demo, attention $attention)"
  echo "JOB=$job"
}

# ------------------------------------------------------------------ status
cmd_status() {
  local label='' run=''
  while [ $# -gt 0 ]; do
    case $1 in
      --label) label=${2?}; shift 2 ;;
      --run) run=${2?}; shift 2 ;;
      -h|--help) usage 0 ;;
      *) die "status: unknown option $1" ;;
    esac
  done
  [ -n "$label" ] && [ -n "$run" ] || die "status needs --label --run"
  safe_name label "$label"; safe_name run "$run"
  local final="$RUNS/$label/$run"
  [ -d "$final" ] || die "run not found: $final"
  local job
  job=$(sed -n 's/^ *"job_id": *"\([0-9]*\)".*/\1/p' "$final/train_meta.json" 2>/dev/null | head -1 || true)

  # Ask Slurm first, then read STATUS, so a job that finishes in between is not reported as dead.
  local queue='' acct='' reason=''
  if [ -n "$job" ]; then
    queue=$(squeue -h -j "$job" -o '%T %M %R' 2>/dev/null || true)
    acct=$(sacct -j "$job" -X -n -P -o State,ExitCode,Elapsed,Start,End,Partition,NodeList 2>/dev/null | head -1 || true)
  fi
  local status
  status=$(tr -d '[:space:]' < "$final/STATUS" 2>/dev/null || true)
  [ -n "$status" ] || status=queued
  # The job writes the final STATUS itself. If it died without doing so (cancelled, node failure, OOM kill, hard time
  # limit) or can never start (its --after job failed), report it from Slurm so status.sh --wait terminates.
  case $status in
    queued|running)
      local state=${acct%%|*}; state=${state%% *}
      if [ -n "$queue" ]; then
        case $queue in
          *DependencyNeverSatisfied*)
            status=failed; reason="the job it waits for (--after) did not succeed, so it will never start; cancel it: scancel $job" ;;
          RUNNING*) status=running ;;
        esac
      elif [ -n "$state" ]; then
        case $state in
          PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING) ;;
          TIMEOUT) status=timeout; reason="Slurm says TIMEOUT but the job did not package (STATUS file: $status)" ;;
          *) status=failed; reason="Slurm says $state but the job did not write a final STATUS (STATUS file: $status)" ;;
        esac
      fi ;;
  esac

  echo "STATUS=$status"
  echo "JOB=$job"
  [ -n "$reason" ] && echo "note: $reason"
  echo "run: $final"
  echo "slurm: ${queue:+queue: $queue  }${acct:+sacct: $(echo "$acct" | tr '|' ' ')}"
  [ -n "$queue$acct" ] || echo "slurm: (no information${job:+ for job $job})"

  echo "== Log tail (progress bars stripped)"
  local log=''
  [ -n "$job" ] && log=$(ls "$LOGS"/*_"$job".out 2>/dev/null | head -1 || true)
  if [ -n "$log" ]; then
    echo "   $log"
    tr '\r' '\n' < "$log" | grep -v -E '[0-9]+%\|.*\||it/s\]|s/it\]' | grep -v '^\s*$' | tail -n 15 || true
  else
    echo "(no log yet)"
  fi

  echo "== Checkpoints"
  local f any=0
  for f in "$final"/checkpoints/*.ckpt; do
    [ -f "$f" ] || continue
    any=1; printf '%s  %s bytes (%s)\n' "${f##*/}" "$(stat -c %s "$f")" "$(du -h "$f" | cut -f1)"
  done
  [ $any = 1 ] || echo "(none packaged yet)"
  local scratch="/scratch/$USER/bp_runs/$label/$run/checkpoints"
  if [ -d "$scratch" ]; then
    echo "on scratch ($scratch):"
    for f in "$scratch"/*.ckpt; do [ -f "$f" ] && printf '  %s  %s\n' "${f##*/}" "$(du -h "$f" | cut -f1)"; done
  fi
  if [ -d "$final/eval" ]; then
    echo "== Evals"
    for f in "$final"/eval/*/eval_job.json; do
      [ -f "$f" ] && printf '%s: %s\n' "$(basename "$(dirname "$f")")" \
        "$(sed -n 's/^ *"status": *"\([a-z]*\)".*/\1/p' "$f" | head -1)"
    done
  fi
  return 0
}

COMMAND=${1:-}; shift || true
case $COMMAND in
  train) cmd_train "$@" ;;
  eval) cmd_eval "$@" ;;
  status) cmd_status "$@" ;;
  -h|--help) usage 0 ;;
  *) usage ;;
esac
