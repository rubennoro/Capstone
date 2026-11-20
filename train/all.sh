#!/bin/bash
# The whole train/ workflow in one command, for good data and little time. Runs on the laptop for hours (the full run
# alone is ~2 h): start it in tmux, or with nohup, so closing the terminal does not stop it.
#
#   pull (+ dataproc package) -> dataset report -> upload -> submit (smoke run, then the full run) -> wait ->
#   fetch the policy -> run report -> compare runs -> offline eval (submit, wait, fetch, report)
#
# Every step calls the same script as the manual run order in train/DIRECTIONS.md, so a step that fails can be rerun
# by hand. Stop it with Ctrl-C (or by closing its tmux window): the step that is running stops and the log says how to
# continue. A plain `kill` of one all.sh process takes effect only when the current step ends (the steps stay in the
# foreground so ssh can still ask for a password or Duo). Traceability: everything printed also goes to data/FOLDER/logs/all_<time>_<name>.log, and each step's
# start and result (ok / failed / skipped / interrupted, with the reason) to the .steps file next to it.
#
# Test hook (not for normal use): TRAIN_ANALYSIS_SRC = folder with the analysis *.py reports instead of train/analysis/.
set -uo pipefail
ORIG_ARGS=("$@")
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/all.sh FOLDER DATASET [--package | --pull] [--no-raw] [--policy plain|prompting] [--no-smoke] [--epochs N]
                    [--note TEXT] [--no-eval] [--per-demo N] [--attention] [--results-dir DIR]... [--dry-run]
                    [-- hydra overrides...]
       train/all.sh FOLDER --run RUN [--no-eval] [--per-demo N] [--attention] [--results-dir DIR]... [--dry-run]

  --package        first run `dataproc package DATASET FOLDER` on the Jetson, then pull (DATASET must be a new name)
  --pull           first pull the task folder and the already packaged DATASET from the Jetson
                   (neither: use data/FOLDER/package/DATASET/ as it already is on the laptop)
  --no-raw         with --pull/--package: skip the raw recordings
  --policy P       plain = plain diffusion, prompting = behavior prompting (default: TRAIN_CONFIG in config.env)
  --no-smoke       submit only the full run (default: a 3-epoch smoke run first; the full run starts if it succeeds)
  --epochs N       epochs of the full run (default DEFAULT_EPOCHS, lowered to fit SLURM_TIME)
  --note TEXT      free text stored with the run(s)
  --no-eval        stop after the run reports (no offline eval job)
  --per-demo N     offline eval: frames evaluated per demo (default 8)
  --attention      offline eval: also render the attention rollout videos (behavior prompting only)
  --results-dir D  compare_runs.py also compares with the runs in D (e.g. results/block_raise); repeatable
  --run RUN        continue a run that is already submitted: wait for it, then fetch, reports and eval. Use it when
                   all.sh stopped after submitting (Ctrl-C, laptop asleep, fetch failed); the log names the command.
  --dry-run        print the commands up to the submission, list the steps after it; run and write nothing
  -- ...           extra Hydra overrides passed to submit.sh

Exit code: 0  every step ok
           1  stopped early: no trained policy on the laptop (the end of the output says which step and why)
           2  the policy is on the laptop, but a report or the offline eval failed (see the .steps file)
           130 interrupted by Ctrl-C (129: terminal/tmux window closed); submitted jobs keep running, continue with --run
example (in tmux): train/all.sh eraser_pick_2 ep2-v1 --package --no-raw --note "first rig data"
EOF
}

PULL=0 PACKAGE=0 NO_RAW=0 POLICY="" SMOKE=1 EPOCHS="" NOTE="" HAVE_NOTE=0 EVAL=1 PER_DEMO=8 ATTENTION=0 RESUME=""
RESULTS_DIRS=() ARGS=() OVERRIDES=() PRE_SUBMIT=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --pull) PULL=1; PRE_SUBMIT+=("$1"); shift ;;
    --package) PACKAGE=1; PULL=1; PRE_SUBMIT+=("$1"); shift ;;
    --no-raw) NO_RAW=1; PRE_SUBMIT+=("$1"); shift ;;
    --policy) POLICY=${2:?--policy needs plain or prompting}; PRE_SUBMIT+=("$1"); shift 2 ;;
    --no-smoke) SMOKE=0; PRE_SUBMIT+=("$1"); shift ;;
    --epochs) EPOCHS=${2:?--epochs needs a number}; PRE_SUBMIT+=("$1"); shift 2 ;;
    --note) NOTE=${2?--note needs a text}; HAVE_NOTE=1; PRE_SUBMIT+=("$1"); shift 2 ;;
    --no-eval) EVAL=0; shift ;;
    --per-demo) PER_DEMO=${2:?--per-demo needs a number}; shift 2 ;;
    --attention) ATTENTION=1; shift ;;
    --results-dir) RESULTS_DIRS+=("${2:?--results-dir needs a folder}"); shift 2 ;;
    --run) RESUME=${2:?--run needs a run name}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; OVERRIDES=("$@"); PRE_SUBMIT+=(--); break ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -n "$RESUME" ]; then
  [ ${#ARGS[@]} = 1 ] || { usage >&2; die "with --run, give only FOLDER (the dataset is in the run's job.json)"; }
  [ ${#PRE_SUBMIT[@]} = 0 ] || die "${PRE_SUBMIT[0]} only applies before submitting; --run continues a submitted run"
  check_name run "$RESUME"
else
  [ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }
fi
FOLDER=${ARGS[0]} DATASET=${ARGS[1]:-}
check_name folder "$FOLDER"
[ -z "$DATASET" ] || check_name dataset "$DATASET"
[ -z "$EPOCHS" ] || check_posint --epochs "$EPOCHS"
check_posint --per-demo "$PER_DEMO"
[ "$NO_RAW" = 0 ] || [ "$PULL" = 1 ] || die "--no-raw only applies to pulling: add --pull or --package"
[ "$ATTENTION" = 0 ] || [ "$EVAL" = 1 ] || die "--attention only applies to the eval: drop --no-eval"
case $POLICY in ""|plain|prompting) ;; *) die "--policy must be plain or prompting (got '$POLICY')" ;; esac
for d in ${RESULTS_DIRS[@]+"${RESULTS_DIRS[@]}"}; do [ -d "$d" ] || die "--results-dir $d: no such folder"; done

load_config
need_var LOCAL_DATA_ROOT CLUSTER_LOGIN CLUSTER_LOGS
# attention videos exist only for behavior prompting: refuse now, not hours later at the eval step
if [ -z "$RESUME" ] && [ "$ATTENTION" = 1 ]; then
  if [ "$POLICY" = plain ] || { [ -z "$POLICY" ] && [ "${TRAIN_CONFIG:-umi_policy_dunetp}" != umi_policy_dunetp ]; }; then
    die "--attention needs behavior prompting; this would train plain diffusion (drop --attention or use --policy prompting)"
  fi
fi
POLICY=$LOCAL_DATA_ROOT/$FOLDER/policy
ANALYSIS=${TRAIN_ANALYSIS_SRC:-$TRAIN_DIR/analysis}
if [ -n "$RESUME" ]; then
  [ -f "$POLICY/$RESUME/job.json" ] ||
    die "no data/$FOLDER/policy/$RESUME/job.json: --run needs a run submitted from this laptop (train/submit.sh or all.sh)"
  DATASET=$(json_get "$POLICY/$RESUME/job.json" dataset)
fi

# ---- logging: run this script again with its output going through tee into the log file (also stderr, in order)
if [ "$DRY_RUN" = 0 ] && [ -z "${TRAIN_ALL_LOG:-}" ]; then
  logdir=$LOCAL_DATA_ROOT/$FOLDER/logs
  mkdir -p "$logdir" || die "cannot create $logdir"
  TRAIN_ALL_LOG=$logdir/all_$(date +%Y-%m-%d_%H%M%S)_${RESUME:-$DATASET}.log
  export TRAIN_ALL_LOG
  # tee ignores Ctrl-C / kill / hangup and ends when the script closes its output, so the explanation the script prints
  # when it is stopped still reaches the log
  bash "${BASH_SOURCE[0]}" ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"} 2>&1 | (trap '' INT TERM HUP; exec tee -a "$TRAIN_ALL_LOG")
  exit "${PIPESTATUS[0]}"
fi
STEPS=""
[ "$DRY_RUN" = 1 ] || STEPS=${TRAIN_ALL_LOG%.log}.steps
TMPD=$(mktemp -d) || die "mktemp -d failed: cannot keep the output of the steps (is /tmp full?)"

# ---- step bookkeeping: every step is "begin NAME", then exactly one "finish RESULT [WHY]" or "stop WHY [NEXT]"
CUR="" DONE=0 PROBLEMS=() FULL_RUN="" SMOKE_RUN="" SUBMITTED=0
_rec() { [ -z "$STEPS" ] || printf '%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$1" "$2" "$3" >>"$STEPS"; }
begin() { CUR=$1; step "[$1] $2"; _rec "$1" started "$2"; }
finish() {  # RESULT [WHY]: ok | failed (the workflow goes on) | interrupted
  _rec "$CUR" "$1" "${2:-}"
  info "-- [$CUR] $1$([ "$DRY_RUN" = 0 ] || echo ' (dry-run)')${2:+: $2}"
  case $1 in failed|interrupted) PROBLEMS+=("$CUR: ${2:-}") ;; esac
  CUR=""
}
skip() { _rec "$1" skipped "$2"; info "-- [$1] skipped: $2"; }

# the command that picks the workflow up again at the wait for the full run
continue_cmd() {
  local c="train/all.sh $FOLDER --run $FULL_RUN" d
  [ "$EVAL" = 1 ] || c+=" --no-eval"
  [ "$PER_DEMO" = 8 ] || c+=" --per-demo $PER_DEMO"
  [ "$ATTENTION" = 0 ] || c+=" --attention"
  for d in ${RESULTS_DIRS[@]+"${RESULTS_DIRS[@]}"}; do c+=" --results-dir $(shq "$d")"; done
  printf '%s' "$c"
}

where_to_look() {
  info "log:   ${TRAIN_ALL_LOG:-(dry-run: none)}"
  info "steps: ${STEPS:-(dry-run: none)}"
}

# what to do after an interruption / unexpected exit during step $1
resume_advice() {
  if [ "$SUBMITTED" = 1 ]; then
    info "the cluster jobs keep running. Continue with: $(continue_cmd)"
  elif [ "$1" = submit ]; then
    info "it stopped while submitting, so a smoke job may be submitted without its full run. Check with"
    info "train/status.sh $FOLDER (and data/$FOLDER/policy/*/job.json): if the full run is there, continue with"
    info "train/all.sh $FOLDER --run <full run>; otherwise cancel what was submitted (ssh $CLUSTER_LOGIN scancel <job id>)"
    info "and run the same command again (with --pull instead of --package if dataproc already packaged $DATASET)."
  else
    info "nothing was submitted yet. Run the same command again (pull and upload resume; if dataproc already packaged"
    info "$DATASET, use --pull instead of --package)"
  fi
}

stop() {  # WHY [NEXT]: the current step failed and the workflow cannot go on
  local where=$CUR
  finish failed "$1"
  _rec all stopped "at $where: $1"
  info ""
  info "==================== all.sh STOPPED at step '$where'"
  info "why:   $1"
  [ -z "${2:-}" ] || info "next:  $2"
  where_to_look
  DONE=1
  exit 1
}

on_signal() {  # NAME NUMBER
  trap '' INT TERM HUP
  local where=${CUR:-between steps}
  [ -z "$CUR" ] || finish interrupted "stopped by SIG$1"
  _rec all interrupted "during $where (SIG$1)"
  info ""
  info "==================== all.sh INTERRUPTED during '$where' (SIG$1)"
  resume_advice "$where"
  where_to_look
  DONE=1
  exit $((128 + $2))
}
trap 'on_signal INT 2' INT
trap 'on_signal TERM 15' TERM
trap 'on_signal HUP 1' HUP

on_exit() {
  local rc=$?
  rm -rf "$TMPD"
  [ "$DONE" = 1 ] && return
  local why="all.sh exited unexpectedly (exit $rc) during '${CUR:-?}'; the output before this shows the last command"
  [ -z "$CUR" ] || _rec "$CUR" failed "$why"
  _rec all failed "$why"
  info ""
  info "==================== all.sh EXITED UNEXPECTEDLY (exit $rc) during step '${CUR:-?}'"
  resume_advice "$CUR"
  where_to_look
}
trap on_exit EXIT

# capture NAME CMD...: run CMD; its stdout is shown (and logged) and also kept in $TMPD/NAME; returns CMD's exit code
capture() { local f=$TMPD/$1; shift; "$@" | tee "$f"; return "${PIPESTATUS[0]}"; }
kv() { tr -d '\r' <"$TMPD/$1" | sed -n "s/^$2=//p" | head -1; }  # first line, like remote_kv
line_of() { tr -d '\r' <"$TMPD/$1" | sed -n "s/^$2: //p" | head -1; }
py() {  # python3 SCRIPT ARGS... (only printed in a dry-run)
  if [ "$DRY_RUN" = 1 ]; then show_cmd python3 "$@" >&2; return 0; fi
  python3 "$@"
}
slurm_log() {  # RUN KIND(train|eval) [JOB]: where the job's Slurm log is on the cluster
  local label job
  label=$(json_get "$POLICY/$1/job.json" label "?")
  job=${3:-$(json_get "$POLICY/$1/job.json" job_id "?")}
  printf '%s' "$CLUSTER_LOGIN:$(rpath "$CLUSTER_LOGS")/bp_$2_${label}_${job}.out"
}
# smoke run of a full run, from the job ids in the local job.json files ("after" = the smoke job)
find_smoke() {
  python3 -I - "$POLICY" "$1" <<'PY'
import json, os, sys
pol, run = sys.argv[1:3]
after = json.load(open(os.path.join(pol, run, "job.json"))).get("after")
for d in sorted(os.listdir(pol)) if after else []:
    try:
        j = json.load(open(os.path.join(pol, d, "job.json")))
    except (OSError, ValueError):
        continue
    if j.get("job_id") == after and j.get("mode") == "smoke":
        print(d)
        break
PY
}

DRY=()
[ "$DRY_RUN" = 0 ] || DRY=(--dry-run)
ROOT_ARG=(--data-root "$LOCAL_DATA_ROOT")

step "all.sh $(shq ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"})"
info "started $(date -Iseconds) on $(hostname); train/ at commit $(git -C "$TRAIN_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
where_to_look
_rec all started "$(shq ${ORIG_ARGS[@]+"${ORIG_ARGS[@]}"})"

# =========================================================================================== 1-4: data -> submitted
if [ -z "$RESUME" ]; then
  if [ "$PULL" = 1 ]; then
    if [ "$PACKAGE" = 1 ]; then what="dataproc package $DATASET on the Jetson, then pull it"
    else what="pull the existing package $DATASET"; fi
    begin pull "$what -> data/$FOLDER/"
    p=("$FOLDER")
    if [ "$PACKAGE" = 1 ]; then p+=(--package "$DATASET"); else p+=(--dataset "$DATASET"); fi
    [ "$NO_RAW" = 0 ] || p+=(--no-raw)
    bash "$TRAIN_DIR/pull.sh" "${p[@]}" ${DRY[@]+"${DRY[@]}"} ||
      stop "pull.sh failed (its error is above)" \
        "check the task on the Jetson: ssh ${JETSON_HOST-} \"${JETSON_ACTIVATE-} && dataproc status $FOLDER\" ('0 exported' = every episode failed processing, train/DIRECTIONS.md step 1). If dataproc already packaged $DATASET, rerun with --pull instead of --package."
    finish ok "data/$FOLDER/package/$DATASET/"
  else
    skip pull "no --pull/--package: using data/$FOLDER/package/$DATASET/ as it is"
  fi

  begin dataset_report "episodes, lengths and the training-time estimate of $DATASET"
  if py "$ANALYSIS/dataset_report.py" "$FOLDER" --dataset "$DATASET" "${ROOT_ARG[@]}"; then
    finish ok "data/$FOLDER/analysis/dataset_report/$DATASET/"
  else
    finish failed "dataset_report.py exited $? (training does not need it; going on)"
  fi

  begin upload "data/$FOLDER/package/$DATASET/ -> $CLUSTER_XFER"
  bash "$TRAIN_DIR/upload.sh" "$FOLDER" "$DATASET" ${DRY[@]+"${DRY[@]}"} ||
    stop "upload.sh failed (its error is above)" \
      "fix the cause and run the same command again: an interrupted upload resumes (if dataproc already packaged $DATASET, use --pull instead of --package)"
  finish ok "$(rpath "$CLUSTER_DATASETS")/<label>/$DATASET/"

  if [ "$SMOKE" = 1 ]; then what="a smoke run, then the full run after it"; else what="the full run only"; fi
  begin submit "$what"
  s=("$FOLDER" "$DATASET")
  [ "$SMOKE" = 0 ] || s+=(--smoke-first)
  [ -z "$POLICY" ] || s+=(--policy "$POLICY")
  [ -z "$EPOCHS" ] || s+=(--epochs "$EPOCHS")
  [ "$HAVE_NOTE" = 0 ] || s+=(--note "$NOTE")
  s+=(${DRY[@]+"${DRY[@]}"})
  [ ${#OVERRIDES[@]} = 0 ] || s+=(-- "${OVERRIDES[@]}")
  capture submit bash "$TRAIN_DIR/submit.sh" "${s[@]}" ||
    stop "submit.sh failed (its error is above)" \
      "check what was submitted with train/status.sh $FOLDER; a smoke job submitted without its full run can be cancelled with: ssh $CLUSTER_LOGIN scancel <job id>"
  if [ "$DRY_RUN" = 1 ]; then
    FULL_RUN="<full run>"
    [ "$SMOKE" = 0 ] || SMOKE_RUN="<smoke run>"
  else
    FULL_RUN=$(kv submit FULL_RUN)
    SMOKE_RUN=$(kv submit SMOKE_RUN)
    [ -n "$FULL_RUN" ] || stop "submit.sh printed no FULL_RUN= line (its output is above)" "see train/status.sh $FOLDER"
    SUBMITTED=1
  fi
  finish ok "full run $FULL_RUN${SMOKE_RUN:+, after the smoke run $SMOKE_RUN}"
  [ "$SUBMITTED" = 0 ] || info "if all.sh stops from here on, the jobs keep running; continue with: $(continue_cmd)"
else
  begin resume "continue the submitted run $RESUME (dataset $DATASET)"
  FULL_RUN=$RESUME
  SMOKE_RUN=$(find_smoke "$RESUME")
  SUBMITTED=1
  finish ok "full run $FULL_RUN${SMOKE_RUN:+, after the smoke run $SMOKE_RUN}"
fi

if [ "$DRY_RUN" = 1 ]; then
  info "[dry-run] then, once the jobs run:"
  show_cmd train/status.sh "$FOLDER" "$FULL_RUN" --wait >&2
  show_cmd train/fetch.sh "$FOLDER" "$FULL_RUN" >&2
  show_cmd python3 train/analysis/run_report.py "$FOLDER" "$FULL_RUN" >&2
  c=(python3 train/analysis/compare_runs.py "$FOLDER" --no-smoke)
  for d in ${RESULTS_DIRS[@]+"${RESULTS_DIRS[@]}"}; do c+=(--results-dir "$d"); done
  show_cmd "${c[@]}" >&2
  if [ "$EVAL" = 1 ]; then
    c=(train/analysis/eval.sh "$FOLDER" "$FULL_RUN" --ckpt top1 --per-demo "$PER_DEMO")
    [ "$ATTENTION" = 0 ] || c+=(--attention)
    show_cmd "${c[@]}" >&2
    show_cmd train/status.sh "$FOLDER" "$FULL_RUN" --eval top1 --wait >&2
    show_cmd train/fetch.sh "$FOLDER" "$FULL_RUN" --no-checkpoints >&2
    show_cmd python3 train/analysis/eval_report.py "$FOLDER" "$FULL_RUN" --ckpt top1 >&2
  fi
  DONE=1
  exit 0
fi

# =========================================================================================== 5-6: train -> laptop
# training_failed STATUS: find out why (the smoke run, or the full run itself), fetch the run's files for the logs, stop
training_failed() {
  local st=$1 cause next sst="" sjob fjob note
  fjob=$(json_get "$POLICY/$FULL_RUN/job.json" job_id "?")
  note=$(line_of train_status note)
  cause="the full run $FULL_RUN ended with STATUS=${st:-?}${note:+ ($note)}"
  next="its Slurm log: $(slurm_log "$FULL_RUN" train)"
  if [ -n "$SMOKE_RUN" ]; then
    step "[train] status of the smoke run $SMOKE_RUN"
    if capture smoke_status bash "$TRAIN_DIR/status.sh" "$FOLDER" "$SMOKE_RUN"; then sst=$(kv smoke_status STATUS); fi
    if [ -z "$sst" ]; then
      cause+="; the status of the smoke run $SMOKE_RUN could not be read (see the output before this)"
    elif [ "$sst" != ok ] || [[ $note == *"(--after)"* ]]; then
      # the smoke run is the cause: it failed, or its Slurm job did not succeed although it wrote STATUS=ok
      # (train.sbatch exits 1 on packaging problems), so Slurm never started the full job
      sjob=$(json_get "$POLICY/$SMOKE_RUN/job.json" job_id "?")
      if [ "$sst" != ok ]; then
        cause="the smoke run $SMOKE_RUN ended with STATUS=$sst, so the full run $FULL_RUN never started (smoke log tail in the output/log)"
      else
        cause="the smoke run $SMOKE_RUN wrote STATUS=ok but its Slurm job $sjob did not succeed (e.g. packaging problems), so the full run $FULL_RUN never started"
      fi
      next="the smoke run's Slurm log: $(slurm_log "$SMOKE_RUN" train "$sjob"). The full job $fjob may stay queued: cancel it with ssh $CLUSTER_LOGIN scancel $fjob"
      step "[train] fetching the smoke run's files (logs, README, no checkpoints)"
      if bash "$TRAIN_DIR/fetch.sh" "$FOLDER" "$SMOKE_RUN" --no-checkpoints; then
        next+=". The smoke run's files (README, logs) are in data/$FOLDER/policy/$SMOKE_RUN/"
      fi
      stop "$cause" "$next"
    fi
  fi
  if [ "$st" = failed ]; then
    step "[train] fetching the full run's files (logs, README, no checkpoints)"
    if bash "$TRAIN_DIR/fetch.sh" "$FOLDER" "$FULL_RUN" --no-checkpoints; then
      next+=". The run's files (README, logs) are in data/$FOLDER/policy/$FULL_RUN/"
    fi
  fi
  stop "$cause" "$next"
}

POLL_MIN=$(( ${TRAIN_STATUS_POLL_SEC:-300} / 60 ))
begin train "wait for $FULL_RUN${SMOKE_RUN:+ (after the smoke run $SMOKE_RUN)}, checking every $POLL_MIN min"
capture train_status bash "$TRAIN_DIR/status.sh" "$FOLDER" "$FULL_RUN" --wait ||
  stop "status.sh failed (its error is above)" "fix it, then continue with: $(continue_cmd)"
ST=$(kv train_status STATUS)
case $ST in
  ok) finish ok "$FULL_RUN finished" ;;
  timeout) finish failed "$FULL_RUN hit its time limit (SLURM_TIME=${SLURM_TIME-?}): going on with the checkpoints it packaged" ;;
  *) training_failed "$ST" ;;
esac

begin fetch "policy -> data/$FOLDER/policy/$FULL_RUN/"
bash "$TRAIN_DIR/fetch.sh" "$FOLDER" "$FULL_RUN" ||
  stop "fetch.sh failed (its error is above)" "fix the cause (disk space?), then continue with: $(continue_cmd) (the fetch resumes)"
CKPT=$(ls "$POLICY/$FULL_RUN"/checkpoints/top1_*.ckpt 2>/dev/null | head -1)
[ -n "$CKPT" ] ||
  stop "no checkpoints/top1_*.ckpt in data/$FOLDER/policy/$FULL_RUN/ after the fetch" \
    "the run packaged no checkpoint: see data/$FOLDER/policy/$FULL_RUN/ (README.md, logs/) and $(slurm_log "$FULL_RUN" train)"
finish ok "$CKPT"

# =========================================================================================== 7-8: reports
begin run_report "training curves and summary of $FULL_RUN"
if py "$ANALYSIS/run_report.py" "$FOLDER" "$FULL_RUN" "${ROOT_ARG[@]}"; then
  finish ok "data/$FOLDER/policy/$FULL_RUN/analysis/"
else
  finish failed "run_report.py exited $? (its error is above); rerun: python3 train/analysis/run_report.py $FOLDER $FULL_RUN"
fi

begin compare_runs "every full run of $FOLDER side by side"
c=("$ANALYSIS/compare_runs.py" "$FOLDER" --no-smoke "${ROOT_ARG[@]}")
for d in ${RESULTS_DIRS[@]+"${RESULTS_DIRS[@]}"}; do c+=(--results-dir "$d"); done
if py "${c[@]}"; then
  finish ok "data/$FOLDER/analysis/compare_runs/"
else
  finish failed "compare_runs.py exited $? (its error is above)"
fi

# =========================================================================================== 9: offline eval
EVAL_CMD="train/analysis/eval.sh $FOLDER $FULL_RUN --ckpt top1 --per-demo $PER_DEMO"
[ "$ATTENTION" = 0 ] || EVAL_CMD+=" --attention"
if [ "$EVAL" = 0 ]; then
  skip eval "--no-eval"
else
  begin eval_submit "offline eval of $FULL_RUN top1 (per-demo $PER_DEMO$([ "$ATTENTION" = 0 ] || echo ", attention videos"))"
  e=("$FOLDER" "$FULL_RUN" --ckpt top1 --per-demo "$PER_DEMO")
  [ "$ATTENTION" = 0 ] || e+=(--attention)
  if capture eval_submit bash "$TRAIN_DIR/analysis/eval.sh" "${e[@]}"; then
    EJOB=$(kv eval_submit JOB)
    finish ok "job $EJOB"

    begin eval_wait "wait for the eval job $EJOB, checking every $POLL_MIN min"
    EST=""
    if capture eval_status bash "$TRAIN_DIR/status.sh" "$FOLDER" "$FULL_RUN" --eval top1 --wait; then
      EST=$(kv eval_status STATUS)
      if [ "$EST" = ok ]; then
        finish ok "eval job $EJOB finished"
      else
        why="eval job $EJOB ended with STATUS=${EST:-?}"
        m=$(line_of eval_status message); [ -z "$m" ] || why+=" ($m)"
        m=$(line_of eval_status note); [ -z "$m" ] || why+=" ($m)"
        finish failed "$why; Slurm log: $(slurm_log "$FULL_RUN" eval "$EJOB"); rerun later: $EVAL_CMD"
      fi
    else
      finish failed "status.sh --eval failed (its error is above); check later: train/status.sh $FOLDER $FULL_RUN --eval top1"
    fi

    # fetched whatever the result, so eval_job.json (status, exit codes, message) is on the laptop too
    begin eval_fetch "eval results -> data/$FOLDER/policy/$FULL_RUN/eval/top1/"
    EFETCH=0
    if bash "$TRAIN_DIR/fetch.sh" "$FOLDER" "$FULL_RUN" --no-checkpoints; then
      EFETCH=1; finish ok "data/$FOLDER/policy/$FULL_RUN/eval/top1/"
    else
      finish failed "fetch.sh exited $? (its error is above); rerun: train/fetch.sh $FOLDER $FULL_RUN --no-checkpoints"
    fi

    if [ "$EST" = ok ] && [ "$EFETCH" = 1 ]; then
      begin eval_report "errors in mm/deg per demo"
      if py "$ANALYSIS/eval_report.py" "$FOLDER" "$FULL_RUN" --ckpt top1 "${ROOT_ARG[@]}"; then
        finish ok "data/$FOLDER/policy/$FULL_RUN/analysis/"
      else
        finish failed "eval_report.py exited $? (its error is above)"
      fi
    else
      skip eval_report "no successful eval results on the laptop"
    fi
  else
    finish failed "eval.sh exited $? (its error is above); rerun later: $EVAL_CMD"
    skip eval_wait "the eval was not submitted"
    skip eval_fetch "the eval was not submitted"
    skip eval_report "the eval was not submitted"
  fi
fi

# =========================================================================================== summary
info ""
if [ ${#PROBLEMS[@]} = 0 ]; then
  info "==================== all.sh DONE: every step ok"
  RC=0
else
  info "==================== all.sh DONE with ${#PROBLEMS[@]} problem(s); the policy is on the laptop"
  for p in "${PROBLEMS[@]}"; do info "  - $p"; done
  RC=2
fi
info "policy:  $CKPT"
info "reports: data/$FOLDER/policy/$FULL_RUN/analysis/, data/$FOLDER/analysis/"
info "cluster: once you no longer need it there, delete $(rpath "$CLUSTER_RUNS")/<label>/$FULL_RUN/ (home quota)"
where_to_look
_rec all "$([ "$RC" = 0 ] && echo ok || echo done_with_problems)" "${#PROBLEMS[@]} problem(s); policy $CKPT"
DONE=1
exit "$RC"
