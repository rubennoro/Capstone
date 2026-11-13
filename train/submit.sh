#!/bin/bash
# Laptop -> cluster: submit a training job on an uploaded dataset. Syncs train/cluster/ + config to CLUSTER_TOOLS,
# runs submit_job.sh there over ssh, and writes data/FOLDER/policy/<run>/job.json locally.
set -euo pipefail
# shellcheck source=lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

usage() {
  cat <<'EOF'
usage: train/submit.sh FOLDER DATASET [--smoke | --smoke-first] [--epochs N] [--topk K] [--note TEXT]
                       [--after JOBID] [--dry-run] [-- hydra overrides...]

  FOLDER, DATASET  the dataset uploaded with train/upload.sh FOLDER DATASET
  --smoke          only a short test run (3 epochs on SMOKE_PARTITION, eval every epoch)
  --smoke-first    a smoke run, then the full run, which starts only if the smoke run succeeded
  --epochs N       epochs of the full run (default DEFAULT_EPOCHS); lowered if they would not fit in SLURM_TIME
  --topk K         best checkpoints to keep (default TOPK)
  --note TEXT      free text stored with the run
  --after JOBID    start only after that Slurm job succeeded
  --dry-run        print the ssh/rsync commands, run and write nothing
  -- ...           extra Hydra overrides for train.py, e.g. -- training.lr_scheduler=constant

example: train/submit.sh block_raise br-nov13 --smoke-first --epochs 60 --note "first rig data"
EOF
}

FOLDER="" DATASET="" MODE=full EPOCHS="" TOPK_ARG="" NOTE="" AFTER=""
ARGS=() OVERRIDES=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --smoke-first) MODE=smoke-first; shift ;;
    --epochs) EPOCHS=${2:?--epochs needs a number}; shift 2 ;;
    --topk) TOPK_ARG=${2:?--topk needs a number}; shift 2 ;;
    --note) NOTE=${2?--note needs a text}; shift 2 ;;
    --after) AFTER=${2:?--after needs a job id}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; OVERRIDES=("$@"); break ;;
    -*) usage >&2; die "unknown option $1" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[ ${#ARGS[@]} = 2 ] || { usage >&2; exit 1; }
FOLDER=${ARGS[0]} DATASET=${ARGS[1]}
check_name folder "$FOLDER"
check_name dataset "$DATASET"
[ -z "$EPOCHS" ] || check_posint --epochs "$EPOCHS"
[ -z "$AFTER" ] || [[ $AFTER =~ ^[0-9]+$ ]] || die "--after must be a Slurm job id, got '$AFTER'"

load_config
need_var CLUSTER_LOGIN CLUSTER_TOOLS LOCAL_DATA_ROOT SLURM_TIME SMOKE_TIME TIME_MARGIN_MIN \
  MIN_PER_EPOCH_PER_10K_STEPS DEFAULT_EPOCHS TOPK
TOPK_=${TOPK_ARG:-$TOPK}
check_posint --topk "$TOPK_"

PKG=$LOCAL_DATA_ROOT/$FOLDER/package/$DATASET/package.json
N_STEPS=""
if [ -f "$PKG" ]; then
  LABEL=$(json_get "$PKG" label_dir)
  N_STEPS=$(json_get "$PKG" totals.n_steps)
  [ "$LABEL" != multi ] || die "$DATASET has several labels; only single-label datasets are supported"
  check_name label "$LABEL"
elif [ "$DRY_RUN" = 1 ]; then
  warn "$PKG is not there yet (it would be pulled first); epoch check skipped"
  LABEL="<label_dir>"
else
  die "missing $PKG (pull the dataset first: train/pull.sh $FOLDER --dataset $DATASET)"
fi

# slurm_minutes TIME: Slurm time limit (MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS) in whole minutes
slurm_minutes() {
  local t=$1 d=0 h=0 m=0 s=0
  local -a p
  if [[ $t == *-* ]]; then
    d=${t%%-*}; IFS=: read -ra p <<<"${t#*-}"
    h=${p[0]:-0} m=${p[1]:-0} s=${p[2]:-0}
  else
    IFS=: read -ra p <<<"$t"
    case ${#p[@]} in
      1) m=${p[0]} ;;
      2) m=${p[0]} s=${p[1]} ;;
      3) h=${p[0]} m=${p[1]} s=${p[2]} ;;
      *) die "cannot read Slurm time '$t'" ;;
    esac
  fi
  for x in "$d" "$h" "$m" "$s"; do [[ $x =~ ^[0-9]+$ ]] || die "cannot read Slurm time '$t'"; done
  echo $(( 10#$d * 1440 + 10#$h * 60 + 10#$m + 10#$s / 60 ))
}

hmm() { awk -v m="$1" 'BEGIN { m = int(m + 0.5); printf "%d:%02d", int(m / 60), m % 60 }'; }

# fit_epochs MODE REQUESTED: print the epochs to use (clamped to what fits in the job's time limit) + an estimate
fit_epochs() {
  local mode=$1 want=$2 limit tmin budget mpe max
  if [ "$mode" = smoke ]; then limit=$SMOKE_TIME; else limit=$SLURM_TIME; fi
  if [ -z "$N_STEPS" ]; then echo "$want"; return; fi
  tmin=$(slurm_minutes "$limit")
  budget=$((tmin - TIME_MARGIN_MIN))
  [ "$budget" -gt 0 ] || die "$mode: time limit $limit leaves no time after TIME_MARGIN_MIN=$TIME_MARGIN_MIN"
  mpe=$(awk -v r="$MIN_PER_EPOCH_PER_10K_STEPS" -v n="$N_STEPS" 'BEGIN { printf "%.4f", r * n / 10000 }')
  max=$(awk -v b="$budget" -v e="$mpe" 'BEGIN { if (e <= 0) print 999999; else print int(b / e) }')
  [ "$max" -ge 1 ] || die "$mode: one epoch (~$(hmm "$mpe")) does not fit in $limit minus $TIME_MARGIN_MIN min"
  if [ "$want" -gt "$max" ]; then
    warn "$mode: $want epochs do not fit in $limit (minus $TIME_MARGIN_MIN min margin); using $max"
    want=$max
  fi
  info "$mode: $want epochs x $(awk -v e="$mpe" 'BEGIN { printf "%.2f", e }') min ($N_STEPS steps) ~ $(hmm "$(awk -v w="$want" -v e="$mpe" 'BEGIN { print w * e }')") h:mm of training; limit $limit, max $max epochs"
  echo "$want"
}

# submit_one MODE EPOCHS AFTER: run submit_job.sh train on the login node; sets SUB_RUN and SUB_JOB
submit_one() {
  local mode=$1 epochs=$2 after=$3 cmd out
  cmd="bash $(rq "$CLUSTER_TOOLS/submit_job.sh") train $(qargs --folder "$FOLDER" --label "$LABEL" --dataset "$DATASET" \
    --mode "$mode" --epochs "$epochs" --topk "$TOPK_" --note "$NOTE")"
  [ -z "$after" ] || cmd+=" $(qargs --after "$after")"
  [ ${#OVERRIDES[@]} = 0 ] || cmd+=" -- $(qargs "${OVERRIDES[@]}")"
  step "Submitting $mode training ($epochs epochs${after:+, after job $after})"
  if [ "$DRY_RUN" = 1 ]; then
    login_ssh "$cmd"
    SUB_RUN="<$mode run>" SUB_JOB="<$mode job id>"
    return
  fi
  out=$(login_ssh "$cmd") || { printf '%s\n' "$out" >&2; die "submission failed (see above)"; }
  printf '%s\n' "$out" >&2
  SUB_RUN=$(remote_kv "$out" RUN)
  SUB_JOB=$(remote_kv "$out" JOB)
  [ -n "$SUB_RUN" ] && [ -n "$SUB_JOB" ] || die "submit_job.sh printed no RUN=/JOB= lines (see above)"
  check_name run "$SUB_RUN"
  local dir=$LOCAL_DATA_ROOT/$FOLDER/policy/$SUB_RUN
  mkdir -p "$dir"
  python3 -I - "$dir/job.json" "$FOLDER" "$LABEL" "$DATASET" "$SUB_RUN" "$mode" "$epochs" "$TOPK_" "$SUB_JOB" \
    "$after" "$NOTE" "$(date -Iseconds)" "${OVERRIDES[@]}" <<'PY'
import json, sys
(path, folder, label, dataset, run, mode, epochs, topk, job, after, note, submitted), overrides = sys.argv[1:13], sys.argv[13:]
rec = {"folder": folder, "label": label, "dataset": dataset, "run": run, "mode": mode, "epochs": int(epochs),
       "topk": int(topk), "job_id": job, "after": after or None, "note": note, "submitted": submitted,
       "overrides": overrides}
with open(path, "w") as f:
    json.dump(rec, f, indent=2)
    f.write("\n")
PY
  info "run $SUB_RUN, job $SUB_JOB -> wrote $dir/job.json"
}

FULL_EPOCHS=${EPOCHS:-$DEFAULT_EPOCHS}
SMOKE_EPOCHS=3
[ "$MODE" != smoke ] || SMOKE_EPOCHS=${EPOCHS:-3}
case $MODE in
  smoke) SMOKE_EPOCHS=$(fit_epochs smoke "$SMOKE_EPOCHS") ;;
  full) FULL_EPOCHS=$(fit_epochs full "$FULL_EPOCHS") ;;
  smoke-first) SMOKE_EPOCHS=$(fit_epochs smoke "$SMOKE_EPOCHS"); FULL_EPOCHS=$(fit_epochs full "$FULL_EPOCHS") ;;
esac

sync_tools
RUNS=()
case $MODE in
  smoke) submit_one smoke "$SMOKE_EPOCHS" "$AFTER"; RUNS+=("$SUB_RUN") ;;
  full) submit_one full "$FULL_EPOCHS" "$AFTER"; RUNS+=("$SUB_RUN") ;;
  smoke-first)
    submit_one smoke "$SMOKE_EPOCHS" "$AFTER"; RUNS+=("$SUB_RUN")
    submit_one full "$FULL_EPOCHS" "$SUB_JOB"; RUNS+=("$SUB_RUN") ;;
esac

step "Submitted"
for r in "${RUNS[@]}"; do
  info "  train/status.sh $FOLDER $r [--wait]"
  info "  train/fetch.sh $FOLDER $r          (once it finished)"
done
