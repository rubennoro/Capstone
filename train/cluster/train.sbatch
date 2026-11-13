#!/bin/bash
# Safe defaults only: submit_job.sh passes every resource (partition, gres, time, cpus, mem, account, output, job name,
# signal, dependency) on the sbatch command line from config.env, and command-line options win over #SBATCH lines.
#SBATCH --job-name=bp_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
# Train behavior prompting on one packaged dataset and package the run folder that train/fetch.sh brings back.
# Derived from scripts/cluster/train.sbatch (same env setup, LIBERO/EGL handling, GPU csv, USR1 timeout handling,
# packaging always runs), but: hydra run dir on scratch, results in $HOME/CLUSTER_RUNS/<label>/<run>/, top-K slim
# checkpoints (K=TOPK, default 1 like the old checkpoint.topk.k=1),
# the raw wandb offline run copied into the run folder, and no git commit/push.
#
# Usage (normally via submit_job.sh train):
#   sbatch ... train.sbatch <run dir> <label> <dataset> <run> <full|smoke> <epochs> <topk> "<note>" [hydra overrides...]
# Config comes only from <run dir>/run_config/submit_config.env, the copy submit_job.sh froze at submission, so
# re-syncing CLUSTER_TOOLS later cannot move a queued job's results.
#
# Run folder FINAL = <run dir> = $HOME/CLUSTER_RUNS/<label>/<run>/:
#   package_run.py output (README.md summary.json epochs.csv history.csv plots/ videos/ logs/ run_config/)
#   checkpoints/top{rank}_<original name>.ckpt (slimmed) + checkpoints/checkpoints.json
#   wandb/offline-run-*/   run_config/hydra/ (the whole .hydra dir)   train_meta.json (submit_job.sh)   STATUS
# STATUS: running at start; ok|failed|timeout written at the very end, after packaging.
# Scratch run dir /scratch/$USER/bp_runs/<label>/<run> is deleted only if STATUS=ok, packaging worked and every top-K
# slim succeeded; otherwise it is kept and its path is written to FINAL/logs/checkpoints_on_scratch.txt.
#
# When top-K checkpoints exist (TrainPolicyWorkspace, lines ~275-321 and ~610-619): the monitor key is
# checkpoint.topk.monitor_key = val_action_mse_error_all (umi_policy_base.yaml), which needs training.sample_every, so
# candidates are saved only at epochs where epoch % sample_every == 0 (0-indexed: 0, 10, 20, ... with the dunetp
# default sample_every=10) plus the last epoch. A 60-epoch run has 7 candidates (0,10,...,50,59) for the K slots
# (K=1 by default: the best one); smoke (sample_every=1) has one per epoch. Fallbacks in the workspace, each renaming the file key via format_str:
#   sample_every=-1 -> val_loss (every val_every); empty val split -> train_action_mse_error_all (every sample_every);
#   no val at all -> train_loss, for which no top-K manager exists (and save_train_loss_ckpt_every=null here), so the
#   only checkpoint is latest.ckpt, written at the final epoch only (save_last_ckpt: False). A timed-out run has no
#   latest.ckpt. Packaging below ranks by whichever key the files use, and uses latest.ckpt only if there is no top-K
#   file at all (checkpoints.json "source": "latest").
set -o pipefail
[ $# -ge 8 ] || { echo "ERROR: train.sbatch needs 8 arguments before the overrides, got $#" >&2; exit 1; }
FINAL=${1:?run dir}
CONF=$FINAL/run_config/submit_config.env
# shellcheck disable=SC2034  # read by config.env (LOCAL_DATA_ROOT default)
TRAIN_DIR=$(dirname "$CONF")
# shellcheck source=/dev/null
source "$CONF" || { echo "ERROR: cannot read $CONF" >&2; exit 1; }
# shellcheck disable=SC2088  # a literal '~/' prefix in a config value, not a tilde to expand
abs() { case $1 in /*) echo "$1" ;; '~/'*) echo "$HOME/${1#\~/}" ;; *) echo "$HOME/$1" ;; esac; }
# Positional args are read AFTER config.env, so a config key with the same name (TOPK!) cannot clobber them.
LABEL=${2:?label}
DATASET=${3:?dataset}
RUN_NAME=${4:?run name}
MODE=${5:?full or smoke}
EPOCHS=${6:?epochs}
TOPK=${7:-1}  # 1 = one checkpoint per run (config.env default; ~2.5 GB slim)
NOTE=${8-}
shift 8

REPO=$(abs "$CLUSTER_REPO")
DATASET_ZIP=$(abs "$CLUSTER_DATASETS")/$LABEL/$DATASET/$DATASET.zarr.zip
LOGS=$(abs "$CLUSTER_LOGS")
SCRATCH=${BP_SCRATCH:-/scratch/$USER}  # BP_SCRATCH exists only for the local tests
RUN_DIR=$SCRATCH/bp_runs/$LABEL/$RUN_NAME
SLURM_LOG=$LOGS/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
GPU_CSV=$LOGS/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv
CONFIG_NAME=umi_policy_dunetp

case $MODE in
  full)  MODE_ARGS=() ;;
  # val/rollout/sample every epoch so every code path runs once (epochs come from --epochs, 3 for smoke)
  smoke) MODE_ARGS=(training.val_every=1 training.rollout_every=1 training.sample_every=1) ;;
  *) echo "ERROR: mode must be full or smoke" >&2; exit 1 ;;
esac

mkdir -p "$FINAL" "$RUN_DIR" "$LOGS"
echo running > "$FINAL/STATUS"
echo "== $LABEL/$RUN_NAME ($MODE, $EPOCHS epochs, top-$TOPK) job $SLURM_JOB_ID on ${SLURMD_NODENAME:-?}"
echo "   dataset $DATASET_ZIP"
echo "   scratch $RUN_DIR -> $FINAL"

module load "$CLUSTER_CONDA_MODULE"
# shellcheck source=/dev/null
source activate "$CLUSTER_CONDA_ENV"
# Headless node: keep mujoco/pygame (imported by the training code) from looking for a display
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
# LIBERO asks an interactive Y/N question on first import; no-op once ~/.libero/config.yaml exists
echo n | python -c "from libero.libero import get_libero_path"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
python - "$RUN_DIR/job_info.json" "$SLURM_JOB_ID" "$LABEL" "$DATASET" "$RUN_NAME" "$MODE" "$CONFIG_NAME" \
  "${SLURM_JOB_PARTITION-}" "${SLURMD_NODENAME-}" "$GPU_NAME" "$DATASET_ZIP" "$EPOCHS" "$TOPK" <<'EOF'
import json, sys
keys = ['job_id', 'task', 'dataset', 'run', 'mode', 'config_name', 'partition', 'node', 'gpu_name', 'dataset_zip',
        'epochs', 'topk']
json.dump(dict(zip(keys, sys.argv[2:])), open(sys.argv[1], 'w'), indent=2)
EOF

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$GPU_CSV" &
GPU_LOG_PID=$!

# ---- train (in its own process group so the timeout handler can stop all of it) ----
STATUS=ok
on_timeout() {
  echo "== Time limit approaching: stopping training and packaging what exists"
  STATUS=timeout
  kill -TERM -- -"$TRAIN_PID" 2>/dev/null
}
trap on_timeout USR1

# Fixed overrides first, user overrides ("$@") last: Hydra 1.2.0 applies `key=value` overrides in order, so a user
# override of any fixed key wins (verified with hydra-core 1.2.0). submit_job.sh refuses hydra.run.dir, logging.dir and
# task.dataset.dataset_path, which this script needs to own.
cd "$REPO/behavior_prompting/behavior_prompting/train_network" || { echo "ERROR: clone missing at $REPO" >&2; STATUS=failed; }
if [ "$STATUS" = ok ]; then
  setsid accelerate launch --num_processes=1 train.py \
    --config-name=$CONFIG_NAME \
    task.dataset.dataset_path="$DATASET_ZIP" \
    task.dataset.cache_dir="/tmp/$USER/zarr_cache" \
    exp_name="$RUN_NAME" \
    group_tag="$LABEL" \
    logging.mode=offline \
    +logging.dir="$RUN_DIR" \
    hydra.run.dir="$RUN_DIR" \
    checkpoint.topk.k="$TOPK" \
    checkpoint.save_train_loss_ckpt_every=null \
    training.num_epochs="$EPOCHS" \
    "${MODE_ARGS[@]}" \
    "$@" &
  TRAIN_PID=$!
  wait $TRAIN_PID
  RC=$?
  if [ "$STATUS" = timeout ]; then
    wait $TRAIN_PID 2>/dev/null
  elif [ $RC -ne 0 ]; then
    STATUS=failed
  fi
else
  RC=1
fi
# Slurm sends USR1 once; if it comes during packaging, its default action would kill the job before STATUS is written
trap 'echo "== USR1 during packaging: finishing the packaging"' USR1
kill $GPU_LOG_PID 2>/dev/null
echo "== Training finished with status: $STATUS (exit code $RC)"

# ---- package (scripts/package_run.py, unchanged; checkpoints are handled below) ----
PROBLEMS=()
python "$REPO/scripts/package_run.py" --run-dir "$RUN_DIR" --out "$FINAL" --status $STATUS --no-checkpoint \
  --gpu-csv "$GPU_CSV" --slurm-log "$SLURM_LOG" --note="$NOTE"
PKG_RC=$?
[ $PKG_RC -eq 0 ] || PROBLEMS+=("package_run.py exited with $PKG_RC")

# ---- top-K checkpoints, slimmed (no optimizer state) ----
# Files are epoch=NNNN-<key>=<value>.ckpt. Rank by the first key in this list that has files (lowest value first;
# all four are 'min' metrics); the workspace falls back along the same order.
CKPT_SRC=$RUN_DIR/checkpoints
CKPT_OUT=$FINAL/checkpoints
mkdir -p "$CKPT_OUT"
SELECTED=() METRIC=
for key in val_action_mse_error_all train_action_mse_error_all val_loss train_loss; do
  lines=()
  for f in "$CKPT_SRC"/epoch=*-"$key"=*.ckpt; do
    base=${f##*/}
    [[ -f $f && $base =~ ^epoch=([0-9]+)-${key}=([0-9][0-9.eE+-]*)\.ckpt$ ]] || continue
    lines+=("${BASH_REMATCH[2]}|$((10#${BASH_REMATCH[1]}))|$base")
  done
  if [ ${#lines[@]} -gt 0 ]; then
    METRIC=$key
    mapfile -t SELECTED < <(printf '%s\n' "${lines[@]}" | sort -t'|' -k1,1g -k2,2n | head -n "$TOPK")
    break
  fi
done
SOURCE=topk
if [ -z "$METRIC" ]; then
  if [ -f "$CKPT_SRC/latest.ckpt" ]; then
    echo "== No top-K checkpoint files in $CKPT_SRC: using latest.ckpt as top1 (checkpoints.json source=latest)"
    SELECTED=("||latest.ckpt")
    SOURCE=latest
  else
    echo "== No checkpoints in $CKPT_SRC"
  fi
elif [ "$METRIC" != val_action_mse_error_all ]; then
  echo "== NOTE: no val_action_mse_error_all checkpoints; ranking the top-K files by $METRIC from their filenames"
fi

ENTRIES=() SLIM_FAILED=()
rank=0
for line in ${SELECTED[@]+"${SELECTED[@]}"}; do
  IFS='|' read -r value epoch base <<< "$line"
  rank=$((rank + 1))
  out_name=top${rank}_$base
  echo "== Slimming $base -> checkpoints/$out_name"
  if python "$REPO/scripts/slim_checkpoint.py" "$CKPT_SRC/$base" "$CKPT_OUT/$out_name" && [ -s "$CKPT_OUT/$out_name" ]; then
    size=$(stat -c %s "$CKPT_OUT/$out_name")
    num='^-?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$'
    [[ $value =~ $num ]] || value=null
    [ -n "$epoch" ] || epoch=null
    mse=null; [ "$METRIC" = val_action_mse_error_all ] && mse=$value
    ENTRIES+=("$(printf '  {"rank": %d, "file": "%s", "epoch": %s, "val_action_mse": %s, "size_bytes": %s, "metric": %s, "value": %s, "source": "%s", "original": "%s"}' \
      "$rank" "$out_name" "$epoch" "$mse" "$size" "$( [ -n "$METRIC" ] && printf '"%s"' "$METRIC" || echo null)" \
      "$value" "$SOURCE" "$base")")
  else
    rm -f "$CKPT_OUT/$out_name"
    SLIM_FAILED+=("$base")
    PROBLEMS+=("slimming $base failed")
  fi
done
{
  echo "["
  for i in "${!ENTRIES[@]}"; do
    [ "$i" -lt $((${#ENTRIES[@]} - 1)) ] && echo "${ENTRIES[$i]}," || echo "${ENTRIES[$i]}"
  done
  echo "]"
} > "$CKPT_OUT/checkpoints.json"
echo "== $((${#ENTRIES[@]})) checkpoint(s) packaged, ${#SLIM_FAILED[@]} failed (checkpoints/checkpoints.json)"

# ---- raw wandb offline run + the whole .hydra dir ----
mkdir -p "$FINAL/wandb"
for d in "$RUN_DIR"/wandb/offline-run-*; do
  [ -d "$d" ] || continue
  cp -a "$d" "$FINAL/wandb/" || PROBLEMS+=("copying $d failed")
done
if [ -d "$RUN_DIR/.hydra" ]; then  # package_run.py copies only config.yaml + overrides.yaml into run_config/
  mkdir -p "$FINAL/run_config/hydra"
  cp -a "$RUN_DIR/.hydra/." "$FINAL/run_config/hydra/" || PROBLEMS+=("copying .hydra failed")
fi

# ---- scratch: delete only when everything worked ----
if [ "$STATUS" = ok ] && [ ${#PROBLEMS[@]} -eq 0 ]; then
  rm -rf "$RUN_DIR"
  echo "== Deleted scratch run dir $RUN_DIR"
else
  mkdir -p "$FINAL/logs"
  {
    echo "$RUN_DIR"
    echo "# kept because: status=$STATUS${PROBLEMS[*]:+; ${PROBLEMS[*]}}"
    for f in "$CKPT_SRC"/*.ckpt; do [ -f "$f" ] && echo "# checkpoint: ${f##*/}"; done
    echo "# scratch is purged eventually; fetch or re-package (python scripts/package_run.py ...) from it soon"
  } > "$FINAL/logs/checkpoints_on_scratch.txt"
  echo "== Kept scratch run dir $RUN_DIR (logs/checkpoints_on_scratch.txt)"
fi

echo "$STATUS" > "$FINAL/STATUS"
echo "== Packaged (status $STATUS${PROBLEMS[*]:+, problems: ${PROBLEMS[*]}}) -> $FINAL"
[ "$STATUS" = ok ] && [ ${#PROBLEMS[@]} -eq 0 ]
