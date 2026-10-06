#!/bin/bash
#SBATCH --job-name=bp_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Train behavior prompting on one dataset, package the standard results folder, and push it to GitHub.
# Usually submitted by submit.sh (which picks the run name and, for smoke runs, --partition=gpu-short --time=02:00:00).
# Usage:
#   sbatch train.sbatch <task> <dataset> <run_name> <full|smoke> "<note>" [extra hydra overrides...]
# Reads  ~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip
# Writes results/<task>/<run_name>/ in the repo clone (pushed; best checkpoint gitignored, kept on the cluster).
# Packaging always runs: on success, on failure, and 10 min before the time limit (USR1 signal).
set -o pipefail
TASK=${1:?task}
DATASET=${2:?dataset}
RUN_NAME=${3:?run name}
MODE=${4:?full or smoke}
NOTE=${5:-}
shift 5

REPO=~/CapstoneProject/Capstone
DATASET_ZIP=~/CapstoneProject/datasets/$TASK/$DATASET/$DATASET.zarr.zip
RESULTS=$REPO/results/$TASK/$RUN_NAME
RUN_DIR=/scratch/$USER/bp_runs/$TASK/$RUN_NAME
SLURM_LOG=~/CapstoneProject/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
GPU_CSV=~/CapstoneProject/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv
CONFIG_NAME=umi_policy_dunetp

case $MODE in
  full)  MODE_ARGS=() ;;
  # 3 epochs with val/rollout every epoch so every code path runs once
  smoke) MODE_ARGS=(training.num_epochs=3 training.val_every=1 training.rollout_every=1 training.sample_every=1) ;;
  *) echo "ERROR: mode must be full or smoke" >&2; exit 1 ;;
esac

module load anaconda3/2024.06
source activate bp
# Headless node: keep mujoco/pygame (imported by the training code) from looking for a display
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
# LIBERO asks an interactive Y/N question on first import; no-op once ~/.libero/config.yaml exists
echo n | python -c "from libero.libero import get_libero_path"

mkdir -p $RUN_DIR $RESULTS
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
python - <<EOF
import json
json.dump({'job_id': '$SLURM_JOB_ID', 'task': '$TASK', 'dataset': '$DATASET', 'run': '$RUN_NAME', 'mode': '$MODE',
           'config_name': '$CONFIG_NAME', 'partition': '$SLURM_JOB_PARTITION', 'node': '$SLURMD_NODENAME',
           'gpu_name': '$GPU_NAME', 'dataset_zip': '$DATASET_ZIP'}, open('$RUN_DIR/job_info.json', 'w'), indent=2)
EOF

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > $GPU_CSV &
GPU_LOG_PID=$!

# ---- train (in its own process group so the timeout handler can stop all of it) ----
STATUS=ok
on_timeout() {
  echo "== Time limit approaching: stopping training and packaging what exists"
  STATUS=timeout
  kill -TERM -- -$TRAIN_PID 2>/dev/null
}
trap on_timeout USR1

cd $REPO/behavior_prompting/behavior_prompting/train_network
setsid accelerate launch --num_processes=1 train.py \
  --config-name=$CONFIG_NAME \
  task.dataset.dataset_path=$DATASET_ZIP \
  task.dataset.cache_dir=/tmp/$USER/zarr_cache \
  exp_name=$RUN_NAME \
  group_tag=$TASK \
  logging.mode=offline \
  +logging.dir=$RUN_DIR \
  hydra.run.dir=$RUN_DIR \
  checkpoint.topk.k=1 \
  checkpoint.save_train_loss_ckpt_every=null \
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
kill $GPU_LOG_PID 2>/dev/null
echo "== Training finished with status: $STATUS (exit code $RC)"

# ---- package the standard results folder ----
python $REPO/scripts/package_run.py --run-dir $RUN_DIR --out $RESULTS --status $STATUS \
  --gpu-csv $GPU_CSV --slurm-log $SLURM_LOG --note "$NOTE"
PKG_RC=$?

# ---- commit + push only this run's folder (flock: jobs share one clone) ----
cd $REPO
PUSHED=0
(
  flock 9
  [ -f results/$TASK/README.md ] && git add -- results/$TASK/README.md
  git add -- results/$TASK/$RUN_NAME
  git commit -q -m "Results: $TASK/$RUN_NAME ($STATUS) [cluster job $SLURM_JOB_ID]" -- results/$TASK
  for attempt in 1 2 3 4 5; do
    git pull -q --rebase --autostash && git push -q && exit 0
    echo "push attempt $attempt failed; retrying"
    sleep $((attempt * 15))
  done
  exit 1
) 9>~/CapstoneProject/.git_push.lock && PUSHED=1
echo "== Packaged (exit $PKG_RC), pushed: $PUSHED -> $RESULTS"

# ---- clean up scratch only when everything worked; keep the raw wandb run in home ----
if [ "$STATUS" = ok ] && [ $PKG_RC -eq 0 ] && [ $PUSHED -eq 1 ]; then
  mkdir -p ~/CapstoneProject/wandb_runs/$TASK
  mv $RUN_DIR/wandb ~/CapstoneProject/wandb_runs/$TASK/$RUN_NAME 2>/dev/null
  rm -rf $RUN_DIR
fi

[ "$STATUS" = ok ] && [ $PKG_RC -eq 0 ]
