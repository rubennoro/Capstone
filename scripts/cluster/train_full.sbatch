#!/bin/bash
#SBATCH --job-name=bp_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Full behavior-prompting training run (step D in CLUSTER_SETUP.md).
# Usage: sbatch train_full.sbatch <dataset .zarr.zip> <exp_name> [extra hydra overrides...]
# When training ends, the job packages its results into ~/CapstoneProject/results/<exp_name>/ (home, never
# purged): plots + CSVs, videos, config, logs, and a slimmed copy of the best checkpoint.
# Smoke test on block_raise (45 demos): ~2.1 min/epoch of training on one H200, so 140 epochs ~5.2 h.
set -eo pipefail
DATASET=${1:?pass the dataset .zarr.zip path}
EXP_NAME=${2:?pass an experiment name}
shift 2

module load anaconda3/2024.06
source activate bp
# Headless node: keep mujoco/pygame (imported by the training code) from looking for a display
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
# LIBERO asks an interactive Y/N question on first import; no-op once ~/.libero/config.yaml exists
echo n | python -c "from libero.libero import get_libero_path"

nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 60 \
  > ~/CapstoneProject/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv &
GPU_LOG_PID=$!
trap "kill $GPU_LOG_PID" EXIT

# Only the best checkpoint is kept: top-1 by val action MSE, no train-loss checkpoints, and the forced final-epoch
# latest.ckpt is deleted after packaging. Checkpoints are ~6.4 GB, so the run dir goes on /scratch (purged
# periodically); a slimmed copy of the best one (no optimizer state) is saved into the results folder in home.
RUN_DIR=/scratch/$USER/bp_runs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_${EXP_NAME}

REPO=~/CapstoneProject/Capstone
RESULTS=~/CapstoneProject/results/$EXP_NAME
JOB_START=$(date +%s)

cd $REPO/behavior_prompting/behavior_prompting/train_network
accelerate launch --num_processes=1 train.py \
  --config-name=umi_policy_dunetp \
  task.dataset.dataset_path=$DATASET \
  task.dataset.cache_dir=/tmp/$USER/zarr_cache \
  exp_name=$EXP_NAME \
  group_tag=block_raise \
  logging.mode=offline \
  hydra.run.dir=$RUN_DIR \
  checkpoint.topk.k=1 \
  checkpoint.save_train_loss_ckpt_every=null \
  "$@"

echo "== Packaging results into $RESULTS"
set +e  # package whatever exists even if one step fails
mkdir -p $RESULTS
cp -r $RUN_DIR/vis $RUN_DIR/.hydra $RUN_DIR/normalizer.pkl $RESULTS/
cp $RUN_DIR/train.log $RESULTS/train_log.txt
cp ~/CapstoneProject/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}*.csv $RESULTS/ 2>/dev/null

# Best checkpoint = lowest val action MSE in its filename; slimmed (no optimizer state) into home
BEST=$(ls $RUN_DIR/checkpoints/*val_action_mse_error_all=*.ckpt 2>/dev/null \
  | sed 's/.*=\([0-9.]*\)\.ckpt$/\1 &/' | sort -g | head -1 | cut -d' ' -f2)
if [ -n "$BEST" ] && python $REPO/scripts/slim_checkpoint.py "$BEST" "$RESULTS/best_slim_$(basename "$BEST")"; then
  # Keep only the best checkpoint (full size on scratch, slimmed in home); remove latest.ckpt and anything else
  find $RUN_DIR/checkpoints -name '*.ckpt' ! -samefile "$BEST" -print -delete
fi

# Plots + CSVs from this job's offline wandb run (the newest one created after the job started)
WANDB_RUN=$(find wandb -maxdepth 1 -name 'offline-run-*' -newermt "@$JOB_START" | sort | tail -1)
if [ -n "$WANDB_RUN" ]; then
  echo "$WANDB_RUN" > $RESULTS/wandb_run_path.txt
  python $REPO/scripts/wandb_offline_to_csv.py "$WANDB_RUN" -o $RESULTS/plots \
    --gpu-csv ~/CapstoneProject/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv
fi
ls -la $RUN_DIR/checkpoints > $RESULTS/checkpoints_on_scratch.txt
du -sh $RESULTS/*
echo "== Done. Results: $RESULTS   Run dir (full-size best checkpoint): $RUN_DIR"
