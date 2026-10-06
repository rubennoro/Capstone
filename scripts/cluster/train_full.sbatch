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

# Checkpoints are ~6 GB each and a run keeps up to 5 (top-2 val, top-2 train loss, final), so the run dir
# goes on /scratch. Copy the checkpoint you want to keep back to home afterwards (slim it first).
RUN_DIR=/scratch/$USER/bp_runs/$(date +%Y.%m.%d)/$(date +%H.%M.%S)_${EXP_NAME}

cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network
accelerate launch --num_processes=1 train.py \
  --config-name=umi_policy_dunetp \
  task.dataset.dataset_path=$DATASET \
  task.dataset.cache_dir=/tmp/$USER/zarr_cache \
  exp_name=$EXP_NAME \
  group_tag=block_raise \
  logging.mode=offline \
  hydra.run.dir=$RUN_DIR \
  "$@"
