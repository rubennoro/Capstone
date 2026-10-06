#!/bin/bash
# Short behavior-prompting training run on the example data (step C in CLUSTER_SETUP.md).
# Usage: bash train_smoke.sh [dataset_path relative to train_network/] [extra hydra overrides...]
# Defaults to the example single-arm dataset. To train straight from a .zarr.zip (no slow unzip on the
# networked filesystem), pass the zip plus task.dataset.cache_dir=/tmp/$USER/zarr_cache: the dataset is
# copied once into an LMDB on the node-local disk, which multiple dataloader workers can read safely.
# 3 epochs with val/rollout/checkpoint every epoch, so every code path runs once in a short time.
set -eo pipefail
DATASET=${1:-datasets/umi/iphumi_example_zarr_datasets/replay_buffer_test-data-042926-single.zarr}
shift || true
EXP_NAME=smoke_$(basename "${DATASET%.zip}" .zarr)
module load anaconda3/2024.06
source activate bp
# Headless node: keep mujoco/pygame (imported by the training code) from looking for a display
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# Log GPU memory every 30 s in the background
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 \
  > ~/CapstoneProject/logs/smoke_gpu.csv &
GPU_LOG_PID=$!
trap "kill $GPU_LOG_PID" EXIT

# LIBERO asks an interactive Y/N question the first time it's imported (to create ~/.libero/config.yaml);
# answer "n" (use default paths) so batch jobs don't crash with EOFError. No-op once the config exists.
echo n | python -c "from libero.libero import get_libero_path"

cd ~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting/train_network
accelerate launch --num_processes=1 train.py \
  --config-name=umi_policy_dunetp \
  task.dataset.dataset_path=$DATASET \
  exp_name=$EXP_NAME \
  group_tag=smoke_tests \
  logging.mode=offline \
  training.num_epochs=3 \
  training.val_every=1 \
  training.rollout_every=1 \
  training.sample_every=1 \
  checkpoint.save_train_loss_ckpt_every=1 \
  "$@" \
  2>&1 | tee ~/CapstoneProject/logs/smoke_train.log
