#!/bin/bash
#SBATCH --job-name=iphumi_proc
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Raw iPhone exports (session firstreal, left gripper, no task labels) -> trainable single-arm .zarr
# (step E in CLUSTER_SETUP.md). Usage: sbatch process_firstreal.sbatch <task_name>
set -e
TASK_NAME=${1:?pass the task name as the first argument}
SESSION=firstreal
OUT_SESSION=firstreal-single
REPO=~/CapstoneProject/Capstone
RAW_DIR=$REPO/data/raw/2026-10-01
DATA_OUT=$REPO/behavior_prompting/behavior_prompting/train_network/datasets/umi/$OUT_SESSION
NW=$SLURM_CPUS_PER_TASK

module load anaconda3/2024.06
source activate iphumi
cd $REPO/iPhUMI/python_package/iphumi/demonstration_processing

echo "== 1. group: copy raw exports into tmp_demonstrations"
python process_demos_iphone.py filters.session_name=$SESSION group.iphone_dir=$RAW_DIR stages="[group]"

echo "== 2. give every demo one task spanning the whole recording"
python $REPO/scripts/label_whole_demo_as_task.py \
  --demonstrations-dir tmp_demonstrations --session-name $SESSION --task-name "$TASK_NAME"

echo "== 3. remaining processing stages (AR tag detect, gripper calibrate, align, validate, label)"
python process_demos_iphone.py filters.session_name=$SESSION num_workers=$NW \
  skip_stages="[group,visualize]"

echo "== 4. dataset session + replay buffer"
python create_session_iphone.py input_session_filters="[$SESSION]" output_session_name=$OUT_SESSION sides=single
python build_umi_dataset_iphone.py session_dir=tmp_sessions/$OUT_SESSION replay_buffer.num_workers=$NW

echo "== 5. unzip into train_network/datasets and rename left -> right keys"
mkdir -p $DATA_OUT
python $REPO/behavior_prompting/behavior_prompting/scripts/unzip_zarr.py -n $NW \
  -i tmp_sessions/$OUT_SESSION/replay_buffer_$OUT_SESSION.zarr.zip -o $DATA_OUT
python $REPO/scripts/rename_zarr_side.py $DATA_OUT/replay_buffer_$OUT_SESSION.zarr --from-side left --to-side right
du -sh $DATA_OUT/*

echo "== 6. (optional) per-demo visualizations for sanity checking; may fail headless"
python process_demos_iphone.py filters.session_name=$SESSION num_workers=$NW stages="[visualize]" \
  || echo "visualize stage failed (non-fatal)"
echo "== DONE"
