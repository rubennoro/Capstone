#!/bin/bash
#SBATCH --job-name=bp_process
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Raw iPhone exports -> trainable single-arm dataset (.zarr.zip) at ~/CapstoneProject/datasets/<task>/<dataset>/.
# Usually submitted by submit.sh. Usage:
#   sbatch process_dataset.sbatch <task> <dataset_name> <sessions,comma,separated> <raw_dir relative to repo> [sides]
# Demos recorded with "no label" get one task (<task>) spanning the whole recording; labeled demos keep their labels.
# The iPhUMI cache (tmp_demonstrations/) is shared, so demos already processed for another dataset are reused.
set -eo pipefail
TASK=${1:?task}
DATASET=${2:?dataset name}
SESSIONS=${3:?comma-separated session names}
RAW_DIR=${4:?raw export dir, relative to the repo}
SIDES=${5:-single}

REPO=~/CapstoneProject/Capstone
OUT_DIR=~/CapstoneProject/datasets/$TASK/$DATASET
NW=$SLURM_CPUS_PER_TASK
SESSION_LIST="[${SESSIONS}]"
cd $REPO/iPhUMI/python_package/iphumi/demonstration_processing

# Never overwrite an existing dataset or iPhUMI session of the same name
if [ -e "$OUT_DIR/$DATASET.zarr.zip" ] || [ -e "tmp_sessions/$DATASET" ]; then
  echo "ERROR: dataset '$DATASET' already exists ($OUT_DIR or tmp_sessions/$DATASET); pick a new name" >&2
  exit 1
fi

module load anaconda3/2024.06
source activate iphumi

echo "== 1. group: copy raw exports into tmp_demonstrations (already-copied demos are skipped)"
python process_demos_iphone.py filters.session_name="$SESSION_LIST" group.iphone_dir=$REPO/$RAW_DIR stages="[group]"

echo "== 2. label demos recorded without task labels as one '$TASK' task"
for SESSION in ${SESSIONS//,/ }; do
  python $REPO/scripts/label_whole_demo_as_task.py \
    --demonstrations-dir tmp_demonstrations --session-name "$SESSION" --task-name "$TASK"
done

echo "== 3. AR tag detect, gripper calibrate, align, validate, label"
python process_demos_iphone.py filters.session_name="$SESSION_LIST" num_workers=$NW skip_stages="[group,visualize]"

echo "== 4. dataset session + replay buffer"
python create_session_iphone.py input_session_filters="$SESSION_LIST" output_session_name=$DATASET sides=$SIDES
python build_umi_dataset_iphone.py session_dir=tmp_sessions/$DATASET replay_buffer.num_workers=$NW

echo "== 5. move the dataset to $OUT_DIR"
mkdir -p $OUT_DIR
mv tmp_sessions/$DATASET/replay_buffer_$DATASET.zarr.zip $OUT_DIR/$DATASET.zarr.zip
ls -la $OUT_DIR
echo "== DONE: $OUT_DIR/$DATASET.zarr.zip"
