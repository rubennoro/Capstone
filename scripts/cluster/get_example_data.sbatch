#!/bin/bash
#SBATCH --job-name=bp_data
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Download the iPhUMI example datasets and unzip the single-arm zarr (step B in CLUSTER_SETUP.md).
set -e
module load anaconda3/2024.06
source activate bp

BP=~/CapstoneProject/Capstone/behavior_prompting/behavior_prompting
DATA_DIR=$BP/train_network/datasets/umi/iphumi_example_zarr_datasets

mkdir -p "$DATA_DIR"
hf download austinpatel/iphumi_example_zarr_datasets --repo-type=dataset --local-dir "$DATA_DIR"
ls -la "$DATA_DIR"

cd "$BP/scripts"
python unzip_zarr.py -n "$SLURM_CPUS_PER_TASK" -i "$DATA_DIR/replay_buffer_test-data-042926-single.zarr.zip"
du -sh "$DATA_DIR"/*
