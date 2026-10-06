#!/bin/bash
#SBATCH --job-name=iphumi_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Build the `iphumi` conda env used to turn raw iPhone exports into a zarr dataset (step E in CLUSTER_SETUP.md).
module load anaconda3/2024.06
cd ~/CapstoneProject/Capstone/iPhUMI/python_package
conda env remove -n iphumi -y || true
set -e
conda env create -f environment.yml
conda run -n iphumi python -c "import iphumi, zarr, cv2, open3d; print('iphumi env OK', zarr.__version__, cv2.__version__)"
