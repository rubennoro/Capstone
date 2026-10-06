#!/bin/bash
#SBATCH --job-name=bp_env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/noroian.r/CapstoneProject/logs/%x_%j.out
# Build the `bp` conda env used for training (behavior_prompting/environment.yml). Takes ~35 min.
# environment.yml pip-installs local packages with paths relative to behavior_prompting/, so run from there.
module load anaconda3/2024.06
cd ~/CapstoneProject/Capstone/behavior_prompting
conda env remove -n bp -y || true
set -e
conda env create -f environment.yml
conda run -n bp python -c "import torch, behavior_prompting; print(torch.__version__, torch.version.cuda)"
