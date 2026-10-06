# block_raise: run_001_2026-10-05

**Status:** completed  ·  **Mode:** full  ·  **Job:** 10851655  ·  **Packaged:** (copied from `~/CapstoneProject/results/block_raise_bp_v1/`) 2026-10-06T04:17:44

> First full run (`block_raise_bp_v1`), 140 epochs. Ran on the old `train_full.sbatch`; its output was copied into this layout by hand.

## Data

| | |
|---|---|
| Dataset | `firstreal-single` |
| Path | `/home/noroian.r/CapstoneProject/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip` |
| Train / val split | val_ratio 0.05 per task, seed 42 |

## Training

| | |
|---|---|
| Policy config | `umi_policy_dunetp` |
| Epochs | 140 completed of 140 configured |
| Batch size | 70 |
| Learning rate | 5e-05 (cosine, warmup 0.05) |
| Precision | bf16 |
| Evaluate / rollout every | 10 / 10 epochs |
| Hardware | NVIDIA H200 on `d4055` (gpu) |
| Command-line overrides | `task.dataset.dataset_path=/home/noroian.r/CapstoneProject/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip`<br>`task.dataset.cache_dir=/tmp/noroian.r/zarr_cache`<br>`exp_name=block_raise_bp_v1`<br>`group_tag=block_raise`<br>`logging.mode=offline`<br>`checkpoint.topk.k=1`<br>`checkpoint.save_train_loss_ckpt_every=null` |

## Results

| Epoch | Train loss | Val loss | Val action MSE |
|---|---|---|---|
| 0 | 0.6617 | 0.2488 | 0.2483 |
| 10 | 0.0200 | 0.0199 | 0.000774 |
| 20 | 0.0153 | 0.0180 | 0.000266 |
| 30 | 0.0141 | 0.0165 | 0.000174 |
| 40 | 0.0132 | 0.0191 | 0.000172 |
| **50** | 0.0126 | 0.0173 | **0.000148** |
| 60 | 0.0121 | 0.0188 | 0.000163 |
| 70 | 0.0118 | 0.0185 | 0.000169 |
| 80 | 0.0114 | 0.0197 | 0.000167 |
| 90 | 0.0111 | 0.0191 | 0.000167 |
| 100 | 0.0109 | 0.0210 | 0.000163 |
| 110 | 0.0107 | 0.0242 | 0.000171 |
| 120 | 0.0106 | 0.0215 | 0.000168 |
| 130 | 0.0105 | 0.0224 | 0.000170 |
| 139 | 0.0105 | 0.0228 | 0.000174 |

- **Speed:** 4.68 steps/s, 518 steps/epoch ≈ 2.1 min/epoch including evaluation
- **Training time:** 4 h 55 min (job wall time)
- **Peak GPU memory:** 34.1 GB
- **Best checkpoint:** epoch 50, val_action_mse_error_all = 0.000148: `checkpoints/best_slim_epoch=0050-val_action_mse_error_all=0.0001481.ckpt` (slimmed, 2.5 GB; gitignored, on the laptop and in `~/CapstoneProject/results/block_raise_bp_v1/`). The full-size copy (6.4 GB, with optimizer state) is in `/scratch/noroian.r/bp_runs/2026.10.05/23.22.47_block_raise_bp_v1/checkpoints/`.
- **Observation:** val action MSE plateaus after about epoch 30, and val loss rises slowly from about epoch 50 while train loss keeps falling, which suggests mild overfitting. The last ~90 epochs added nothing.

## Files

| Path | Contents |
|---|---|
| `plots/loss_per_step.png` | Train loss at every gradient step (log scale) |
| `plots/loss_per_epoch.png` | Epoch-average train loss vs val loss |
| `plots/action_mse_per_epoch.png` | Action error split into position / rotation / gripper width, train vs val |
| `plots/lr_and_grad_norm.png` | Learning-rate schedule and gradient norm |
| `plots/gpu_memory.png` | GPU memory over the job |
| `epochs.csv`, `history.csv` | Per-epoch metrics; every logged step |
| `summary.json` | Machine-readable summary of this page |
| `videos/` | Rollout + attention videos: `final` only (the old sbatch didn't keep per-epoch videos) |
| `run_config/` | Hydra config + overrides, normalizer, job_info.json |
| `logs/` | Training log, Slurm console log (progress bars stripped), GPU memory samples, wandb offline run path, scratch checkpoint location |
| `checkpoints/` | Best checkpoint only, slimmed (no optimizer state); gitignored |

**Videos:** `rollout_*.mp4` replays a training demo at the policy's input resolution; `attention_and_rollout_*.mp4` shows the current frame (top) and the prompt demo's frames (bottom), each tinted by how much the policy attends to it (pale green = low, darker blue = high). In a well-trained policy, the high-attention band moves through the prompt as the task progresses.
