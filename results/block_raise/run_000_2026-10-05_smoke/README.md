# block_raise: run_000_2026-10-05_smoke

**Status:** completed  ·  **Mode:** smoke  ·  **Job:** 10850802  ·  **Packaged:** (migrated from `first_smoketest/`) 2026-10-05T23:54:37

> First 3-epoch smoke test on block_raise; checks the pipeline end to end and measures speed.

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
| Epochs | 3 completed of 3 configured |
| Batch size | 70 |
| Learning rate | 5e-05 (cosine, warmup 0.05) |
| Precision | bf16 |
| Evaluate / rollout every | 1 / 1 epochs |
| Hardware | NVIDIA H200 on `d4055` (gpu-short) |
| Command-line overrides | `task.dataset.dataset_path=/home/noroian.r/CapstoneProject/Capstone/iPhUMI/python_package/iphumi/demonstration_processing/tmp_sessions/firstreal-single/replay_buffer_firstreal-single.zarr.zip`<br>`exp_name=smoke_replay_buffer_firstreal-single`<br>`group_tag=smoke_tests`<br>`logging.mode=offline`<br>`training.num_epochs=3`<br>`training.val_every=1`<br>`training.rollout_every=1`<br>`training.sample_every=1`<br>`checkpoint.save_train_loss_ckpt_every=1`<br>`task.dataset.cache_dir=/tmp/noroian.r/zarr_cache` |

## Results

| Epoch | Train loss | Val loss | Val action MSE |
|---|---|---|---|
| 0 | 0.2134 | 0.0501 | 0.0389 |
| 1 | 0.0400 | 0.0305 | 0.0295 |
| 2 | 0.0286 | 0.0270 | 0.0192 |

- **Speed:** 4.57 steps/s, 518 steps/epoch ≈ 1.9 min/epoch
- **Training time:** 10.5 min
- **Peak GPU memory:** 34.1 GB
- **Best checkpoint:** epoch 2, val_action_mse_error_all = 0.01920: `checkpoints/best_epoch=0002-val_action_mse_error_all=0.0191988.ckpt` (full size, 6.4 GB; only on the laptop, gitignored. The cluster copies were deleted.)

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
| `videos/` | Rollout + attention videos per evaluation: `final` |
| `run_config/` | Hydra config + overrides, normalizer, job_info.json |
| `logs/` | Training log, Slurm console log (progress bars stripped), GPU memory samples |
| `checkpoints/` | Best checkpoint only, slimmed (no optimizer state); gitignored, lives on the cluster |

**Videos:** `rollout_*.mp4` replays a training demo at the policy's input resolution; `attention_and_rollout_*.mp4` shows the current frame (top) and the prompt demo's frames (bottom), each tinted by how much the policy attends to it (pale green = low, darker blue = high). In a well-trained policy, the high-attention band moves through the prompt as the task progresses.
