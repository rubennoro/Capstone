# block_raise: run_002_2026-10-06_smoke

**Status:** completed  ·  **Mode:** smoke  ·  **Job:** 10878177  ·  **Packaged:** 2026-10-06T17:52:37

> pipeline test

## Data

| | |
|---|---|
| Dataset | `firstreal-single` |
| Path | `/home/noroian.r/CapstoneProject/datasets/block_raise/firstreal-single/firstreal-single.zarr.zip` |
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
| Hardware | NVIDIA H200 on `d4054` (gpu-short) |
| Command-line overrides | `task.dataset.dataset_path=/home/noroian.r/CapstoneProject/datasets/block_raise/firstreal-single/firstreal-single.zarr.zip`<br>`task.dataset.cache_dir=/tmp/noroian.r/zarr_cache`<br>`exp_name=run_002_2026-10-06_smoke`<br>`group_tag=block_raise`<br>`logging.mode=offline`<br>`+logging.dir=/scratch/noroian.r/bp_runs/block_raise/run_002_2026-10-06_smoke`<br>`checkpoint.topk.k=1`<br>`checkpoint.save_train_loss_ckpt_every=null`<br>`training.num_epochs=3`<br>`training.val_every=1`<br>`training.rollout_every=1`<br>`training.sample_every=1` |

## Results

| Epoch | Train loss | Val loss | Val action MSE |
|---|---|---|---|
| 0 | 0.2134 | 0.0501 | 0.0389 |
| 1 | 0.0400 | 0.0305 | 0.0295 |
| 2 | 0.0286 | 0.0270 | 0.0192 |

- **Speed:** 4.61 steps/s, 518 steps/epoch ≈ 1.9 min/epoch
- **Training time:** 8.8 min
- **Peak GPU memory:** 34.1 GB
- **Best checkpoint:** epoch 2, val_action_mse_error_all = 0.01920: `checkpoints/best_epoch=0002-val_action_mse_error_all=0.0191988.ckpt` (slimmed, not in git)

Fetch the checkpoint to the laptop (from the repo root):
```bash
rsync -a noroian.r@xfer.discovery.neu.edu:CapstoneProject/Capstone/results/block_raise/run_002_2026-10-06_smoke/checkpoints/ results/block_raise/run_002_2026-10-06_smoke/checkpoints/
```

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
| `videos/` | Rollout + attention videos per evaluation: `epoch_0000`, `epoch_0001`, `epoch_0002`, `final` |
| `run_config/` | Hydra config + overrides, normalizer, job_info.json |
| `logs/` | Training log, Slurm console log (progress bars stripped), GPU memory samples |
| `checkpoints/` | Best checkpoint only, slimmed (no optimizer state); gitignored, lives on the cluster |

**Videos:** `rollout_*.mp4` replays a training demo at the policy's input resolution; `attention_and_rollout_*.mp4` shows the current frame (top) and the prompt demo's frames (bottom), each tinted by how much the policy attends to it (pale green = low, darker blue = high). In a well-trained policy, the high-attention band moves through the prompt as the task progresses.
