# First smoke test: behavior prompting on `block_raise`

A short 3-epoch training run of the behavior prompting policy (`umi_policy_dunetp`) on our own
iPhUMI data, run on the Explorer cluster on 2026-10-05. Its purpose was to check that training works
end to end and to measure speed before the full 140-epoch run. The policy itself is barely trained.

- **Dataset:** `firstreal-single` (45 demos of `block_raise`, lifting an eraser with one gripper;
  37,971 frames at 60 Hz). 43 demos for training and 2 held out for validation (`val_ratio: 0.05`).
- **Hardware:** 1× NVIDIA H200 (job `10850802`, `gpu-short` partition), 16 CPUs.
- **Settings:** the defaults from `umi_policy_dunetp` (batch size 70, lr 5e-5 with cosine schedule, bf16), plus
  3 epochs with validation, rollout and checkpointing every epoch. The exact config is in `run_config/`.

## Results

| | Epoch 0 | Epoch 1 | Epoch 2 |
|---|---|---|---|
| Train loss (epoch average) | 0.213 | 0.040 | **0.029** |
| Val loss | 0.050 | 0.031 | **0.027** |
| Val action MSE (all) | 0.0389 | 0.0295 | **0.0192** |

- All the losses and action errors fall every epoch, and validation tracks training, so there's no sign of overfitting yet.
  The epoch-0 train average is much higher than val only because it averages over the first steps, when the loss
  starts around 1.0. Val is measured at the end of the epoch.
- **Speed:** 4.66 steps/s and 518 steps per epoch, so **about 2.1 minutes per epoch of training** and about 3 minutes
  including the validation, rollout and checkpoints that ran every epoch here.
  **Full-run estimate:** 140 epochs with evaluation every 10 comes to about 5.2 h, which fits one 8-hour job on one H200.
- **GPU memory:** peak **34.1 GB** of 141 GB.
- **Job time:** 13.7 min in total, with about 2.5 min of startup (imports, copying the dataset to node-local LMDB, normalizer).

## What's in this folder

| Path | Contents |
|---|---|
| `plots/loss_per_step.png` | Train loss at every gradient step (log scale); dotted lines mark epoch boundaries |
| `plots/loss_per_epoch.png` | Epoch-average train loss vs val loss |
| `plots/action_mse_per_epoch.png` | Action error split into position / rotation / gripper width, train vs val |
| `plots/lr_and_grad_norm.png` | Learning-rate schedule (warmup + cosine to 0 over 3 epochs) and gradient norm |
| `plots/gpu_memory.png` | GPU memory over the job |
| `history.csv` | Every logged step: `train_loss`, `lr`, `grad_norm`, ... |
| `epochs.csv` | One row per epoch with all train/val loss and action-MSE metrics |
| `videos/rollout_block_raise.mp4` | One of our recorded demos replayed at the policy's input resolution (224×224, 20 fps, 16.6 s). The policy is queried every 12 frames, but its predictions aren't drawn here. This is the footage for the top panel of the attention video |
| `videos/attention_and_rollout_umi_block_raise.mp4` | Top: the current frame of the replayed demo. Bottom grid: 14 frames in order from a *different* `block_raise` demo, which is the **prompt**. Each tile's background tint (pale green = low, darker blue = high) shows how much the policy attends to that moment of the prompt |
| `run_config/config.yaml`, `overrides.yaml` | The exact Hydra config the run used |
| `run_config/normalizer.pkl` | Data normalization statistics (needed to run the policy) |
| `logs/smoke_train_log.txt` | The training console output, without the progress-bar lines |
| `logs/gpu_memory.csv` | Raw `nvidia-smi` samples, every 30 s |
| `checkpoints/` | **Not in git** (`*.ckpt` is ignored). Local copy of the best checkpoint, `epoch=0002-val_action_mse_error_all=0.0191988.ckpt` (6.4 GB) |

**The videos show the final epoch only.** The runner overwrites `vis/` every epoch, and offline wandb only copies
the file when the run ends, so the three per-epoch attention videos it saved are identical. With such a
lightly trained policy, the attention over the prompt frames is close to uniform (all tiles the same pale green).
In a well-trained policy, a band of higher attention should move through the prompt as the task progresses
(approach frames while approaching, lift frames while lifting). That's the thing to look for in the full run.

## Regenerating the plots

```bash
# needs wandb==0.27.2 (the version that wrote the run), pandas and matplotlib; the cluster's `bp` env has them
python scripts/wandb_offline_to_csv.py <train_network/wandb/offline-run-...> -o first_smoketest/plots \
  --gpu-csv <logs/smoke_gpu.csv>
```
The source run on the cluster is
`train_network/runs/2026.10.05/22.47.06_smoke_replay_buffer_firstreal-single_policy_umi_prompt_prompt_diffusion_unet`
with wandb run `train_network/wandb/offline-run-20261005_224755-kh6290z5`.
