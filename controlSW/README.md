# controlSW: on-gripper control software

**Goal:** take the policy deployment code in
[`behavior_prompting/behavior_prompting/deployment/`](../behavior_prompting/behavior_prompting/deployment/) and get it
running locally on the **Raspberry Pi 5 mounted on the gripper**. The Pi has to do both jobs at once:

1. **collect the observations** the policy needs (camera images + gripper state), and
2. **run the trained policy** on them (e.g. `block_raise` [run_001](../results/block_raise/run_001_2026-10-05/)) and output
   actions.

There's no off-board GPU in the loop: inference happens on the Pi. This folder will hold the Pi-specific code, setup
scripts and configs. **Full plan (feasibility, Franka control, optimizations, build order): [details.md](details.md).**

## What's in `deployment/` today

Both scripts are **RobotMQ (RMQ) servers**. The robot-side controller connects to them as a client
(`PolicyAgent` in [`deps/real-env/real_env/agents/policy_agent.py`](../behavior_prompting/deps/real-env/real_env/agents/policy_agent.py),
driven by a task such as `real_env/tasks/iphumi_arx5_bimanual_task.py`). Default endpoint: `tcp://0.0.0.0:18765`.

| Script | What it does | Needs |
|---|---|---|
| `detached_inference.py` | Loads a trained `.ckpt`, takes observations (camera images + gripper poses/widths), runs the diffusion policy, returns a chunk of actions. Topics: `policy_config`, `policy_reset`, `policy_prompt`, `policy_inference`, `export_recorded_data`. | PyTorch, the full `behavior_prompting` package, a checkpoint, the dataset (for prompt demos), a CLIP tokenizer |
| `detached_inference_action_replay.py` | No model: replays a recorded demo's actions open-loop from a `.zarr.zip` replay buffer. Same topics plus `query_initial_gripper_transform`. | numpy, scipy, zarr, robotmq |

Example (as written today, on a CUDA machine):

```bash
python detached_inference.py -i <best.ckpt> --dataset-path <dataset>.zarr.zip --device cuda
python detached_inference_action_replay.py -r <dataset>.zarr.zip -t block_raise -n 0
```

## Target layout on the Pi

Two processes on the same Pi, talking over RMQ on localhost (`tcp://127.0.0.1:18765`), so the existing server/client
protocol is reused unchanged:

| Process | Cores | Job |
|---|---|---|
| **Capture + control** | 1 | Reads cameras and gripper state at a fixed rate into a short ring buffer; builds the observation dict (`gripper_right_*`, `camera_right_*`); executes the current action chunk |
| **Inference** (`detached_inference.py --device cpu`) | 3 | Receives the latest observation, returns the next action chunk |

**Asynchronous execution:** the gripper keeps executing the current chunk while the next one is computed. The model was
trained to execute 12 of every 16 predicted actions (`exec_action_horizon: 12`), roughly 0.5–1 s of motion. That gives
the **latency target**: one `policy_inference` call has to finish within one executed chunk (≲ 0.5–1 s) on 3 Pi cores.

## The main problem: the model is too big for the Pi as is

Rough estimate, not yet measured:
- The run_001 policy has ~410 M parameters: two CLIP ViT-B/16 image encoders (~172 M), one per camera, plus a diffusion
  U-Net (~239 M).
- Each call encodes 4 images (2 cameras × 2 frames), about 70 GFLOPs, then runs 16 DDIM denoising steps, about 50–80 GFLOPs.
- The prompt demo is encoded once when the prompt is set (`policy.prompt()`), not on every call.
- At a realistic 15–40 GFLOP/s for PyTorch on the Pi 5's Cortex-A76 cores, that's **~3–10 s per call, about 5–10× over
  the target**.
- Memory is fine: ~1.6 GB of fp32 weights.

Ways to close the gap, cheapest first:

1. **Measure the real latency on the Pi.** Everything below depends on that number.
2. **No retraining:**
   - fewer DDIM steps (16 → 4–8; check action quality)
   - fp16 / int8 quantization
   - an ARM-optimized runtime (ONNX Runtime, ExecuTorch)
   - pin inference threads to 3 cores so it doesn't fight the capture process
3. **Retrain a deployment-sized policy** (the big lever) with the same pipeline (`scripts/run_pipeline.sh`), then compare it
   against run_001:
   - a smaller image encoder (e.g. ResNet-18 / ViT-S, or one shared encoder)
   - a lower image resolution
   - a smaller U-Net
   - a one-step distilled / consistency policy instead of 16 diffusion steps
4. **Hardware assist:** a Raspberry Pi AI HAT+ (Hailo) running the image encoder while the CPU runs the U-Net.

## Other things to sort out

- **Dependencies.** Install PyTorch (CPU, aarch64 wheels), `robotmq` (`real-env/env.yaml` pins `0.1.14`), `transformers`,
  `diffusers`, `hydra-core`, `dill`, `zarr`, `imageio` and the `behavior_prompting` package itself on Raspberry Pi OS (64-bit).
  Some of these may not ship aarch64 wheels and will need building.
- **Checkpoint.** Start from the slimmed checkpoint (2.5 GB, no optimizer state), then convert to an inference-only format.
- **Single gripper.** The deployment docstrings describe the bimanual (left + right) setup. `block_raise` uses one gripper,
  stored as `right` (`sides=single`), so only the `gripper_right_*` / `action_right_*` keys apply. Check that the server and
  the robot-side task handle the single-arm `shape_meta`.
- **Camera source (open question).** The policy was trained on the gripper-mounted iPhone's `main` and `ultrawide` cameras.
  - If the Pi reads those iPhone streams, the observations match training.
  - If the Pi uses its own camera, the policy has to be trained on data from that camera.
- **Gripper state.** The observations include end-effector pose and gripper width. Decide where the Pi gets them (iPhone
  tracking, encoders, etc.).

## Suggested order

1. Set up the Pi (64-bit OS, Python env) and get `detached_inference_action_replay.py` serving on localhost. It has no
   model or PyTorch, so it proves `robotmq`, `zarr` and the client loop work on the Pi.
2. Install CPU PyTorch + `behavior_prompting`, run `detached_inference.py --device cpu` with the run_001 checkpoint, and time
   one inference call with dummy observations, alone and with a capture process running.
3. Apply the no-retraining optimizations and re-time.
4. If it's still over the target, train a deployment-sized policy and repeat.
5. Build the capture process with the real camera / gripper inputs and test end to end on the gripper.
