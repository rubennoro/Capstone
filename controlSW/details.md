# controlSW details: running the policy and the Franka from the Pi 5

Everything runs on the **Raspberry Pi 5 mounted on the gripper**. There's no other computer in the loop. The Pi:

1. collects the observations (the same data the iPhone gives, minus SLAM: camera images + gripper state),
2. runs the trained policy (e.g. `block_raise` [run_001](../results/block_raise/run_001_2026-10-05/)), and
3. drives a **Franka arm** with the resulting actions.

Speed figures below are back-of-envelope estimates, not measurements. Measuring them is part of the build order.

## 1. Is it doable as is?

**Yes, but slowly.** The three risks that decide whether it actually works on the robot are listed below.

### No SLAM is fine

The policy's pose inputs (`gripper_right_eef_pos` / `gripper_right_eef_rot_axis_angle`) are **relative to the latest
observation**, so at deployment they come straight from the Franka's forward kinematics (libfranka's `O_T_EE`). That's how
UMI-style policies are normally deployed: SLAM only provides poses in the training data. If the gripper cameras give the
same images as the phone (same lenses, same mounting), the observations match training.

### run_001 on the Pi CPU, unmodified

- **Software:** PyTorch has official aarch64 wheels. Most other packages are pure Python. `robotmq` (pinned `0.1.14`) may
  need building from source.
- **Memory:** ~410 M parameters, about 1.6 GB of fp32 weights. An 8 GB or 16 GB Pi works.
- **Compute per `policy_inference` call:**
  - Image encoders: two CLIP ViT-B/16s (~172 M, one per camera) encode 4 images (2 cameras × 2 frames), about 70 GFLOPs.
  - Diffusion U-Net: ~239 M parameters, run 16 times (DDIM steps), attending to 512 prompt tokens each time, about
    50–80 GFLOPs.
  - The prompt demo is encoded once when the prompt is set (`policy.prompt()`), not on every call.
- **Speed:** the Pi 5's 4 Cortex-A76 cores reach ~15–40 GFLOP/s in real PyTorch fp32, so **~3–10 s per call**. That's
  with all 4 cores; with only 2 left for inference (see §2) it's slower.
- **Behavior:** run synchronously: observe, infer while the arm holds still, then execute 12 actions
  (`exec_action_horizon: 12`, about 0.5–1 s of motion), and repeat. For a slow, quasi-static task like `block_raise`,
  stop-and-go should still complete the task. One risk: the training demos are continuous motion, so pausing is mildly
  out-of-distribution.

### The hard part is controlling the Franka, not the model

- The Franka's control interface (libfranka / FCI) needs a **1 kHz real-time loop** over Ethernet. If a cycle is late, the
  arm stops with a `communication_constraints_violation` reflex.
- That loop is normally run on a dedicated x86 PC with a real-time kernel. Running it on a Pi that's also saturating its
  CPU with PyTorch is **the biggest risk in the whole setup**.
- What it needs: a PREEMPT_RT kernel, one isolated core for the control loop, and inference limited to the remaining
  cores. Memory-bandwidth contention from inference can still cause jitter, so prove this setup first, separately from
  the model.

## 2. System layout on the Pi

```
            ┌──────────────────────── Raspberry Pi 5 (RT kernel) ───────────────────────┐
cameras ──► │ core 0: capture        core 1-2: inference             core 3 (isolated): │
gripper ──► │  frames + width   ──►   detached_inference.py   ──►     Franka controller  │ ──Ethernet──► Franka (FCI, 1 kHz)
            │  + Franka EE pose      (RMQ on localhost)              (libfranka, C++)   │ ◄── EE pose / joint state
            │        ▲           policy actions (16 rel. poses)          │               │
            │        └──────────── EE pose from forward kinematics ◄─────┘               │
            └───────────────────────────────────────────────────────────────────────────┘
```

| Core | Process | Job |
|---|---|---|
| 0 | Capture + control (`real-env` task + `PolicyAgent`) | Reads cameras, gripper width and Franka EE pose; builds the observation dict; converts actions to pose targets; drives the gripper |
| 1–2 | Inference (`detached_inference.py --device cpu`) | Latest observation in → next action chunk out, over RMQ on `tcp://127.0.0.1:18765` |
| 3 (isolated) | Franka controller (libfranka, C++) | 1 kHz Cartesian impedance loop tracking the current pose target |

## 3. Making the Franka work

### 3.1 Low-level arm control: a libfranka loop on its own core

- Use a C++ **Cartesian impedance controller** that smoothly tracks a target pose. The policy only updates the target
  every few hundred ms; the loop interpolates in between.
- Don't write it from scratch. **franky** (pip `franky-control`) and **panda-py** wrap libfranka, run the 1 kHz loop in
  C++, and expose Python calls like "move to this pose" and "follow these waypoints". Both need building from source for
  ARM64 against the robot's libfranka version (which depends on Panda vs FR3).
- Pi setup:
  - a **PREEMPT_RT** kernel
  - `isolcpus=3` (core 3 runs only the control loop)
  - a **direct Ethernet cable** to the Franka control box, not through a switch
- libfranka on a Pi isn't the standard setup, so this is the first thing to test.

### 3.2 Plug it into `real-env`: most of the code already exists

[`real-env`](../behavior_prompting/deps/real-env/) already supports ARX5 and UR5 through a common `BaseController`
interface (`real_env/controllers/base_controller.py`). There's no Franka support yet. For the Franka:

- **`real_env/controllers/franka_cartesian.py`** (`FrankaCartesian(BaseController)`): implements `connect`, `reset`,
  `get_eef_pose_and_joint_pos` (from libfranka state) and `schedule_eef_traj` (passes waypoints to franky).
  `ur5_cartesian.py` (~380 lines) is the closest template.
- **`real_env/tasks/umi_franka_task.py`** + **`real_env/configs/tasks/umi_franka.yaml`**: a copy of the UR5/ARX5 task with
  the Franka swapped in, a home pose, and our gripper.
- **`real_env/agents/policy_agent.py`** stays as is. It already turns the policy's relative actions into absolute pose
  targets using the current EE pose.

### 3.3 Coordinate frames: most first deployments break here

- The policy outputs motions relative to the **gripper/camera frame used in training** (where the phone sat). Set the
  Franka's end-effector frame (`F_T_EE` / TCP) to exactly that point and orientation on our gripper, from CAD.
- If it's off, every action gets rotated or offset. The ARX5 controller's `z_forward_to_x_forward_tranform` helpers handle
  this kind of axis convention.
- Test by replaying a recorded demo open-loop with `detached_inference_action_replay.py` before running the model. If the
  replay moves sensibly, the frames are right.

### 3.4 Observations

| Observation | Source on the robot |
|---|---|
| `gripper_right_eef_pos`, `gripper_right_eef_rot_axis_angle` | Franka forward kinematics at our TCP (`O_T_EE`), converted to relative pos3 + rot6d |
| `gripper_right_gripper_width` | Our gripper's encoder / actuator |
| `camera_right_main`, `camera_right_ultrawide` | Gripper cameras (must match the phone's view and lenses) |

The capture process timestamps these together. `block_raise` is single-arm and stored as `right` (`sides=single`), so
only the `right` keys apply.

### 3.5 Gripper actuation

The policy also outputs `action_right_gripper_width`.
- **Our own actuator:** its driver runs in the capture/control process.
- **Franka Hand:** controlled through libfranka's (non-real-time) gripper API.

## 4. Optimizations

Speedups are rough estimates; measure each one.

| Optimization | Retrain? | Rough gain | Notes |
|---|---|---|---|
| DDIM steps 16 → 8 or 4 | No | U-Net 2–4× | One config value; check action quality |
| int8 dynamic quantization of Linear layers (`torch.ao`, QNNPACK) | No | ~2× on the ViTs | ViTs are mostly Linear layers, so they quantize well on ARM |
| ONNX Runtime (XNNPACK) or ExecuTorch export | No | 1.5–3× | Faster ARM kernels than eager PyTorch; export takes work |
| Thread pinning / `torch.set_num_threads(2)` | No | Avoids slowdowns | Required anyway to protect the Franka loop |
| Smaller image encoder (ResNet-18 / ViT-S, or one shared encoder) | Yes | 4–8× on vision | Biggest single lever |
| Lower image resolution | Yes | ~2–4× on vision | |
| Smaller U-Net | Yes | 2–4× | |
| One-step consistency / distilled policy | Yes | ~16× on the U-Net | More research effort |

- **Without retraining**, stacking these could reach about 1–3 s per call: usable stop-and-go.
- **With a deployment-sized retrained model**, sub-second inference on 2 cores is plausible, allowing near-continuous
  motion. Train it with the same `scripts/run_pipeline.sh` on the same data and compare val MSE and rollout videos
  against run_001.

## 5. Build order

Each step is testable on its own. Steps 1 and 4 can run in parallel.

1. **Control on the Pi (go / no-go).** RT kernel + franky moves the Franka smoothly to poses. Then keep it stable while a
   CPU stress test runs on cores 0–2. If this fails, no model optimization saves the setup.
2. **`FrankaCartesian` + `umi_franka_task`** working with spacemouse teleop (`agents/spacemouse_agent.py` already exists).
3. **Calibrate the TCP frame**, then replay a recorded demo open-loop with `detached_inference_action_replay.py`.
4. **Time run_001** on 2 Pi cores with dummy observations, then apply the no-retraining optimizations and re-time.
5. **Closed loop:** policy on the Pi, stop-and-go first.
6. **If too slow:** train a smaller deployment-sized model and repeat steps 4–5.

## Open questions

- **Franka model:** Panda or FR3? It determines the libfranka version.
- **Gripper:** our own actuator, or a Franka Hand?
