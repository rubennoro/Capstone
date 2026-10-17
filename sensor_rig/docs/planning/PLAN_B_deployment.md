# PLAN B: from the trained policy to the Franka

Scope: turn live sensor data + robot state into exactly the observation run_001 takes, run inference, and turn the
action chunk into Franka arm + gripper commands. Planner output only. No repo files were changed.

Path abbreviations:
- `RE/` = `behavior_prompting/deps/real-env/real_env/`
- `BP/` = `behavior_prompting/behavior_prompting/`
- `RUN` = `results/block_raise/run_001_2026-10-05/run_config/config.yaml`
- `PKG/` = third-party sources unpacked in `<scratchpad>/pkgs/ex/` (robotmq 0.1.14, robologger 0.1.7, robot_utils 0.1.4)

All line numbers below were re-checked against the working tree on 2026-10-16. Where this plan disagrees with R3 or
`controlSW/details.md`, it says so.

---

## 0. Summary

**Observation contract (run_001), per inference call:**

| model key | shape sent to model | source at deployment | time samples |
|---|---|---|---|
| `camera_right_main_rgb` | (2,3,224,224) f32 in [0,1] | RigCamera `main`, 960x720 virtual iPhone main → RGB → pad to 960² → INTER_AREA 224 | newest packet `T` and `T−50 ms` |
| `camera_right_ultrawide_rgb` | (2,3,224,224) | RigCamera `ultrawide`, 960x720 virtual iPhone UW, same transform | `T` and `T−100 ms` |
| `gripper_right_eef_pos` | (2,3) | **commanded** TCP targets (action history), relative to the newest one | 50 ms apart; last row = 0 |
| `gripper_right_eef_rot_axis_angle` | (2,6) rot6d | same, rotation part | last row = `[1,0,0,0,1,0]` |
| `gripper_right_gripper_width` | (2,1) metres, absolute | **commanded** width history | 50 ms apart |
| `gripper_right_eef_rot_axis_angle_wrt_start` | (2,6) | FK relative to the episode start | ignored by the policy |
| prompt | one demo of task `block_raise` from `firstreal-single.zarr.zip` | `policy_prompt` request, once per episode | n/a |

**Action contract:** a (16,10) chunk = pos3 + rot6d (relative to the TCP pose at observation time, in the TCP frame) +
absolute width (m). Spacing 50 ms (20 Hz), so the chunk spans 0.75 s. `PolicyAgent` anchors it on measured FK →
absolute base-frame TCP targets stamped `T + k·50 ms` → the task sends them to a `CARTESIAN_JOINT` controller →
interpolator → Franka Cartesian impedance at 1 kHz. The gripper width goes to the same controller as its 1-DoF "joint".

**Hardware (fixed by the lead):** one Jetson Orin Nano (JetPack 6, 8 GB shared memory) is the central computer. It
has 2 IMX219 CSI cameras (main ~90° M12, ultrawide ~160° fisheye) and an ICM-42688-P IMU (not used at deployment).
There is no Pi.

**Recommended stack:**
- **On the Jetson:** RigCamera (`backend: argus`), `detached_inference.py --device cuda`, the task + PolicyAgent, and
  `FrankaCartesian` (a `BaseController`, 200 Hz interpolation + gripper driver). All in one `time.monotonic()`
  domain.
- `FrankaCartesian` streams TCP targets to a thin **franka_bridge** process that owns the 1 kHz libfranka loop
  (panda-py `CartesianImpedance`; franky as the alternative). The bridge is a separate process behind a ZMQ endpoint,
  so it can run **either on the Jetson itself** (PREEMPT_RT kernel, isolated core, direct Ethernet) **or on a small RT
  PC** with only a config change.
- **Recommendation on the open question (§2.6):** try the bridge on the Jetson first, behind a 1–2 day go/no-go
  stress test (B3-hw step 1). It meets the single-computer goal, and if it fails the arm stops (a reflex), it doesn't
  move unsafely. Fall back to a used x86 mini-PC with PREEMPT_RT (~$150–250) if the test fails. Don't build anything
  that only works in one of the two layouts.

**Critical path:** decisions + procurement (B0: gripper actuator, Panda/FR3 → libfranka version, JetPack RT kernel)
→ Jetson RT go/no-go + bridge on the real arm (B3-hw) → TCP calibration + open-loop replay (B4) → closed loop (B5).

**Buildable now, without hardware:** B1 (laptop harness: mock controller + mock spacemouse + RigCamera mock + real
inference server + real PolicyAgent), B2 (upstream fixes), B3-sw (FrankaCartesian against a fake bridge, task, configs),
and the B6 method on the laptop GPU (the real numbers need the Jetson, no robot required).

---

## 1. Observation path, precisely

### 1.1 What the policy consumes (RUN, verified)

- Horizons: `low_dim_obs_horizon: 2`, `img_obs_horizon: 2`, `action_horizon: 16`.
- Spacing: `obs_down_sample_steps: 3` at 60 Hz = 50 ms; `ultrawide_down_sample_steps: 1` counted at the native 10 Hz
  ultrawide rate = 100 ms (`RUN:73-79`).
- Keys and shapes: see the table in §0 (`RUN:90-176`).
- Rotation: `rotation_rep: rotation_6d` with a raw axis-angle (3) in the zarr.
- Pose representation: `pose_repr: {obs_pose_repr: relative, action_pose_repr: relative}` (`RUN:177-179`).
  - Training: `rel = inv(T_last) @ T_t` for both obs and action (`BP/train_network/dataset/umi_task_dataset.py:427-436`,
    `BP/train_network/common/pose_repr_util.py:62-64`).
  - The action slice starts at the current index (`BP/train_network/common/sampler.py:714-715`), so action[0] is the
    pose at observation time t (≈ identity) and action[k] is the pose at t + k·50 ms.
- Images inside the model: eval `CenterCrop ratio 0.95` (`RUN:181-186`), then CLIP ViT-B/16 with
  `use_vision_norm: true`, one per camera (`share_rgb_model: false`, `RUN:268-284`).
- Prompting: `use_prompting: true`, `prompt_sample_mode: pair`, `prompt_sequence_length: 512`,
  `max_prompt_full_demos: 1`, `prompt_chunk_n_actions: 20`, `prompt_relative_to: last` (`RUN:84-89`, `:232`).
- Encoder: `PairPromptObsEncoder` with `prompt_with_obs_decoder_enabled: true` (`RUN:261-310`). `prompt()` encodes the
  prompt demo once and caches the tokens (`BP/train_network/model/prompt/prompt_obs_encoder.py:842-850`).
- Other: `num_inference_steps: 16` (DDIM, `RUN:249`); sim runner `exec_action_horizon: 12` (`RUN:219`). No
  `task_language` key, so `task_name` in the obs is never tokenized (`BP/deployment/detached_inference.py:121-133`).
- Training ranges (from `run_config/normalizer.pkl`, read with the scratch env):
  - gripper width min 0.0447, max 0.0844, mean 0.0673 m. The minimum is the eraser grasp, not "closed".
  - action pos over a chunk within [−0.22, 0.25] m on x/y, [−0.45, 0.26] m on z.
  - per-50 ms proprio pos within ±0.033 m, i.e. peaks around 0.6 m/s (handheld demos plus SLAM noise).

### 1.2 The chain, stage by stage

1. **RigCamera** (`RE/peripherals/rig_camera.py`, logic in `sensor_rig/sensor_rig/deploy/frame_source.py`) returns
   `{"main","ultrawide"}` as (720,960,3) uint8 BGR: virtual iPhone 16 Pro pinhole cameras, main f=703.17, UW f=388.52.
   It also returns one timestamp: the main frame's true CLOCK_MONOTONIC capture time (rig.yaml offsets applied, no
   latency subtracted).
   - `camera_latency_s: 0.0` in `rig_wrist_right.yaml`, so `base_camera.py:221` is a no-op.
   - `publish_mode: auto` gives 60 Hz packets (fixed_rate if the physical main runs at 30 fps).
2. **BaseCamera.run** asserts shapes, converts BGR→RGB (`RE/peripherals/base_camera.py:225-228`), and puts each frame
   in a shared-memory topic plus one `timestamp` (`:235-240`). **Shared memory means the task must run on the same
   host as the camera.**
3. **Task `get_observations`**: copy `iPhumiARX5Task.get_observations` (`RE/tasks/iphumi_arx5_task.py:24-82`) into
   the Franka task.
   - `get_downsampled_camera_images(client, image_history_len=2, agent_update_freq_hz, cam_obs_freq_hz)`
     (`RE/utils/umi_utils.py:9-76`):
     - `cam_obs_freq_hz` = main: f, ultrawide: f·(10/1)/(60/3) = f/2 (`RE/agents/policy_agent.py:82-101`). At
       f = 20 Hz: main frames 50 ms apart, ultrawide 100 ms apart, both matching training.
     - Anchor = the newest packet timestamp `T` (`umi_utils.py:55`). Nearest frame per target. Raises if more than
       2/fps = 33 ms off (`:64-72`).
   - Robot state: `last_ts = time.monotonic()` (because `match_camera_latency: false`, `iphumi_arx5_task.py:37-44`).
     FK pose and width are read at `[last_ts − 1/f, last_ts]` via `BaseCartesianJointClient.get_eef_xyz_wxyz(t)` /
     `get_joint_pos(t)`, which interpolate the controller's published state (`RE/controllers/base_controller_client.py:338-383`, `:461-506`).
   - `crop_image` and the gripper mask must **not** apply. `iPhumiARX5Task.get_observations` ignores both. The
     `umi_*` base versions crop and mask (`RE/tasks/umi_arx5_task.py:245-261`), which would be wrong.
   - Output: `robot0_eef_xyz_wxyz (2,7)`, `robot0_gripper_width (2,1)`, `robot0_main_camera (2,720,960,3)` RGB,
     `robot0_ultrawide_camera (2,720,960,3)`, `timestamps (2,)` (main-frame times), `task_name`,
     `robot0_eef_xyz_wxyz_wrt_start`.
4. **`PolicyAgent.predict_actions`** (`RE/agents/policy_agent.py:103-253`):
   - Saves `obs_last_eef` = measured FK at `last_ts` (`:135-138`).
   - `action_as_proprio: true` replaces the pose and width with `history_actions` (`:141-145`). The history comes
     from `get_history_actions(length=2, action_type="policy")` (`RE/tasks/umi_arx5_task.py:339-377`):
     - **continuous mode:** entries `exec_h−1, exec_h` (7, 8) of the previous absolute chunk (`:314-317`);
     - **pause mode:** entries 6, 7, because the chunk was truncated to `exec_h` at `:292-297` before the history was
       stored;
     - **first call after spacemouse:** the single spacemouse command, repeated, so the relative motion is identity
       (`:365-374`).
   - Relative pose: `get_relative_pose(pose_t, poses[-1])`, then `xyz_wxyz_to_pose_9d` → pos3 + rot6d (`:157-166`).
     - `get_relative_pose` computes `inv(ref)·new` (`PKG/robot_utils/pose_utils.py:130-150`), the same as training.
     - The rot6d convention matches (R3 §2.3: `robot_utils/pose_utils.py:303-306` vs `BP/common/pose_util.py:95-98`).
   - `keep_one_step_dummy_proprio: true`, so nothing is trimmed (`:151-155`).
   - Sends `episode_idx=0`, the camera keys and `task_name` unchanged (`:199-204`), via
     `request_with_data("policy_inference", timeout_s=5.0)` (`:206-208`).
   - **Payload:** 4 × 960·720·3 = 8.3 MB per call (B6 item).
5. **Inference server** (`BP/deployment/detached_inference.py`):
   - Maps `robot0_*` to `gripper_right_*` / `camera_right_*` from the sorted `_eef_pos` prefixes. Single arm, so
     robot0 = right (`:134-162`).
   - `_process_obs` (`:418-457`): if HxW ≠ 224, `get_image_transform_with_border(in_res=(960,720), out_res=(224,224))`.
     That pads to a 960x960 square (120-px black bars top and bottom), resizes with `cv2.INTER_AREA`, then
     `moveaxis → (T,3,224,224)` float32 /255. This is identical to training (R3 §3; the two `cv_util.py` copies are
     the same).
   - Low-dim values pass through as float32. Each value gets `unsqueeze(0)` for the batch (`:331-337`).
   - `policy.predict_action` (`BP/train_network/policy/diffusion_unet_policy.py:124-171`): normalize, obs encoder
     (CenterCrop 0.95, CLIP), cross-attend to the cached prompt tokens, 16 DDIM steps, unnormalize → (16,10).
   - **No prompt means an error:** forward asserts `self.is_prompted` (`prompt_obs_encoder.py:765-769`). The server
     replies with an "ERROR" string and `PolicyAgent` raises `ValueError` (`policy_agent.py:211-212`).
6. **Prompt**:
   - `PolicyAgent.send_prompt_request(task_name, prompt_index)` (60 s timeout, `policy_agent.py:273-286`) → server
     `_handle_prompt` (`detached_inference.py:376-401`).
   - The server picks demo `prompt_index` of `task_name` from `--dataset-path` (the train split by default,
     `--prompt-from-dataset`), takes `sample["obs"]["prompt"]`, and calls `policy.prompt()` once.
   - For block_raise: `task_name = "block_raise"`. `process_dataset.sbatch:38-41` labels unlabeled demos with the task
     name, and the run_001 videos are `rollout_block_raise.mp4`. Confirm against `meta/task_names` when the zarr is on
     the laptop.
   - `prompt_index` ∈ [0, n_train) with n_train ≈ 42 (45 demos, `val_ratio: 0.05`).
   - **The prompt is cleared by every `policy_reset`.** `policy.reset()` → `PairPromptObsEncoder.reset()` sets
     `is_prompted=False` (`prompt_obs_encoder.py:852-856`).
     - `reset_policy_agent` runs on `R` and automatically when leaving policy mode
       (`RE/tasks/umi_arx5_task.py:383-390`).
     - So the task must re-prompt at every episode start, like the bimanual task's `_prompted` flag
       (`RE/tasks/iphumi_arx5_bimanual_task.py:395-410`, cleared at `:432`).
     - The `C` (single-step) key does **not** call `start_episode` (`RE/tasks/base_task.py:156-157`), so the prompt
       must also be sent lazily in `run_policy_control`.

### 1.3 Where each input comes from at deployment

| input | deployment source | notes |
|---|---|---|
| main / UW images | RigCamera on the Jetson (`backend: argus`, IMX219 main ~90° + fisheye ~160°, remapped to the virtual iPhone cameras) | geometry already verified offline (sensor_rig `test_deploy_equivalence.py`: deploy vs export MAE ≤ 0.2 at 224) |
| image timestamps | main capture time, CLOCK_MONOTONIC | measure the residual latency with a QR/LED timecode; put it in rig.yaml, not `camera_latency_s` |
| `robot0_eef_xyz_wxyz` (measured) | Franka FK at the **training TCP**: `O_T_EE` with EE = TCP (§2.6), published by `FrankaCartesian` as `eef_pose` | used **only** as the anchor for absolute targets (and wrt_start, which is ignored) |
| `robot0_gripper_width` (measured) | gripper actuator position mapped to training width units (§2.8) | only seeds the spacemouse/first-step history. The model sees commanded widths |
| proprio fed to the model | the task's own commanded action history | no sensor involved |
| prompt | zarr on the inference host | the 845 MB zarr has to be on the laptop/Jetson (scp from the cluster) |
| IMU / VIO | **not used** at deployment (R3 §5) | |

### 1.4 Timing rules

- `agent_update_freq_hz` **must be 20**. The default is 10 (`RE/configs/agents/policy_agent.yaml:5`). At 10, the
  robot runs at half speed and all image/proprio spacings are doubled.
- `fps % int(agent_hz) == 0` is asserted (`RE/tasks/umi_arx5_task.py:117-120`). The rig config says fps 60, so this
  holds.
- Frame-selection tolerance is 2/fps = 33 ms. UW pairing error is ≤ ½ UW period (≤ 17 ms at 30 fps).
- Action timestamps are `T + k/f` with `T` = newest main capture time (`policy_agent.py:247-251`). They are re-based
  to `now + k/f` on the first chunk after spacemouse, in single-step mode, and whenever
  `pause_when_policy_inference` is set (`umi_arx5_task.py:278-289`).
- Arm trajectories are sent with `+ lookahead` (0.1 s) (`:299-303`). The controller interpolates at
  `now + lookahead_time_s` (ARX5: 0.1, `RE/controllers/arx5_cartesian.py` `send_robot_target`). The two cancel, so
  the target sent at wall time `T + k/f` is action k.
- The gripper trajectory is sent **without** the lookahead (`umi_arx5_task.py:305-309`) but interpolated with it, so
  gripper commands **lead by 0.1 s**. That looks deliberate (it compensates gripper lag); keep it.
- The controller drops trajectory points earlier than `now + trim_new_trajectory_time_s` (0.09)
  (`RE/common/interpolator.py` `update`, `:29-46` / `:201-...`). In continuous mode this drops the actions already
  stale because of capture + inference latency.
- Continuous mode waits until `ts[exec_h−1] − 0.02` before observing again (`umi_arx5_task.py:329-335`). Pause mode
  waits until `ts[exec_h−1] + camera_latency_s` (`:319-328`); `camera_latency_s` there is a task parameter (0.08)
  unrelated to the camera config.
- **Continuous-mode feasibility.** Let L = capture latency + transfer + inference, f = 20 Hz, and lookahead λ = 0.1 s.
  - A chunk observed at T covers wall time up to T + 15/f + λ = T + 0.85 s.
  - The next chunk is observed no earlier than T + max(L, (exec_h−1)/f − 0.02) and arrives L after that.
  - No gap therefore needs max(L, 0.33) + L < 0.85, i.e. **L < ~0.42 s at exec_h = 8**.
  - Above that, use pause mode (stop-and-go) or a faster model.
  - Inside it, the actions actually executed are those with k/f > L − 0.01, so a larger L throws away more of each
    chunk.

**Latency budget on the Jetson Orin Nano (estimates; measure in B6):**

| term | estimate | notes |
|---|---|---|
| capture → packet (Argus + remap of 2 cams) | 30–60 ms | Argus delivery (~30 ms placeholder `offset_prior_ns`) + remap ~2–4 ms + age within the 60 Hz publish |
| task obs assembly + localhost serialize of 8.3 MB | 10–30 ms | shared-memory peeks + msgpack of 4 full frames. N5 pre-resize cuts it to < 5 ms |
| server pad+resize ×4 | ~5 ms | CPU |
| model (CLIP ×4 + 16 DDIM steps) | **bf16 0.15–0.30 s; fp32 0.30–0.60 s** | controlSW/parts.md option C. The first call is much slower (cuDNN autotune), so warm up at server start |
| reply + relative→absolute + schedule | < 5 ms | |
| **L total** | **bf16 ≈ 0.20–0.40 s; fp32 ≈ 0.35–0.70 s** | continuous mode needs L < ~0.42 s at exec_h 8: **bf16 is likely OK, fp32 marginal → pause mode** |

**One control cycle on the Jetson (continuous mode, f = 20 Hz, exec_h = 8, bf16, assumed capture latency 40 ms and
L ≈ 0.30 s). t = 0 is the newest main capture time T.**

| t (ms) | process (Jetson core) | event |
|---|---|---|
| −100 | camera (cores 0–1) | older UW frame captured (UW sample at T−100) |
| −50 | camera | older main frame captured |
| 0 | camera | newest main frame captured (T). UW paired by nearest capture time (±17 ms at 30 fps) |
| ~35 | camera | Argus delivery + remap done, packet published (60 Hz, fixed_rate) |
| ~40–55 | task (cores 2–4) | `get_observations`: main {T−50, T}, UW {T−100, T}. FK and width read at `t_obs−50, t_obs` |
| ~60–75 | agent | relative proprio from commanded history (actions 7, 8 of the previous chunk), 8.3 MB to localhost |
| ~75–320 | inference (GPU + 1 core) | resize ×4, CLIP ×4 + 16 DDIM steps in bf16 |
| ~330 | agent | (16,10) → absolute targets anchored at FK(t_obs), stamped T + k·50 ms |
| ~335 | task | `schedule_eef_traj(ts + 0.1)`, `schedule_joint_traj(ts)` |
| ~335 | FrankaCartesian (200 Hz) | trims points < now + 0.09: arm k ≤ 6 dropped, k = 7..15 merged |
| 350–750 | FrankaCartesian | target for action k at T + k·50 ms (+τ lookahead), gripper k at T + k·50 − 100 ms |
| ~335 | task | `wait_until(T + 330 ms)` has passed, so the next observation starts immediately. The next chunk arrives at ≈ T + 0.67 s, before this one ends at T + 0.85 s: no gap |
| every 1 ms | franka_bridge (isolated core 5, or the RT PC) | impedance law tracks the latest target |

With fp32 (L ≈ 0.5 s) the next chunk arrives at ≈ T + 1.0 s, after the 0.85 s end, so the arm holds for about 150 ms
every cycle. Use pause mode, or bf16 / DDIM 8.

**Pause mode (recommended for the first closed loop):**
1. Inference runs while the arm holds.
2. The reply is re-based: action k is due at `t_reply + k/f` (+λ for the arm). Actions 0..7 are executed.
3. The task waits until `t_reply + 0.35 + 0.08`, then observes the now-static scene.
4. Cycle time ≈ L + 0.45 s. This is mildly out of distribution (demos never pause), as for UMI.

---

## 2. Action path, precisely

### 2.1 Format

- `action_pred` (16,10), unnormalized: `[0:3]` pos (m), `[3:9]` rot6d, `[9]` width (m, absolute)
  (`detached_inference.py:459-472`).
- The pose is relative to the TCP pose at observation time, in that TCP frame. Action 0 ≈ identity.
- 20 Hz, so the 16 steps span 0.75 s.

### 2.2 PolicyAgent: relative to absolute

- `pose_9d_to_xyz_wxyz` → `get_absolute_pose(obs_last_eef, rel_k)` = `T_FK(t_obs) · rel_k`
  (`policy_agent.py:217-230`; `PKG/robot_utils/pose_utils.py:107-128`).
- Widths are used as they are (`:245`). Timestamps are `T + k/f` (`:247-251`). `smooth_action_window_size: 1`, so no
  smoothing.
- **Anchor choice (decision for B5).**
  - Measured FK under Cartesian impedance has a steady tracking error e (gravity/friction model error, a few mm at
    moderate stiffness).
  - Because every chunk is re-anchored on measured FK, a constant e becomes a drift of about e per chunk while the
    policy commands "stay". It also slightly shortens motions.
  - Option: anchor on the controller's **target** at t_obs instead (`BaseCartesianJointClient.get_eef_target_xyz_wxyz`
    already exists, `base_controller_client.py:385-431`). Guard: fall back to FK if |target − FK| > 2 cm (blocked or
    contact).
  - Implement it as a task flag. Try measured first (as UMI does), and switch if drift shows up.

### 2.3 Task: scheduling the chunk

- `run_policy_control` (`RE/tasks/umi_arx5_task.py:270-337`): re-base if needed, truncate to `exec_h` in
  pause/single-step, `schedule_eef_traj(abs_poses, ts + lookahead, absolute)` and
  `schedule_joint_traj(widths, ts)`, store the history, then wait.
- `exec_h`: use 8 at first (0.4 s per chunk). The training rollouts used 12 (0.6 s); try it in B5.
- Client → server: a `command` topic message `{command: SCHEDULE_EEF_TRAJ, data (N,7), timestamps, use_relative_timestamps}`
  (`base_controller_client.py:433-458`). Before sending, the client checks the controller's `error` topic and raises.

### 2.4 What the controller must implement (`BaseController`, CARTESIAN_JOINT)

From `RE/controllers/base_controller.py:18-403`:
- `__init__(robot_name, server_endpoint, control_mode="CARTESIAN_JOINT", ctrl_freq, config_str)`. The base creates the
  `command`, `error`, `joint_state`, `joint_target`, `eef_pose`, `eef_target` and `config` topics.
- `connect()` / `disconnect()` / `reset()`. `reset` is called inside the run loop and is blocking (move home). The
  client waits for more than 10 new `eef_pose` messages afterwards (`base_controller_client.py:311-334`).
- `get_eef_pose_and_joint_pos(wait_until_update)` → `(xyz_wxyz TCP in base, [width_m], t_monotonic)`. Timestamps must
  be **strictly increasing** (the client asserts it).
- `get_eef_target()`, `get_joint_target()`.
- `schedule_eef_traj(traj (N,7), abs_ts)` and `schedule_joint_traj(traj (N,1), abs_ts)`: update the interpolators.
- `send_robot_target()`: interpolate at `now + lookahead_time_s` and send the target.
- `record_data()`: `RobotCtrlLogger`s `right_arm` / `right_end_effector`, as in `arx5_cartesian.py:152-180`.
  `MainLogger.start_recording` validates every entry in `logger_endpoints` and raises if one is missing
  (`PKG/robologger/loggers/main_logger.py:106-121`). Either provide them or leave them out of the task config.
- Optional `handle_no_command_timeout` + `no_cmd_timeout_s` (`base_controller.py:284-294`).
- Templates:
  - `ARX5Cartesian` (`RE/controllers/arx5_cartesian.py`): CARTESIAN_JOINT, interpolators, dynamic latency, loggers,
    and the z-forward TCP conversion pattern;
  - `UR5Cartesian` (`RE/controllers/ur5_cartesian.py`): CARTESIAN only, `setTcp` (`:140`), `servoL` with lookahead
    (`:229-271`).
- Reuse `RE/common/interpolator.py` as it is. `PoseTrajInterpolator` has SLERP, trimming and merging.
  - **Caveat (verified):** if any segment exceeds `max_pos_speed_m_per_s` / `max_rot_speed_rad_per_s`, `update`
    **rejects the whole new trajectory** and keeps the old one (`interpolator.py:258-291`; joint version `:78-95`).
  - Conservative limits therefore make the robot hold or freeze, not slow down.
  - Plan: set the limits from the action stats (0.5 m/s, 1.5 rad/s) and do the real clamping in the bridge (§2.7).
  - Log every rejection.

### 2.5 Control stack: choice and justification

| option | how | pros | cons |
|---|---|---|---|
| **A. panda-py `CartesianImpedance`** (recommended) | C++ 1 kHz torque loop. Python calls `set_control(pos, quat)` at any rate. Built-in nullspace term and target filtering | Compliant, so grasping and table contact are safe. Matches UMI's Franka deployment (Cartesian impedance via Polymetis). Simple "latest target" interface fits the real-env interpolator | Steady-state tracking error (see §2.2). x86_64 wheels per libfranka version; aarch64 must be built from source |
| B. franky (`franky-control`) | Ruckig-planned Cartesian waypoint motions, asynchronous re-targeting | Respects Franka velocity/acceleration/jerk limits by construction, so few motion-generator reflexes. Precise | Position-controlled, so stiff on contact. Re-planning at 20–200 Hz is less smooth than impedance target streaming |
| C. libfranka C++ directly | own 1 kHz `cartesian_pose` or torque callback | full control | the most work, easy to trip `*_discontinuity` reflexes; don't |
| D. Polymetis / deoxys | | | Polymetis is unmaintained, deoxys is Panda-centric. Skip |

Pick A. Keep B as the fallback if the impedance tracking error is unacceptable. Both wrap libfranka, so the libfranka
version must match the robot: **Panda (system 4.x) → libfranka ≤ 0.9.x; FR3 (system 5.x) → libfranka ≥ 0.10**
(check the official compatibility table for the exact system version). FCI must be enabled on the robot (Desk).

### 2.6 Where the 1 kHz loop runs

The FCI needs a 1 kHz round trip. libfranka must answer each robot state packet within the 1 ms cycle. Occasional
lost packets are tolerated (the success rate is reported in `RobotState::control_command_success_rate`). Sustained
lateness triggers the `communication_constraints_violation` reflex, which stops the arm. **So the failure mode of a
jittery host is a safe stop, not unsafe motion.** That makes "try the Jetson first" acceptable.

**The open question: 1 kHz loop on the Jetson, or on a separate small RT PC?**

| aspect | on the Jetson Orin Nano (single computer) | separate x86 RT mini-PC |
|---|---|---|
| RT kernel | JetPack 6 ships an optional PREEMPT_RT kernel (the `nvidia-l4t-rt-kernel` packages). It has to work with the CSI/Argus camera drivers and the CUDA driver; verify both on the RT kernel | standard Ubuntu PREEMPT_RT (Franka's documented setup) |
| CPU | 6× Cortex-A78AE. Plan: core 5 isolated (`isolcpus=5 nohz_full=5 rcu_nocbs=5`) for the bridge's libfranka thread; cores 0–1 Argus + remap; cores 2–4 task, FrankaCartesian, inference Python. `nvpmodel` max mode + `jetson_clocks` (no DVFS stalls) | dedicated |
| Main jitter sources | **shared LPDDR5 (68 GB/s) contended by GPU inference, ISP and remap;** shared L3; GPU/ISP/NIC interrupts (pin the NIC IRQ to core 5, others away); thermal throttling on a hot gripper | none from inference/camera |
| Network | one onboard GbE port. It must be a **direct link to the Franka control box**, so SSH/dev access goes over Wi-Fi or a USB-Ethernet adapter | Franka link on the PC. The Jetson ↔ PC link is a second cable (or switch) |
| Software | libfranka, panda-py (pybind11) and franky need **source builds on aarch64** (libfranka ≥ 0.14 also needs pinocchio). There are no prebuilt wheels | prebuilt x86_64 wheels for panda-py/franky per libfranka version |
| Clock domain | one host, trivially consistent | still consistent: FrankaCartesian stays on the Jetson and only targets cross the link (below) |
| Cabling on the arm | Jetson on the gripper: one Ethernet cable from the gripper to the control box + power | same Ethernet cable to the PC instead + power. The PC is near the robot |
| Cost and complexity | $0, one box, matches the "single central computer" brief | +$150–250 (used NUC), one more box to power and maintain |
| Risk | unproven combination (aarch64 libfranka + RT + GPU load) | low, standard |

**Recommendation:**
1. **Build the bridge as its own process with a ZMQ endpoint**, so co-located (`tcp://127.0.0.1`) and remote
   (`tcp://<rt-pc>`) deployments are the same code.
2. **Try the Jetson first, with an explicit go/no-go (B3-hw step 1, 1–2 days):**
   - RT kernel booted with the cameras and CUDA working;
   - libfranka `communication_test`;
   - then the bridge holding the arm in impedance at home for 30 min while the Jetson runs, at the same time,
     RigCamera at 60 Hz, `detached_inference.py` in a back-to-back inference loop (bf16), the task, and `stress-ng --vm`
     on cores 0–4.
   - **Pass:** zero reflexes, `control_command_success_rate` ≥ 0.99 at all times (log the minimum per second), bridge
     loop p99.9 cycle time < 0.5 ms.
   - Repeat while the arm moves on a 1 Hz sine, and with the gripper warm (fan, enclosure).
3. **If it fails, or the RT kernel breaks Argus/CUDA,** move only the bridge to a used x86 PREEMPT_RT mini-PC.
   Nothing else changes.
4. Either way, the Franka link must be a direct cable (no switch) and the Jetson must not do anything else on that
   NIC.

**Process split (one clock domain in both layouts).**
- Every real-env timestamp is `time.monotonic()` of one host (R3 §0). RigCamera and the task must share a host
  (shared memory). So `FrankaCartesian` (BaseController, 200 Hz Python loop, interpolators, gripper driver) runs on the
  Jetson in both layouts.
- The bridge holds only the 1 kHz impedance loop. Its link (ZMQ REQ/REP or robotmq) has these requests:
  `set_target(pose_O_T_TCP)`, `get_state()` → `{O_T_EE, q, dq, O_F_ext_hat_K, robot_mode, success_rate, error}`,
  `move_joints(q_home)`, `recover()`, `stop()`.
- State timestamp = Jetson monotonic at the reply, minus RTT/2 (RTT is microseconds on localhost and about 0.2 ms on a
  direct link).
- No cross-host clock sync is needed. Jitter only affects the 5 ms target updates, which the bridge's 1 kHz loop
  smooths with the target filter. The bridge's own libfranka thread runs SCHED_FIFO on the isolated core. Its Python
  ZMQ thread stays off that core (panda-py runs the control loop in a C++ thread).
- Rejected: running `FrankaCartesian` itself in the bridge with timestamp conversion. That creates a second clock
  domain for no gain.

### 2.7 TCP frame setup

- **Training TCP:** `W_T_TCP = W_T_I · I_T_TCP`, with `I_T_TCP` from `gripper_calibration.yaml` (`iPhone17,1`, and
  `JetsonRig1,1` copies it):
  - R = [[1,0,0],[0,−0.966,0.259],[0,−0.259,−0.966]], t = (0, −0.0644, −0.2114) in the ARKit camera frame (x right,
    y up, z backward).
  - In the OpenCV frame C of the (virtual) main camera (x right, y down, z forward):
    `C_T_TCP = diag(1,−1,−1,1) · I_T_TCP` → R = [[1,0,0],[0,0.966,−0.259],[0,0.259,0.966]], t = (0, 0.0644, 0.2114).
  - So the TCP is 21.1 cm ahead of and 6.4 cm below the main camera, pitched 15°. +z is the approach direction along
    the fingers, +x is image-right (the finger-opening axis), +y is "down" in the image.
- **On the Franka:** `F_T_TCP = F_T_C · C_T_TCP`.
  - `F_T_C` (flange → virtual main camera) comes from CAD first, then from hand-eye calibration (B4).
  - Set it as the robot's EE frame so that `O_T_EE` = training TCP and the impedance stiffness is centred at the TCP.
    In libfranka: the nominal EE (`F_T_NE`) via the Desk end-effector config, and `NE_T_EE` via `setEE`. Also set
    `EE_T_K = I` (stiffness frame).
  - Also set the **payload** (mass, CoM, inertia of gripper + rig + compute) in Desk or with `setLoad`. A wrong mass
    degrades gravity compensation and collision detection.
  - Fallback: keep EE = flange and convert in `FrankaCartesian` (`O_T_TCP = O_T_F · F_T_TCP`, command
    `O_T_F = O_T_TCP · inv(F_T_TCP)`), as ARX5 does with `z_forward_to_x_forward_tranform`
    (`arx5_cartesian.py:33-80`). Then the impedance is centred at the flange, which is worse for compliance.
- **Why it matters:** actions are body-frame relative.
  - A rotation error in `F_T_TCP` rotates every commanded motion.
  - A translation error couples commanded rotations into spurious translations (lever arm).
  - The base frame and home pose don't affect the relative math (wrt_start is ignored). They only affect what the
    camera sees.

### 2.8 Safety

- **Hardware:** the external activation device (user stop) is in the operator's hand for every run. Keep a clear
  zone.
- **Bridge checks (Jetson core 5 or RT PC), in order, on every `set_target`:**
  1. Watchdog: if no target arrives for more than 0.2 s, hold the last target. If none arrives for more than 2 s, stop
     the controller (the arm brakes, then idles).
  2. Workspace box clamp on the TCP position, in the base frame, measured around the table, plus an orientation cone
     around the home approach axis (e.g. ±60°).
  3. Per-update step clamp: ≤ v_max·dt with v_max = 0.25 m/s at first, raised to 0.5 later. Rotation ≤ 1.0 rad/s.
     This is a clamp, not a reject.
  4. Moderate impedance to start: translational ~600–1000 N/m, rotational ~30–50 Nm/rad, nullspace posture toward the
     home configuration. Tune in B3-hw.
  5. libfranka `setCollisionBehavior` thresholds lowered from the defaults (for example about 20 N / 10 Nm for
     contact; tune). Keep libfranka's rate limiter and low-pass filter on.
- **Reflex handling:**
  - On a `franka::ControlException`, the bridge reports `error` in `get_state`. `FrankaCartesian` puts it on the
    `error` topic, so the next `schedule_*` raises in the task and policy mode stops.
  - Recovery (`automaticErrorRecovery`) only on an explicit operator command (task key or bridge CLI). Never
    automatic during a policy run.
  - Note bug N1 (§3): the base controller publishes errors on a topic that doesn't exist. Fix it first.
- **Task-side checks:**
  - Assert `agent_update_freq_hz == 20`, `crop_image is None` and `draw_gripper_mask is False`.
  - Refuse to start policy mode if |FK − home| > a threshold.
  - Pause mode by default.
  - Optionally a max chunk displacement check, for example reject a chunk with |Δpos| > 0.3 m and hold.

### 2.9 Gripper

- **The policy sees the fingers.** The UW camera images the iPhUMI fingers and AR tags, and `artag` geometry defines
  the width.
- **Recommended: motorize the iPhUMI gripper**, so the fingers, tags and camera views stay identical to training.
  - Use a serial-bus servo with position and current feedback. The Feetech STS3215 (~$20, LeRobot SO-100 class) is the
    budget pick; a Dynamixel XM430 with current-based position control is the premium one.
  - Driver: a `GripperDriver` class inside `FrankaCartesian` (USB serial on the Jetson):
    `set_width(m)`, `get_width()`, `set_current_limit()`.
- **Franka Hand:** non-real-time `franka::Gripper::move/grasp`, 0–80 mm, slow, and it **looks different** (out of
  distribution for an iPhone-trained policy). Only usable with iPhUMI-like finger tips and tags and the rig mounted at
  the same relative pose. Not recommended.
- **Width units:**
  - Training width = AR-tag centre distance − the minimum distance from the gripper-calibration episode, in metres,
    0 = closed (`iPhUMI/.../utils/gripper_util.py:44-55`, `utils/interpolation_util.py:47-49`).
  - The run_001 range is 0.045–0.084.
  - Calibrate the actuator → width map with the **same tag detector** on RigCamera UW frames while sweeping the servo
    (B4), then invert it for commands.
- **Grasp force:** the demos' grasp widths equal the eraser width (0.045). Position control to exactly 0.045 would
  barely touch the object. Use a current limit (a compliant squeeze) and command `w − δ` (δ ≈ 5 mm) when
  `w < w_open − ε`. Tune it.
- **Timing:** the gripper already leads the arm by 0.1 s (task convention). Keep it.

---

## 3. Done vs remaining (verified)

### 3.1 Exists

| piece | where | state |
|---|---|---|
| RigCamera + RigFrameSource | `RE/peripherals/rig_camera.py`, `sensor_rig/sensor_rig/deploy/frame_source.py`, configs `rig_wrist_right.yaml` / `rig_wrist_mock.yaml` | done, unit and integration tested without hardware (sensor_rig/docs/deployment.md §5). Not yet on hardware |
| PolicyAgent | `RE/agents/policy_agent.py` | works for single arm (`robot_num: 1`). 5 s timeout (E5) |
| Inference server | `BP/deployment/detached_inference.py` | works for single arm via sorted prefixes. Prompting supported |
| Action replay server | `BP/deployment/detached_inference_action_replay.py` | bimanual only (E7) |
| Controller base, client, interpolators | `RE/controllers/base_controller*.py`, `RE/common/interpolator.py` | reusable. Error-topic bug N1 |
| ARX5 / UR5 controllers | `RE/controllers/arx5_cartesian.py`, `ur5_cartesian.py` | templates |
| Tasks | `RE/tasks/umi_arx5_task.py` (`run_policy_control`), `iphumi_arx5_task.py` (correct obs), `iphumi_arx5_bimanual_task.py` (prompting) | none is usable unmodified for single-arm prompting |
| Spacemouse agent and server | `RE/agents/spacemouse_agent.py`, `PKG/robot_utils/teleop_utils/spacemouse/` | the client needs a server (state = 6 motion + 2 buttons float64, topic `spacemouse_state`, `tcp://localhost:15557`). **8 values, not 7 as R3 §7 says** |
| Checkpoint | `results/block_raise/run_001_2026-10-05/checkpoints/best_slim_epoch=0050-...ckpt` on the laptop | ok |
| Laptop GPU | RTX 3050 Ti Laptop, 4 GB (`nvidia-smi`) | enough for the ~1.6 GB fp32 model; latency unmeasured |

### 3.2 Missing or broken (each verified)

| # | issue | evidence | fix (package) |
|---|---|---|---|
| E1 | No single-arm task sends the prompt; unprompted inference errors out | `send_prompt_request` is only called at `iphumi_arx5_bimanual_task.py:401`. Assert at `prompt_obs_encoder.py:768` | new task re-prompts per episode and in single-step (B3) |
| E2 | `iPhumiARX5Task` uses an unset `self.task_name`; launcher wants `iphumi_arx5_<argv>.yaml` | `iphumi_arx5_task.py:64`, `:90`. `base_task.py:94-113` passes `task_name` to MainLogger without storing it | set it in the new task; optional 1-line upstream (B2) |
| E3 | `agent_update_freq_hz: 10` vs the trained 20 Hz | `RE/configs/agents/policy_agent.yaml:5` | task config override + assert (B3) |
| E5 | Hard 5 s inference timeout (also `policy_config` 5 s, `policy_reset` 5 s) | `policy_agent.py:42`, `:207`, `:257` | B2 diff |
| E7 | Action replay is bimanual-only; its action[k] is demo[t+(k+1)·stride], one step ahead of training's convention; stride 1 = 60 Hz frames (3× slow at 20 Hz) | `detached_inference_action_replay.py:134-142`, `:270-273`, `:310-313` | B2 diff; run with `--stride 3` |
| E4 | iPhone double latency subtraction | `iphone.py:90` + `base_camera.py:221` | not needed (RigCamera avoids it) |
| E6 | `umi_*` crop/mask trap | `umi_arx5_task.py:245-261`; `iphumi_arx5.yaml` inherits `crop_image: [256,256]` from `umi_arx5.yaml` (unused by the iPhUMI obs, but a trap) | new task sets `crop_image: null`, `draw_gripper_mask: false`, and asserts |
| E8 | VideoLogger defaults to `h264_nvenc` | `video_logger.py:30` | handled in RigCamera. Keep the camera out of `logger_endpoints` on the Orin Nano |
| E10 | `BaseCameraClient` peek race | `base_camera.py:351-391` | optional B2 diff |
| **N1 (new)** | Controller errors are published on topic `"errors"`, but the topic registered (and read by clients) is `"error"`. Controller-side failures never reach the task, and `put_data` to an unregistered topic may itself throw (unverified robotmq behaviour) | `base_controller.py:50` vs `:275`; client `base_controller_client.py:440`, `:522` | B2 1-line diff. **Required for safety** |
| **N2 (new)** | The prompt is cleared on every policy reset (stop, `R`); `C` single-step never prompts | `prompt_obs_encoder.py:852-856`, `detached_inference.py:280`, `base_task.py:156-157` | B3 task |
| **N3 (new)** | Interpolator rejects (doesn't clamp) over-speed chunks, so the robot holds silently | `interpolator.py:258-291`, `:78-95` | limits ≥ action stats; clamp in the bridge (B3) |
| **N4 (new)** | Endpoint mismatch: `umi_arx5.yaml` / `iphumi_arx5.yaml` use camera `tcp://localhost:16700` and logger `55554`; `rig_wrist_right.yaml` uses `16702` / `55556` | `RE/configs/tasks/umi_arx5.yaml` vs `rig_wrist_right.yaml:7-8` | new task config uses 16702 / 55556 |
| **N5 (new)** | 8.3 MB per inference request (4 full-res frames) | `policy_agent.py:199-202` | B6: optionally pre-resize to 224 on the client with the same `get_image_transform_with_border` (the server skips it when the shape is already 224, `detached_inference.py:428-431`) |
| **N6 (new)** | Spacemouse `pose_orientation_types` 0–3 assume the gripper z is horizontal (`RE/configs/agents/spacemouse_agent.yaml:6-12`). A top-down Franka home may need a new type | `RE/agents/spacemouse_agent.py:117-131` | check in B3-hw; small addition if needed |
| M1 | No Franka controller, bridge, task or configs | n/a | B3 |
| M2 | No mock controller and no mock spacemouse; `SpacemouseClient` crashes without a server (`spacemouse_client.py:13-23`) | `grep` finds none in real-env | B1 |
| M3 | No laptop environment with robotmq + robologger + robot_utils + torch + behavior_prompting (the scratch env has none of them) | P2 §6.6 | B1 |
| M4 | The prompt zarr isn't on the laptop | `find` found no `firstreal*zarr*` locally | B0 (scp from the cluster) |

---

## 4. Work packages (dependency order)

Sizes: S ≤ 1 day, M 2–5 days, L 1–2 weeks. Proposed code home: a repo-owned package
`controlSW/franka_deploy/` (Python package `franka_deploy`, with its own configs and tests). Upstream real-env and
behavior_prompting only get the B2 diffs. This keeps the vendored code minimal, like sensor_rig (D8).

### B0. Decisions and procurement (no code, start now)

- Decide: gripper actuator, Panda vs FR3 and its system version (which fixes the libfranka version), FCI licence and
  Desk access. Find out whether the lab already has an RT PC for the Franka (the free fallback).
- Check that the JetPack 6 RT kernel matches the installed L4T release. Plan a USB-Ethernet adapter or Wi-Fi for dev
  access, because the onboard GbE goes to the Franka.
- Order the servo (and a used mini-PC only if the B3-hw go/no-go fails). Design the gripper motor mount, and the rig + compute mount on the Franka
  flange (ISO 9409-1-A50).
- Copy `~/CapstoneProject/datasets/block_raise/firstreal-single/firstreal-single.zarr.zip` to the laptop (the user runs
  `scp` from `xfer.discovery.neu.edu`).
- DoD: written decisions in AGENTS.md; parts ordered.

### B1. Offline harness on the laptop (software only, M; needs B2 E5 + N1, or monkeypatches)

1. Environment: conda env `realenv`, Python 3.10.
   - real-env `env.yaml` pins: robotmq 0.1.14, robologger 0.1.7, robot-utils 0.1.4, hydra 1.3.2, zarr 2.18.3,
     numpy 2.2.6, opencv 4.12.
   - CUDA torch + the behavior_prompting deps; `pip install -e` behavior_prompting, real-env, sensor_rig.
2. `franka_deploy/mocks/mock_spacemouse_server.py`: an RMQServer `spacemouse_server` on `tcp://localhost:15557`,
   topic `spacemouse_state`, 8 float64 zeros at 100 Hz. Optional keyboard nudges.
3. `franka_deploy/mocks/mock_cartesian.py: MockCartesianController(BaseController, CARTESIAN_JOINT)`, using
   `PoseTrajInterpolator` / `JointTrajInterpolator`, 100 Hz. Modes:
   - `integrate`: FK = target filtered with a first-order lag τ = 50 ms, optionally plus a constant offset e to emulate
     impedance error (tests the §2.2 drift);
   - `replay`: FK = the demo's TCP trajectory (ARKit pose · `I_T_TCP`), time-aligned with the camera replay.
     Commands are only logged. This is "teacher forcing": every chunk can be compared with the demo's true future.
   - Optional `RobotCtrlLogger`s; the default task config has empty `logger_endpoints` (check that MainLogger
     accepts `{}`).
4. Camera: build a simulated raw rig episode from a held-out firstreal demo with `sensor_rig/sim` (it exists, used by
   `test_deploy_equivalence.py`), then `rig_camera wrist_mock backend_opts.episode_dir=<it>`.
   - For `replay` mode, use one shared clock origin for camera and controller. `MockCamera` with `restamp: true`
     re-bases each camera separately (deployment.md §7), so add a shared-origin option or start both from one t0
     file.
5. Inference: `python BP/deployment/detached_inference.py -i <best_slim ckpt> --dataset-path firstreal-single.zarr.zip --device cuda`.
6. Task: the B3 task class (`iphumi_franka`) with mock endpoints (B3-sw is a prerequisite for this step; do them
   together).
7. Instrumentation: `franka_deploy/tools/obs_tap.py` wraps PolicyAgent and dumps an npz per call (obs sent, reply,
   absolute targets, timestamps, `t_obs`, inference ms, trimmed count). `tools/check_contract.py` checks:
   - keys, shapes and dtypes exactly as in §0;
   - main image after the server transform vs the zarr image of the same demo frame: MAE ≤ 1.5 grey levels at 224
     (the export path measured ~0.6);
   - main sample spacing 50 ± 17 ms, UW 100 ± 17 ms, proprio 50 ms;
   - in `replay` mode, action error vs the demo future at k = 4, 8, 15. Compare with the training env-runner rollout
     on the same demo (the `UmiPromptingRunner` path gives the reference). Accept if the medians are within 1.5× of
     the offline numbers.
- DoD: 10 minutes each of pause and continuous mode with no exceptions. Contract checks pass. Measured laptop
  inference latency recorded. A 1-page report with error vs k.

### B2. Upstream fixes (software only, S). Exact minimal diffs, for approval

```diff
--- RE/agents/policy_agent.py
@@ def __init__(
         use_relative_action: bool,
         smooth_action_window_size: int,
+        inference_timeout_s: float = 5.0,
         **kwargs,
     ):
@@
-            topic="policy_config", data=serialize(True), timeout_s=5.0
+            topic="policy_config", data=serialize(True), timeout_s=max(5.0, inference_timeout_s)
@@
         self.smooth_action_window_size: int = smooth_action_window_size
+        self.inference_timeout_s: float = inference_timeout_s
@@ def predict_actions(
-            topic="policy_inference", data=serialize(obs_to_send), timeout_s=5.0
+            topic="policy_inference", data=serialize(obs_to_send), timeout_s=self.inference_timeout_s
--- RE/configs/agents/policy_agent.yaml
+inference_timeout_s: 5.0
```
Note: `self.inference_timeout_s` must be assigned after `super().__init__` or used via the local variable in the
config request, as shown.

```diff
--- RE/controllers/base_controller.py   (N1)
@@ def run(self):
                     self.rmq_server.put_data(
-                        topic="errors",
+                        topic="error",
```

```diff
--- RE/tasks/base_task.py   (E2, optional; the new task sets it anyway)
@@ def __init__(
         self.task_config: dict[str, Any] = json.loads(config_str)
+        self.task_name: str = task_name
```

E7, a sketch for `detached_inference_action_replay.py` (about 25 lines):
- In `__init__`, `self.arms = sorted(a for a in ("left","right") if f"data/gripper_{a}_eef_pos" in rb)`. This matches
  the inference server's sorted prefixes, so single arm gives robot0 = right.
- Load `demo_xyz_wxyz[a]` / `demo_gripper[a]` per arm.
- Compute `demo_initial_intergrip_xyz_wxyz` only when there are 2 arms (otherwise identity).
- `sides`: add the choice `"auto"` (the default), which maps to all present arms.
- In `_handle_inference`, loop `for arm_idx, a in enumerate(self.arms)`.
- Option `--action-offset {0,1}`, default 1 (current behaviour). 0 reproduces the training convention action[0] =
  current.

E10 (optional): in `BaseCameraClient.get_latest_images_dict_THWC`, peek the camera topics first and the timestamps
last, then keep the last `min(len)` of each.

- DoD: diffs applied after approval, and the B1 harness passes with them.

### B3. Franka controller, bridge, task, configs (L; software part now, hardware part later)

**B3-sw (no hardware):**
1. `franka_deploy/bridge/franka_bridge.py` (runs on the Jetson's isolated core, or on the fallback RT PC; same code):
   - a ZMQ REP server wrapping panda-py (`Panda`, `CartesianImpedance`) behind a `Backend` interface;
   - `FakeBackend` (kinematic lag model) for laptop tests;
   - the §2.8 safety checks in the bridge;
   - per-second loop stats: minimum `control_command_success_rate` and cycle-time percentiles;
   - `scripts/jetson_rt_setup.md` + a script: RT kernel install, `isolcpus`/IRQ affinity, `nvpmodel`/`jetson_clocks`,
     the aarch64 source build of libfranka + panda-py (cross-check on an aarch64 docker image on the laptop);
   - a CLI: `franka_bridge --robot-ip ... --config bridge.yaml`, plus `recover` and `home` subcommands.
2. `franka_deploy/controllers/franka_cartesian.py: FrankaCartesian(BaseController, "CARTESIAN_JOINT")`:
   - `ctrl_freq` 200 Hz.
   - `connect`: bridge hello, check that the reported EE matches `F_T_TCP` in the config, seed the interpolators from
     the current FK and width.
   - `send_robot_target`: interpolate at `now + lookahead_time_s`, call `bridge.set_target`, and
     `gripper.set_width(cmd)` at ≤ 50 Hz.
   - `get_eef_pose_and_joint_pos`: cached bridge state from the same RPC, monotonic-stamped, strictly increasing.
   - `reset`: `bridge.move_joints(q_home)`, then re-seed the interpolators.
   - Loggers `right_arm` / `right_end_effector`. Bridge errors go to the `error` topic.
   - `handle_no_command_timeout`: hold, don't home.
   - Gripper plug-in: `GripperDriver` (`FeetechGripper`, `DynamixelGripper`, `FakeGripper`) with a width calibration
     table.
3. `franka_deploy/tasks/iphumi_franka_task.py: iPhumiFrankaTask(iPhumiARX5Task)` (about 80 lines):
   - `__init__(task_name, prompt_index, anchor="measured"|"target", ...)` sets `self.task_name`.
   - Asserts `policy_agent.agent_update_freq_hz == 20`, `crop_image is None`, `not draw_gripper_mask`, `robot_num == 1`.
   - Overrides `start_episode` (prompt if not prompted), `run_policy_control` (prompt lazily, optional anchor
     switch, optional chunk sanity check) and `reset_policy_agent` (clear `_prompted`).
   - Optional home-pose check before `c`.
   - It reuses `UmiARX5Task`'s `BaseCartesianJointClient` unchanged: `arx5_controller_endpoint` is just an endpoint.
     Renaming the parameters is cosmetic.
4. Configs:
   - `franka_deploy/configs/tasks/iphumi_franka.yaml` (defaults: real-env `policy_agent` + `spacemouse_agent`):
     - `agent_update_freq_hz: 20`, `action_execution_horizon: 8`, `inference_timeout_s: 30`;
     - `pause_when_policy_inference: true`;
     - `wrist_camera_endpoint: tcp://localhost:16702`, `logger_endpoints: {}`, controller `tcp://localhost:26010`;
     - `arx5_lookahead_time_s: 0.1`, `match_camera_latency: false`, `crop_image: null`, `draw_gripper_mask: false`;
     - `task_name: block_raise`, `prompt_index: 0`.
   - `configs/controllers/franka_cartesian.yaml`:
     - `bridge_endpoint: tcp://<rt-pc>:5555`, `ctrl_freq: 200`, `lookahead_time_s: 0.1` (tune +τ);
     - speed limits 0.5 m/s / 1.5 rad/s, `trim_new_trajectory_time_s: 0.09`, `dynamic_latency: false` initially;
     - `F_T_TCP` 4x4, `q_home`, gripper block.
   - `configs/bridge.yaml`: workspace box, step clamps, impedance, collision thresholds, watchdog.
5. Launchers: `franka_deploy/launch/{mock,real}.sh` start camera, controller, spacemouse, inference and task in tmux
   panes (same layout as `RE/shell_scripts/kill_all_processes.sh` ports).
6. Tests (pytest, laptop):
   - FakeBackend round trip;
   - interpolator → bridge clamp behaviour (over-speed chunks are clamped and logged, not dropped);
   - error propagation reaches the task as `RuntimeError` (needs N1);
   - prompting on `c`, `C` and after a stop;
   - the timing test from §1.4 with an injected inference delay of 0.1–0.6 s.
- Optional M: a MuJoCo FR3/Panda model (mujoco_menagerie) behind `Backend` with a Cartesian impedance law, to tune
  limits and stiffness before touching the robot.
- DoD: the B1 harness runs with `FrankaCartesian` + `FakeBackend` in place of the mock controller; all tests pass.

**B3-hw (needs the arm and the Jetson):**
1. **Jetson RT go/no-go (1–2 days, gates the host decision; §2.6):**
   - install the RT kernel and confirm Argus (both cameras) and CUDA still work;
   - build libfranka + panda-py; run `communication_test`;
   - run the 30-min co-load test: impedance hold, then a 1 Hz sine, with RigCamera + back-to-back bf16 inference +
     the task + `stress-ng --vm` on cores 0–4.
   - Pass = zero reflexes and success rate ≥ 0.99. Fail = move the bridge to an x86 PREEMPT_RT mini-PC (prebuilt
     wheels, about 1 day of setup) and continue.
2. Run the bridge alone: read state, move home, 2 cm steps on each axis at 0.05 m/s, then a 1 Hz sine of ±3 cm.
   Tune stiffness and the target filter.
3. Run `FrankaCartesian` on the Jetson: spacemouse teleop through the real-env task (`c` not pressed).
   Check `pose_orientation_types` (N6). Measure tracking error and lag τ, and set `lookahead_time_s = 0.1 + τ`.
4. Stress: run inference and RigCamera on the Jetson during teleop. Bridge loop stats: no
   `communication_constraints_violation` in 30 min.
- DoD: smooth teleop, measured τ and e, the stress test passes, and both e-stop and reflex recovery have been
  exercised.

### B4. TCP and frame calibration, width calibration, open-loop replay (hardware, M; needs the rig from Piece A)

1. CAD `F_T_C` (flange → virtual main camera) and `F_T_TCP = F_T_C · C_T_TCP` (§2.7).
2. Hand-eye check:
   - Fix a ChArUco board. Take 20 arm poses with diverse rotations.
   - At each, take a RigCamera virtual-main image (rectified, K known) and solve PnP for `C_T_board`. Read
     `O_T_F` from the bridge.
   - Run `cv2.calibrateHandEye` (Tsai/Park) for `F_T_C`. Accept if it agrees with CAD within 3 mm / 1°. Otherwise use
     the calibrated value.
   - This also checks the rig's virtual-camera rotation (calib-align).
3. Set the EE in Desk and with `setEE` (and the payload). Pivot test: command ±20° rotations about EE x/y/z. The TCP
   (fingertip centre) must stay on a marked point within 2 mm.
4. Width map: sweep the servo, detect the finger tags in the UW (same detector and calibration as iPhUMI), and fit
   `servo_pos ↔ width`. Check that 0 = closed matches the calibration convention and that the max covers 0.085.
5. Open-loop replay with the patched action replay server (E7):
   - `--stride 3 --action-offset 0`, task pause mode, `exec_h 16` (`policy_agent_action_replay.yaml`), starting from a
     pose resembling the demo start;
   - at 0.5× speed first (stride 3 with `agent_hz` 10), then 1×;
   - log FK and compare the relative trajectory with the demo's (it's frame-invariant).
- DoD: hand-eye vs CAD within tolerance; pivot test passes; replay relative error ≤ 1 cm / 3° over 5 s at 1×; the grasp
  closes on the eraser with the current limit.

### B5. Closed loop (hardware, M–L)

1. Pause mode, `exec_h 8`, prompt index 0, eraser in a canonical spot. 10 trials. Record success and the
   attention-vis video (`--attention-vis`).
2. Sweeps (one at a time, 10 trials each):
   - prompt index (3 different demos);
   - `exec_h` 8 vs 12;
   - `match_camera_latency` true vs false;
   - anchor measured vs target (§2.2).
3. Continuous mode once L < 0.4 s (B6). Then position randomization.
- DoD: ≥ 7/10 successes in pause mode at the canonical spot. Failure modes documented. AGENTS.md updated.

### B6. Performance on the Jetson Orin Nano (software; laptop for the method, Jetson for the numbers; M)

1. Time `predict_action` on the laptop 3050 Ti: cold and warm, fp32 and bf16 autocast (training used bf16 mixed
   precision), DDIM 16 vs 8 (check action change vs 16 on the B1 tap data).
2. Payload: measure `request_with_data` time for 8.3 MB on localhost and over GbE. If it's over 20 ms, add the client
   pre-resize (N5): about 20 lines in the task, using the same function.
3. Jetson Orin Nano: same matrix (fp32 / bf16 autocast / DDIM 16 vs 8), plus `torch.backends.cudnn.benchmark`, a
   warm-up call at server start, and the full end-to-end L from the task's npz tap (capture → schedule) **with
   RigCamera and the bridge running**.
4. **Memory (8 GB shared):** the slim checkpoint (2.5 GB) is unpickled into CPU RAM, then ~1.6 GB of weights plus a
   ~1 GB CUDA context go to the GPU; RigCamera holds 2 × 0.5 GB shared memory, plus Argus buffers.
   - Measure the peak with `tegrastats`.
   - If it's tight, export an inference-only checkpoint (EMA weights only, optionally bf16; about 30 lines reusing the
     server's loading code), and add NVMe swap.
5. CPU placement: inference Python on cores 2–4, away from camera cores 0–1 and isolated core 5. Check that the remap
   p99 stays ≤ 4 ms under inference load.
6. If L stays > 0.42 s: TensorRT/torch.compile for the CLIP encoders, DDIM 8, or stay in pause mode. A smaller retrained
   model is the last resort.
- DoD: a Jetson latency/memory table, a chosen precision and steps, and the decision between continuous and pause
  mode.

### Critical path and parallelism

```
B0 decisions/procurement ──► B3-hw (Jetson RT go/no-go → bridge on arm) ──► B4 (TCP + width + replay) ──► B5 closed loop
   │                               ▲                         ▲
   │   B2 diffs ──► B1 harness ────┤ (B3-sw tested in B1)    │ Piece A: rig mounted + calibrated rig.yaml
   └── B3-sw ──────────────────────┘                         │ gripper actuator built
       B6 (laptop method → Jetson numbers, no robot) ───────► bf16/steps, pause vs continuous
```
- Buildable now, without hardware: B1, B2, B3-sw (including the optional MuJoCo backend), the B6 method. As soon as
  the Jetson is in hand, even without the robot: B6 numbers and the RT-kernel + Argus + CUDA half of the go/no-go.
  Estimate: about 2 weeks for one person.
- Hardware critical path once parts arrive: B3-hw (2–4 days) → B4 (3–5 days) → B5 (1–2 weeks). The likeliest delays
  are the gripper actuator mechanics, the aarch64 libfranka/RT bring-up (with the RT-PC fallback), and FCI access.

---

## 5. Risks and decisions for the user

| decision / risk | options | recommendation |
|---|---|---|
| **1 kHz loop host** (open question) | (a) on the Jetson: RT kernel, isolated core 5, direct GbE; (b) small x86 PREEMPT_RT PC running only the bridge | **(a) first, gated by the B3-hw go/no-go; (b) as the fallback.** Same bridge code either way, and a Jetson overrun produces a reflex stop, not unsafe motion |
| **Inference precision / mode** | fp32 (0.3–0.6 s) vs bf16 (0.15–0.3 s) on the Orin Nano; pause vs continuous | bf16 + continuous if B6 confirms L < 0.42 s; otherwise pause mode. The laptop GPU is only for B1 development |
| Jetson memory (8 GB shared) | | inference-only checkpoint + NVMe swap if `tegrastats` shows pressure (B6) |
| RT kernel vs Argus/CUDA compatibility | | verify on day 1 of B3-hw. If it breaks, fallback (b) |
| Single NIC used by the Franka | | USB-Ethernet or Wi-Fi for development access |
| **Gripper hardware** | motorized iPhUMI gripper vs Franka Hand | motorize the iPhUMI gripper (identical visuals and width semantics). The Franka Hand is out of distribution |
| **Panda vs FR3** | fixes the libfranka version, joint limits, Desk/FCI licence | find out now. Pin panda-py/franky to that libfranka |
| Viewpoint/photometry mismatch (rig vs iPhone) | | Piece A risk. A remap can't move optical centres (deployment.md §7). If B5 fails on perception, record rig data and fine-tune or retrain |
| Impedance tracking error → drift with the measured anchor | measured vs target anchor | test in B1 (mock offset) and B5 |
| Interpolator rejects fast chunks | limits vs clamp | bridge clamps; interpolator limits ≥ data stats |
| Continuous mode infeasible if L > 0.42 s | pause mode, smaller model, DDIM 8 | pause mode first |
| Pause mode is out of distribution (static images after each chunk) | | accept for v1 (UMI precedent); continuous later |
| Gravity/payload model wrong (gripper + rig + compute ≈ 1–1.5 kg on the flange) | | identify it in Desk; check the FR3 3 kg payload |
| Prompt choice | which train demo | start with an index whose start pose resembles the robot's start. Try 3 |
| Safety bug N1 hides controller errors | | fix before any hardware run |

---

## 6. Open questions

- Exact `meta/task_names` and the train-split indices in `firstreal-single` (check when the zarr is local).
- Does `MainLogger` accept empty `logger_endpoints` with `Morphology.SINGLE_ARM`? If not, provide the loggers or one
  dummy.
- robotmq behaviour on `put_data` to an unregistered topic (N1): does it raise or drop silently?
- panda-py `CartesianImpedance` parameter names, defaults and filter for the chosen libfranka version (read its docs at
  install time). The franky async API, if it becomes the fallback.
- Gripper current-limit tuning and the squeeze offset δ for the eraser.
