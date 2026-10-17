# R3: Deployment observation path, and what a replacement sensor peripheral must provide

Scope: read-only study of how observations are produced at deployment for the `block_raise` run_001 policy, and the
contract a Jetson-based sensor rig must satisfy to replace the iPhone.

Path abbreviations used below:
- `RE/` = `/home/ruben/Capstone/behavior_prompting/deps/real-env/real_env/`
- `BP/` = `/home/ruben/Capstone/behavior_prompting/behavior_prompting/`
- `IP/` = `/home/ruben/Capstone/iPhUMI/python_package/iphumi/`
- `IOS/` = `/home/ruben/Capstone/iPhUMI/ios_app/iPhUMI/`
- `RUN` = `/home/ruben/Capstone/results/block_raise/run_001_2026-10-05/run_config/config.yaml`
- Third-party packages (robotmq 0.1.14, robologger 0.1.7, robot-utils 0.1.4) were downloaded and unpacked to
  `<scratchpad>/pkgs/ex/` for inspection
  (they aren't installed on the laptop).

---

## 0. TL;DR

1. **Process layout.** There are 4 or 5 processes connected by robotmq (ZeroMQ, plus POSIX shared memory for images):
   camera server (`BaseCamera` subclass), arm controller server (`BaseController` subclass), spacemouse server,
   inference server (`BP/deployment/detached_inference.py`), and the task process (a `BaseTask` subclass that owns a
   `PolicyAgent`). All timestamps are `time.monotonic()` on one host.
2. **Image path.** The camera publishes full-resolution 4:3 RGB uint8 frames (iPhone: 960x720). The task picks frames by
   timestamp. The inference server pads each frame to a square with black bars and resizes it to 224x224 with
   `cv2.INTER_AREA` (`get_image_transform_with_border`), which is exactly what training did. The model then applies
   CenterCrop 0.95 inside itself.
3. **Pose comes only from robot FK, never from the phone.** The iPhone streams no pose at deployment. Also, with
   `action_as_proprio: true`, the eef-pose and gripper-width observations the model sees are the robot's **previously
   commanded actions**, not measured state. Measured FK is used only as the reference for turning relative actions
   into absolute ones.
4. **The Jetson peripheral's job.** Subclass `BaseCamera`, implement one method that returns
   `{"main": HxWx3 uint8 BGR, "ultrawide": HxWx3 uint8 BGR}` plus a `time.monotonic()` capture timestamp. Output must
   look like iPhone-16-Pro frames: 4:3, rectified to the iPhone intrinsics, the same mounting relative to the TCP,
   and 60 fps.
5. **Blocking gaps in the existing code for single-arm `block_raise`:**
   - No single-arm task sends the behavior prompt (only the bimanual task does).
   - `iPhumiARX5Task` uses `self.task_name`, which is never set.
   - Its launcher needs a config file that doesn't exist.
   - The default `agent_update_freq_hz: 10` doesn't match the trained 20 Hz.
   - `predict_actions` has a hard 5 s RMQ timeout.
   - The action-replay server only handles bimanual data.

---

## 1. real-env architecture

### 1.1 Peripherals: `BaseCamera` (`RE/peripherals/base_camera.py`)

**Constructor** (`base_camera.py:40-108`):
`BaseCamera(name, server_endpoint, logger_endpoint, camera_latency_s, camera_configs, display_resolution_WH, config_str="")`

- `name`: RMQ server name and VideoLogger name. It must follow the robologger convention
  `right_wrist_camera_<i>` / `left_wrist_camera_<i>` / `head_camera_<i>` / `body_camera_<i>` / `third_person_camera_<i>`
  (`robologger/classes.py:19-33`, enforced by `MainLogger._validate_video_logger_names`, `main_logger.py:325-335`).
- `server_endpoint`: e.g. `tcp://localhost:16700`. Creates `RMQServer(server_name=name, ...)` (`:69-71`).
- `camera_configs`: dict `cam_name -> {type: rgb|depth, width, height, fps, ...}`. For **each camera** it calls
  `add_shared_memory_topic(cam_name, message_remaining_time_s=1.0, shared_memory_size_gb=0.5)` (`:83-86`). That gives
  a time-bounded ring buffer holding about 1 s of frames. Other topics: `timestamp`, `command`, `error` (10 s),
  `config` (`:88-91`).
- `VideoLogger(name, endpoint=logger_endpoint, attr={"camera_configs": ...})` (`:97-101`). It requires
  `width/height/fps/type` per camera (`video_logger.py:46-63`).
- `config_str`: the JSON dump of the whole Hydra config. It's published on topic `config` at `run()` start
  (`:153-156`). Clients **require** it: `BaseCameraClient.get_config()` reads `config["camera_configs"]` to learn
  camera names, shapes and fps (`:321-332`).

**The abstract method** (`:114-147`): `get_images_bgr_hwc_and_timestamp() -> (dict[str, np.uint8 HxWx3 BGR], float)`.
The timestamp must be a real `time.monotonic()` value for when capture started. The docstring asks for fallbacks
rather than empty results.

**Main loop `run()`** (`:149-307`), per iteration:
1. Drain the `command` topic. Only `CameraCommand.RESET` is supported; it calls `self.reset()`, which does nothing by
   default (`:110-112`, `:165-181`).
2. Call the subclass's grab method (`:184`), then print FPS.
3. Assert every frame's shape is exactly `(height, width, 3)` for rgb, or `(height, width)` for depth (`:197-217`), and
   that `timestamp > 0` (`:218-220`).
4. `corrected_timestamp = timestamp - camera_latency_s` (`:221`).
5. Convert rgb **BGR→RGB** (`cv2.cvtColor`, `:225-228`), then `put_data(topic=cam_name, data=frame.tobytes())` for
   each camera, followed by **one** `put_data("timestamp", serialize(corrected_timestamp))` (`:235-240`). So each
   iteration publishes one frame per camera plus one shared timestamp. Clients assume the topic lengths line up.
6. If recording, `video_logger.log_frames({cam: {"frame": rgb, "timestamp": corrected}})` (`:245-270`). VideoLogger
   pipes frames into ffmpeg with `codec="h264_nvenc"` by default (`video_logger.py:30`, `:94-125`, stderr sent to
   DEVNULL).
7. If `display_resolution_WH` is set, show each stream with `cv2.imshow` (`:272-307`). This must be `null` on a
   headless Jetson.

**Client `BaseCameraClient`** (`:310-403`):
- `get_latest_images_dict_THWC(frame_num)` peeks the last `frame_num` timestamps (`:351-361`), then the last
  `frame_num` raw buffers for each camera, and reshapes them as uint8 `(H,W,3)` (rgb) or float32 `(H,W)` (depth)
  (`:365-391`). It asserts the timestamps are non-decreasing (`:393-395`) and returns
  `({cam: (N,H,W,3)}, timestamps(N,))`.
- Note: the timestamp topic and each camera topic are peeked separately, so a frame published between the peeks can
  shift them by one frame. It's minor at 60 fps, but it's a real race.
- **Shared memory means the client must run on the same host as the camera server.** robotmq passes a POSIX shm name
  over ZMQ and the client `mmap`s it (`robotmq/core/src/rmq_client.cpp:152-186`).

**Concrete subclasses:**
- `iPhone` (`RE/peripherals/iphone.py:19-154`), described in §4.
- `WebCam` (`RE/peripherals/webcam.py:15-95`): OpenCV `VideoCapture`; `camera_id`, `cam_img_format` (FOURCC);
  autofocus off. Timestamp = `CAP_PROP_POS_MSEC/1000` (`:74`). On V4L2 that's driver/buffer time, **not guaranteed to
  be `time.monotonic()`**, so don't copy this pattern. Main camera only.
- Launchers: `run_iphone()` (`iphone.py:157-172`), `run_webcam()`/`run_gopro()` (`webcam.py:98-131`). Each composes
  the Hydra config, sets `cfg.config_str = json.dumps(cfg)`, instantiates, and calls `.run()`. They're registered as
  console scripts in `pyproject.toml` (`iphone`, `webcam`, `gopro`, ...).

**Peripheral configs** (`RE/configs/peripherals/`):

| file | name | server | logger | cams |
|---|---|---|---|---|
| `iphone_wrist.yaml` | `right_wrist_camera_0` | `tcp://localhost:16700` | `55554` | main 960x720@30, ultrawide 960x720@30 (depth commented out), `camera_latency_s: 0.08`, display 320x240 |
| `iphone_wrist_right.yaml` | `right_wrist_camera_0` | `16702` | `55556` | inherits |
| `iphone_wrist_left.yaml` | `left_wrist_camera_0` | `16701` | `55555` | inherits |
| `iphone_third.yaml` | `third_person_camera_0` | `16703` | `55557` | same cams |
| `gopro0.yaml` / `webcam0/1.yaml` | | | | WebCam, main only |

Each iPhone camera entry also carries `peertalk_port` (5555 main, 5556 ultrawide, 5557 depth).

### 1.2 Controllers

**`BaseController`** (`RE/controllers/base_controller.py:18-403`):
- `__init__(robot_name, server_endpoint, control_mode ∈ {JOINT, CARTESIAN, CARTESIAN_JOINT}, ctrl_freq, config_str)`.
- Topics: `command`, `error`, and, depending on the mode, `joint_state`/`joint_target`/`eef_pose`/`eef_target`, plus
  `config` (`:41-106`).
- Abstract methods: `connect`, `disconnect`, `reset`, `send_robot_target` (interpolate and send),
  `record_data` (`:119-202`).
- Optional, depending on the mode: `get_joint_pos`, `get_joint_target`, `get_eef_pose_and_joint_pos`,
  `get_eef_pose`, `get_eef_target`, `schedule_joint_traj(traj, abs_timestamps)`,
  `schedule_eef_traj(traj_xyz_wxyz, abs_timestamps)`.
- `run()` (`:206-403`) loops at `ctrl_freq`:
  1. Drain commands: RESET / CONNECT / DISCONNECT / SCHEDULE_JOINT_TRAJ / SCHEDULE_EEF_TRAJ. Relative timestamps
     are converted to absolute by adding `time.monotonic()` (`:236-259`).
  2. Publish state `{"eef_xyz_wxyz", "timestamp"}` / `{"joint_pos", "timestamp"}`.
  3. `wait_until` the next tick, then `send_robot_target()`, `record_data()`, and publish targets.
  4. Optional no-command timeout hook (`:284-294`).
- Pose convention: `[x, y, z, qw, qx, qy, qz]`, timestamps in `time.monotonic()`.

**`BaseCartesianJointClient`** (`RE/controllers/base_controller_client.py:270-541`), used by the ARX5 tasks:
- Waits for the `eef_pose` and `joint_state` topics. `dof` = length of `joint_pos` (1 for the gripper).
- `get_eef_xyz_wxyz(t)` / `get_joint_pos(t)` interpolate the buffered state at time `t`. They return the last sample
  if `t` is newer and assert strictly increasing timestamps (`:338-383`, `:461-506`).
- `get_eef_target_xyz_wxyz(t)` (`:385-431`).
- `schedule_eef_traj` / `schedule_joint_traj` raise if the `error` topic has anything (`:433-541`).

**ARX5 reference implementation** (`RE/controllers/arx5_cartesian.py`):
- Mode CARTESIAN_JOINT; the "joint" is the 1-DoF gripper width.
- Converts between the arm SDK's x-forward EE frame and the **z-forward TCP frame used by the policy**
  (`z_forward_to_x_forward_tranform`, `:33-52`; inverse `:55-80`).
- SDK timestamps are offset into `time.monotonic()` (`:205-208`, `:437`).
- `schedule_eef_traj` does dynamic latency matching (`find_delta_latency`) before updating the interpolator
  (`:455-496`).
- Two `RobotCtrlLogger`s, `right_arm` and `right_end_effector`, at the cartesian/joint logger endpoints (`:155-180`).
- Config: `RE/configs/controllers/arx5_cartesian.yaml` (ctrl 100 Hz, lookahead 0.1 s, trim 0.09 s).

**Where the Franka plugs in:** a new `FrankaCartesian(BaseController)` in CARTESIAN_JOINT mode (so the existing
`BaseCartesianJointClient` and the ARX5-style tasks work unchanged). It must:
- publish `eef_pose` (TCP pose in the base frame, z-forward / iPhUMI TCP convention) and `joint_state`
  (`[gripper_width_m]`) with monotonic timestamps;
- accept absolute-timestamp `schedule_eef_traj` / `schedule_joint_traj` calls;
- provide `get_eef_target` / `get_joint_target`;
- create `RobotCtrlLogger`s named `right_arm` / `right_end_effector`. Otherwise drop those keys from the task's
  `logger_endpoints`, or `MainLogger.validate_logger_endpoints` raises when recording starts
  (`main_logger.py:106-121`).

### 1.3 Tasks (`RE/tasks/`)

- **`BaseTask`** (`base_task.py:89-314`): keyboard loop. Terminal keys: `c` start policy, `C` single step, `r` reset,
  `R` reset agent, `q` quit, `x`, `e`, `i`, `?`. Global keys: `s` / `y` / `n` stop. `MainLogger` uses
  `logger_endpoints`, and `self.clients` collects configs for each episode. A `SpacemouseAgent` is required.
  - Note: `task_name` goes to `MainLogger` but is **not stored as `self.task_name`** (`:94`, `:113`).
- **`UmiUR5Task`** (`umi_ur5_task.py`): UR5 + WSG50, main camera only, keys `robot0_*`.
  - `get_observations` (`:195-284`): applies `draw_gripper_mask` (default true in `umi_ur5.yaml:20`) and
    `crop_image: [256,256]` via `resize_with_cropping`, a **center crop** (`:273-277`; `umi_ur5.yaml:43`).
  - Runs a warmup inference in `__init__` (`:142-155`).
- **`UmiARX5Task`** (`umi_arx5_task.py`): same idea with one `BaseCartesianJointClient`.
  - fps-divisibility assert (`:117-120`), mask, crop (`:257-261`), `run_policy_control` (`:270-335`).
  - No prompt sending. No `task_name` attribute.
- **`iPhumiARX5Task`** (`iphumi_arx5_task.py:17-82`): overrides `get_observations` (`:24-82`).
  - Adds `robot0_ultrawide_camera`. **Doesn't crop and doesn't mask** (good: it matches training).
  - Includes `"task_name": self.task_name` (`:64`). **That attribute is never set, so it raises AttributeError.**
  - `run_iphumi_arx5_task` composes `iphumi_arx5_<argv1>` (`:90`), but only `iphumi_arx5.yaml` exists.
  - Inherits `UmiARX5Task`, so it **never sends a prompt**.
- **`iPhumiUR5Task`** (`iphumi_ur5_task.py`): old variant. Uses `get_latest_images_dict_THWC(1)` and requires a
  `depth` key (`:70`), so it's not usable for run_001.
- **`iPhumiARX5BimanualTask`** (`iphumi_arx5_bimanual_task.py`): the most up-to-date task and the one the inference
  server was written against.
  - `robot0` = left, `robot1` = right (`:477-488`).
  - Cross-arm obs (`:490-503`), wrt-start obs (`:505-524`).
  - Prompting: `_send_prompt` → `policy_agent.send_prompt_request(task_name, prompt_index)`, called automatically in
    `start_episode` (`:395-410`).
  - `run_policy_control` (`:557-619`).
  - Config: `RE/configs/tasks/iphumi_arx5_bimanual.yaml`.

**For single-arm `block_raise` on a Franka you need a new task:**
- Observations like `iPhumiARX5Task.get_observations`, plus prompting like the bimanual task.
- A `FrankaCartesian` client.
- `crop_image: null`, `draw_gripper_mask: false`.
- Single-arm robot index mapping: the server maps the sorted prefixes `['gripper_right']` → `robot0`
  (`detached_inference.py:134-162`).

### 1.4 Agents (`RE/agents/policy_agent.py`)

- `__init__` (`:20-77`): `RMQClient` to `policy_server_endpoint`, then a `policy_config` request (5 s timeout,
  `:41-44`).
  - Sets `proprio_history_len = action_history_len = proprio_length + 1` = `low_dim_obs_horizon` = 2.
  - `image_history_len = img_obs_horizon` = 2 (`:45-58`).
- Agent config `RE/configs/agents/policy_agent.yaml`: `agent_update_freq_hz: 10`, `action_prediction_horizon: 16`,
  `action_execution_horizon: 8`, `keep_one_step_dummy_proprio: true`, `action_as_proprio: true`,
  `use_relative_action: true`, `smooth_action_window_size: 1`, endpoint `tcp://localhost:18765`.
- `cam_obs_freq_hz` (`:82-101`):
  - main = `agent_update_freq_hz`;
  - ultrawide = `agent_update_freq_hz * (10/ultrawide_down_sample_steps) / (60/obs_down_sample_steps)`.
  - For run_001 that's ultrawide = agent_hz × 10/20 = agent_hz / 2.
- `predict_actions` and `send_prompt_request` are covered in §2.

### 1.5 Inference server: `BP/deployment/detached_inference.py`

- CLI (`:475-503`): `-i ckpt`, `--endpoint tcp://0.0.0.0:18765`, `--dataset-path` (prompt demos; required for
  prompting), `--prompt-from-dataset train|val`, `--device cuda|cpu`, `--vis-prompt`, `--attention-vis`,
  `--eval-dataset`.
  - `docs/robot_deployment.md` calls it `detached_inference_real_env.py`; the actual file is `detached_inference.py`.
- Loads the checkpoint's `cfg` and workspace and moves the policy to the device (`:87-117`).
- Builds `_obs_key_map`: `robot{i}_{eef_pos,eef_rot_axis_angle,gripper_width,eef_rot_axis_angle_wrt_start}` →
  `gripper_<arm>_*`, and `robot{i}_{main,ultrawide}_camera` → `camera_<arm>_{main,ultrawide}_rgb` (`:141-162`).
- Topics (`:212-217`, loop `:250-370`):
  - `policy_config` → `{workspace: {proprio_length, img_obs_horizon, obs_down_sample_steps,
    ultrawide_down_sample_steps}, use_prompting, policy_name, run_name, epoch, ...}` (`:223-244`);
  - `policy_reset` (renders the attention video in a background thread);
  - `policy_prompt` → `_handle_prompt` loads demo `prompt_index` of `task_name` from the prompt dataset and calls
    `policy.prompt(...)` once (`:376-401`);
  - `policy_inference` → `_process_obs` → `policy.predict_action` → `_process_action`;
  - `export_recorded_data`.
- `_process_obs` (`:418-457`): for camera keys, if the input HxW isn't 224x224, applies
  `get_image_transform_with_border(in_res=(W,H), out_res=(224,224))` per frame (pad to a square with black, then
  `cv2.INTER_AREA`). Then moves channels to `(T,3,224,224)` float32 /255. Low-dim values pass through as float32.
- `_process_action` (`:459-472`): splits `(K, 10·n_robots)` into `action{i}_eef_pos (K,3)`,
  `action{i}_eef_rot_axis_angle (K,6)` (rot6d), and `action{i}_gripper_width (K,1)`. These are unnormalized and
  relative to the latest observation.
- `detached_inference_action_replay.py` replays demos open-loop, but it **hard-codes bimanual keys**
  (`gripper_left_*` and `gripper_right_*`, `:134-142`; `sides` chooses which to execute, `:155-159`). It will
  KeyError on the single-arm `firstreal-single` buffer unless patched. (details.md §3.3 plans to use it for TCP
  validation.)

### 1.6 Launching (what runs, and where)

On one host, typically:
1. Camera: `iphone wrist` (or `python -m real_env.peripherals.iphone wrist_right`).
2. Arm: `arx5 right` (Hydra `arx5_right`).
3. Spacemouse server: `robot_utils.teleop_utils.spacemouse.spacemouse_server` on `tcp://localhost:15557`.
   `SpacemouseClient.__init__` immediately peeks `spacemouse_state` and indexes `[0]`, so it **crashes if no server
   is running** (`robot_utils/teleop_utils/spacemouse/spacemouse_client.py:13-35`).
4. Inference:
   `python detached_inference.py -i ckpt --dataset-path prompts.zarr.zip [--device cuda]`.
5. Task: `iphumi_arx5_bimanual <name> project_name=... task_name="..." prompt_index=0`.

`RE/shell_scripts/kill_all_processes.sh` lists the ports.

---

## 2. The exact observation dict per step (run_001)

### 2.1 What the model was trained on (from `RUN`)

| model key | shape | horizon | down_sample (at 60 Hz) | effective spacing |
|---|---|---|---|---|
| `camera_right_main_rgb` | (3,224,224) | 2 | 3 | 50 ms (t-50 ms, t) |
| `camera_right_ultrawide_rgb` | (3,224,224) | 2 | 1, applied at the native **10 Hz** ultrawide rate | 100 ms; the two most recent UW frames at or before t |
| `gripper_right_eef_pos` | (3) | 2 | 3 | 50 ms, relative to the latest pose |
| `gripper_right_eef_rot_axis_angle` | raw (3) → (6) rot6d | 2 | 3 | relative to the latest pose |
| `gripper_right_gripper_width` | (1) | 2 | 3 | absolute meters (0 = closed) |
| `gripper_right_eef_rot_axis_angle_wrt_start` | (6) | 2 | 3 | `ignore_by_policy: true`, `in_replay_buffer: false` |
| action | (16, 10) = pos3 + rot6d + width | 16 | 3 | **20 Hz → 0.8 s horizon**, relative to the current pose |

- Sources: `RUN:76-176`, `pose_repr` relative/relative (`RUN:178-180`).
- Upsampled-key handling (UW horizon counted at its own 10 Hz rate): `BP/train_network/common/sampler.py:672-705`.
- Relative obs/action w.r.t. `pose_mat[-1]`: `BP/train_network/dataset/umi_task_dataset.py:419-440`.
- Image transforms inside the model: eval `CenterCrop ratio 0.95`; train `RandomCrop 0.95` + `ColorJitter(0.3, 0.4,
  0.5, 0.08)` (`RUN:181-197`).
- Sim env-runner `exec_action_horizon: 12` (`RUN:219`).
- DDIM, 16 inference steps (`RUN:249`).
- Prompting: `use_prompting: true`, `prompt_sequence_length 512`, pair mode, `prompt_relative_to: last`.

### 2.2 What the task builds (`iPhumiARX5Task.get_observations`, `iphumi_arx5_task.py:24-82`)

**Images.** `get_downsampled_camera_images(wrist_client, image_history_len=2, agent_update_freq_hz,
cam_obs_freq_hz)` (`RE/utils/umi_utils.py:9-76`):
- For each camera, `frame_num = round((T-1)*fps/target_freq)+1` (`:42`). It fetches `max(frame_num)` frames (`:49-50`).
- It anchors on the newest timestamp and steps back `1/target_freq` per frame, choosing the nearest frame (`:55-74`).
- It raises if a chosen frame is more than `2/fps` from its target (`:66-72`).
- Output: `{main: (2,H,W,3), ultrawide: (2,H,W,3)}` RGB uint8, plus per-camera timestamps.

**Robot state.** `robot_state_timestamps = [last_ts - i/agent_hz for i in (1, 0)]`, where `last_ts =
time.monotonic()` when `match_camera_latency: false` (the default), otherwise the newest main-camera timestamp
(`:37-44`). Then `get_eef_xyz_wxyz(ts)` and `get_joint_pos(ts)` are read from the controller client (`:46-52`).

The resulting dict:
```
robot0_eef_xyz_wxyz            (2,7) float64  absolute TCP poses from robot FK
robot0_gripper_width           (2,1) float64  from the controller's joint_state (gripper encoder)
robot0_main_camera             (2,720,960,3) uint8 RGB  (no crop, no mask)
robot0_ultrawide_camera        (2,720,960,3) uint8 RGB
timestamps                     (2,) float64   main-camera timestamps (latency-corrected)
task_name                      str            (broken: self.task_name not set)
robot0_eef_xyz_wxyz_wrt_start  (2,7)          relative to the episode-start pose (start = eef_poses[0] when policy mode begins)
```

### 2.3 What `PolicyAgent.predict_actions` sends (`policy_agent.py:103-253`)

1. Save `obs_last_eef = observations["robot0_eef_xyz_wxyz"][-1]` (**measured FK**) as the reference for later
   (`:135-138`).
2. **`action_as_proprio: true`:** replace `robot0_eef_xyz_wxyz` and `robot0_gripper_width` with
   `history_actions["action0_*"]` (`:141-145`).
   - The history is the last 2 entries of `policy_history_actions`. Those are the first `exec_h+1` actions of the
     previous chunk (task `umi_arx5_task.py:314-317`), so the 2 entries used are actions `exec_h-1` and `exec_h`.
   - Before the first policy chunk, the history is the spacemouse command padded by repetition, so the relative motion
     is the identity (`umi_arx5_task.py` `get_history_actions`).
3. `keep_one_step_dummy_proprio: true`, so nothing is trimmed (`:151-155`). For each pose, `get_relative_pose(pose_t,
   poses[-1])`, then `xyz_wxyz_to_pose_9d` → `robot0_eef_pos (2,3)` float32 (last row = 0) and
   `robot0_eef_rot_axis_angle (2,6)` float32 rot6d (last row = identity `[1,0,0,0,1,0]`) (`:157-166`). The rot6d
   convention (first two rows of R) matches training: `robot_utils/pose_utils.py:303-306` vs
   `BP/common/pose_util.py:95-98`.
4. `robot0_gripper_width (2,1)` float32 = **commanded** widths (`:174-177`).
5. `robot0_eef_rot_axis_angle_wrt_start (2,6)` (`:191-197`), ignored by the policy.
6. Camera keys and `task_name` pass through unchanged as uint8 THWC (`:200-202`). Add `episode_idx = 0` (`:204`).
7. `request_with_data("policy_inference", ..., timeout_s=5.0)` (`:206-208`). **A hard 5 s timeout.** The Pi-CPU plan
   (details.md: 3–10 s per call) would fail here.
8. Reply → for each robot, `pose_9d_to_xyz_wxyz(pos ‖ rot6d)` → `get_absolute_pose(obs_last_eef, rel_k)` → absolute
   `(16,7)` targets. Gripper widths are used as-is (absolute) (`:217-245`).
9. `timestamps = obs_last_timestamp + arange(16)/agent_hz`, where `obs_last_timestamp` is the newest main-camera
   timestamp (`:247-251`).

### 2.4 Action execution (`UmiARX5Task.run_policy_control`, `umi_arx5_task.py:270-335`; bimanual equivalent `:557-619`)

- If the previous mode was SPACEMOUSE or single-step, or `pause_when_policy_inference` is set, the timestamps are
  re-based to `time.monotonic() + k/agent_hz` (`:279-290`).
- In pause or single-step mode, only the first `action_execution_horizon` actions are kept (`:293-298`). Otherwise
  all 16 are sent.
- Arm: `schedule_eef_traj(abs_poses, timestamps + lookahead_s (0.1), absolute)`. Gripper:
  `schedule_joint_traj(widths, timestamps)` (`:299-308`).
- The controller interpolator trims and merges the new trajectory. ARX5 also shifts it by a dynamically matched
  latency (`arx5_cartesian.py:466-496`).
- Wait until `timestamps[exec_h-1] - 0.02` (continuous mode), or `+ camera_latency_s` (pause mode) (`:319-333`).
  Then the next observation is taken.
- **Rate consistency:**
  - Training action spacing is 3/60 s = **50 ms (20 Hz)**. Deploying with the default `agent_update_freq_hz: 10`
    executes each 50 ms step over 100 ms, so the robot runs at half speed.
  - It also spaces main images 100 ms apart (trained: 50 ms), UW images 200 ms apart (trained: 100 ms), and commanded
    proprio 100 ms apart.
  - Matching training needs `agent_update_freq_hz: 20`. The bimanual action-replay config uses 20
    (`iphumi_arx5_bimanual_action_replay.yaml:12`); UMI configs deliberately run slower (8–12) to slow tasks down.
  - `UmiARX5Task` asserts `fps % int(agent_update_freq_hz) == 0` when `image_history_len > 1` (`:117-120`). At 20 Hz
    with the iPhone config's `fps: 30`, **the assertion fails**, so the camera config should say 60 fps.
- `exec_h` 8 (agent default) vs 12 (the training sim runner): 8×50 ms = 0.4 s per chunk.

### 2.5 Timestamps and latency

- iPhone: the timestamp is taken **before** the blocking read of the latest message, already minus
  `camera_latency_s` (`iphone.py:90`). `BaseCamera.run` then subtracts `camera_latency_s` **again** (`:221`). With
  0.08 s, frames are effectively stamped 0.16 s early.
  - With `match_camera_latency: false` this only affects action timestamps in continuous mode (they start further in
    the past, so the controller trims more of the chunk).
- The new peripheral should return a true capture timestamp in the monotonic domain and **not** subtract latency
  itself. Set `camera_latency_s` to the measured residual (0 if the timestamp is a true capture time).
- The robot-state alignment hack: robot state is sampled at "now", not at the image time (`iphumi_ur5_task.py:39-48`:
  "if doesn't include camera latency, policy works much better").
- Existing latency tool: `RE/../scripts/calibrate_camera_latency.py` (QR-timecode based). It imports
  `IPHONE_INPUT_STREAMS_CONFIG` from `iphone.py`, which no longer exists, so it's broken as-is.
  `IP/deployment/test_iphone_usb_streaming.py --eval_latency` is the working QR latency tool for the phone; reuse its
  QR utilities (`RE/utils/qr_code_util.py`).

---

## 3. Training vs deployment image processing: mismatch analysis

**Training** (`IP/demonstration_processing/build_umi_dataset_stages/gen_replay_buffer_zarr.py`):
- Decode each video frame with PyAV to `rgb24` (`:597-602`).
- `get_image_transform_with_border(in_res=(iw,ih), out_res=(224,224), mode="rgb")` (`:579-583`): pad symmetrically to
  `max(iw,ih)` square with black (4:3 gives black bars top and bottom), then `cv2.resize(..., INTER_AREA)`
  (`IP/common/cv_util.py:5-34`). `out_res_rgb` defaults to 224 (`:384-385`).
- Stored as JPEG-XL level 99 (`:426`).
- UW is stored at its native ~10 Hz with upsample indices (`build_umi_dataset_iphone.yaml:13`).

**Raw data facts** (checked on `data/raw/2026-10-01/...firstreal_demonstration_left/*.json`):
- Device iPhone 16 Pro (`iPhone17,1`).
- Main 1920x1440, K f=1406.3, c=(953.3, 723.2) → HFOV 68.6°, VFOV 54.2°, DFOV ≈80.9°, about 59 Hz (dt 17 ms).
- Ultrawide **640x480**, K f=259.0, c=(320.3, 239.7) → HFOV 102°, VFOV 86°, DFOV ≈114° (pinhole intrinsics from
  ARKit, so presumably already distortion-corrected), **10.0 Hz**.
- Pose and depth at about 59 Hz.

**Deployment:** the iPhone app streams 960x720 RGB JPEG (§4), and the inference server applies the **same**
`get_image_transform_with_border` (`BP/common/cv_util.py` is identical to `IP/common/cv_util.py`; only `depth2xyzmap` is
missing) → 224x224, INTER_AREA. Then the model's CenterCrop 0.95.

| aspect | training | deployment (iPhone) | risk |
|---|---|---|---|
| geometry | 4:3 → pad square → 224 | 4:3 → pad square → 224 (server) | none, **as long as the task doesn't crop**. The `umi_*` `get_observations` center-crop to 256x256 (`umi_arx5_task.py:257-261`, `umi_ur5_task.py:273-277`) and draw a UMI gripper mask, which would be a large mismatch. The `iphumi_*` overrides skip both. |
| color order | rgb24 | BGR from `imdecode` → BaseCamera BGR→RGB (`base_camera.py:226`) → RGB | none |
| main source res | 1920x1440 H.264 video | 1920x1440 → CoreImage 0.5x scale → 960x720 JPEG q0.9 | low (sharpness or compression differences; ColorJitter covers color) |
| ultrawide source | ARKit 640x480 private buffer, 10 Hz | AVCapture 1920x1440 → 960x720 JPEG, streamed at the main rate | low to medium: different pipeline (ARKit vs AVCaptureMultiCam). FOV and distortion correction should match but that's unverified. Deployment UW frames are up to 100 ms fresher than in training (training takes the most recent 10 Hz frame). |
| focus | main continuous AF; UW locked at ARKit's lens position | main continuous AF (`DeploymentViewController.swift:458-460`); UW locked at the stored `arKitUltrawideLensPosition` (`:397-406`) | matched by design |
| temporal spacing | main 50 ms, UW 100 ms | depends on `agent_update_freq_hz` (§2.4) | **high if left at 10 Hz** |

**For the Jetson rig** (the main risk moves here): the images must reproduce iPhone geometry after pad-to-square.
- Output 4:3 frames (960x720 recommended), **rectified/remapped to pinhole intrinsics equal to the iPhone's**:
  - main: f≈703.2, c≈(476.6, 361.6) at 960x720, HFOV 68.6°;
  - ultrawide: f≈388.5, c≈(480.4, 359.5) at 960x720, HFOV 102°.
- A ~75° main lens must be cropped or undistorted down to 68.6° HFOV. A ~120° ultrawide (probably fisheye-ish M12)
  needs a fisheye → pinhole remap to 102° HFOV. Extreme corners beyond the pinhole FOV are dropped.
- Image "up" must match the iPhone's landscape sensor orientation (the app sets no rotation).
- The optical centers must sit where the iPhone main and UW cameras sat relative to the fingers. TCP is
  `I_T_TCP` = [[1,0,0,0], [0,-0.966,0.259,-0.0644], [0,-0.259,-0.966,-0.2114]] in the ARKit camera frame: 21.1 cm
  ahead of and 6.4 cm below the main camera, tilted 15°. See
  `IP/demonstration_processing/calibration/gripper_calibration.yaml:7-15`; the iPhone 16 Pro entry is copied from
  the 15 Pro.
- Fingers and AR tags should appear in the same image positions as in training (the UW must see both finger tags).
- If a lens can't cover the FOV, retraining at the new FOV is the alternative.

---

## 4. What the iPhone deployment stream provides

**iOS side** (`IOS/Deployment/DeploymentViewController.swift`):
- `AVCaptureMultiCamSession` with `.builtInUltraWideCamera` + `.builtInWideAngleCamera`, or with
  `.builtInLiDARDepthCamera` if depth is enabled (`:83`, `:332-372`).
- Resolution modes (`:121-134`, `:290-300`): High = capture 1920x1440; Low = 640x480. **Both are scaled by 0.5
  before streaming**, giving 960x720 or 320x240.
  - The default setting index is 0 = **Low** (`:290`), so the phone must be switched to High to match
    `iphone_wrist.yaml` (960x720). Otherwise `BaseCamera`'s shape assertion fails.
- FPS: 30 or 60. Default index 1 = **60** (`:302-307`). `iphone_wrist.yaml` says 30. Only `fps` from the config is
  used downstream (frame selection and the divisibility assert), so set the config to match the phone.
- Format selection: the first format with `isMultiCamSupported`, no HDR, the exact dimensions, and support for the
  fps (`:381-396`, `:439-452`).
- Focus: main `.continuousAutoFocus` (`:458-460`). UW locked at `arKitUltrawideLensPosition` (or continuous AF if the
  setting is on) (`:397-406`).
- **Not set explicitly** (so iOS defaults apply): exposure (continuous auto), white balance (auto), video
  stabilization (not enabled on `AVCaptureVideoDataOutput` connections), `isGeometricDistortionCorrectionEnabled`
  (iOS default is on where supported, which matters for the UW; unverified), zoom 1.0, orientation (sensor-native
  landscape, `videoRotationAngle = 0` on preview only).
- Encoding: `CIImage` → scale 0.5 → `CGImage` → `UIImage.jpegData(compressionQuality: 0.9)` (`:32-43`).
  - USB: one JPEG per message via `USBManager` (peertalk/usbmux) on ports 5555 main / 5556 UW / 5557 depth
    (`:114-116`, `:513-520`, `:556-562`).
  - Ethernet: base64 JPEG over Socket.IO to `hostIP` (default 192.168.123.18) on the same ports (`:101-108`,
    `:521-531`).
- Depth (optional): float32 320x240 at 30 Hz, raw bytes (`:603-623`).
- **No pose and no IMU are streamed** (also stated in `iPhUMI/docs/usage.md` "Supported streaming modalities": "We
  currently do not support streaming iPhone pose").

**Python side:**
- `RE/peripherals/iphone.py` supports **USB/peertalk only** (`pypeertalk.PeerTalkClient(device, peertalk_port)`,
  `:43-73`).
- Per iteration: `timestamp = monotonic - latency`; for each camera in turn, `client.get_latest_message(100)` →
  `cv2.imdecode` (BGR) → assert the shape (`:87-154`).
- On a decode error it `continue`s, which **drops that camera from this packet** (`:125-133`). BaseCamera then
  publishes fewer camera frames than timestamps, which can misalign the client's index-based selection. The new
  peripheral must always return every camera.
- The Ethernet receiver (Flask-SocketIO) exists only in `IP/deployment/common/iphone_camera.py:246-300` (test scripts),
  not in real-env.
- Measured latency is about 100 ms (docs). The config uses `camera_latency_s: 0.08`.

---

## 5. Is phone or VIO pose used at deployment? **No.**

- `iphone.py` publishes only images (and optionally depth). No pose topic exists.
- Every task reads `eef_xyz_wxyz` and `gripper_width` from the controller client (robot FK and gripper encoder), e.g.
  `iphumi_arx5_task.py:46-52`, `iphumi_arx5_bimanual_task.py:468-472`.
- `PolicyAgent` uses measured FK only as the absolute reference (`policy_agent.py:135-138`, `:224-228`). The proprio
  fed to the model is the **commanded action history** (`action_as_proprio: true`, `:141-145`).
- The wrt-start obs is also FK-derived and is ignored by run_001.
- **So the IMU and VIO on the Jetson rig play no part in deploying run_001.** They matter only if the rig is also used
  to **collect** training data (replacing ARKit poses: 60 Hz TCP-frame poses, the gripper width from AR tags in the
  UW, and a 10 Hz UW). That's a separate contract, tied to `gen_replay_buffer_zarr.py` and the dataset plan.
- Gripper width at deployment comes from the gripper encoder (`joint_state`). In training it's AR-tag distance minus
  the calibrated minimum, in meters, 0 = closed (`IP/demonstration_processing/utils/gripper_util.py:19-60`,
  `interpolation_util.py:35-49`). The Franka or custom gripper width must use the same zero point and scale
  (finger-gap meters, 0 when closed). Check the max against `*_gripper_range.json` (not on the laptop).

---

## 6. The minimum contract for a drop-in `JetsonCamera` / `SensorRig` peripheral

**Class:** `real_env/peripherals/jetson_camera.py: class JetsonCamera(BaseCamera)`, with a
`run_jetson_camera()` launcher that copies `run_iphone()` (compose Hydra, `cfg.config_str = json.dumps(...)`,
instantiate, `.run()`). Register a console script in `pyproject.toml`.

**Implement:**
```python
def get_images_bgr_hwc_and_timestamp(self) -> tuple[dict[str, np.ndarray], float]:
    # returns {"main": (720,960,3) uint8 BGR, "ultrawide": (720,960,3) uint8 BGR}, capture_time_monotonic
```
- Keys must be exactly `main` and `ultrawide` (tasks hardcode `images_dict["main"]` / `["ultrawide"]`, and
  `cam_obs_freq_hz` keys use these names).
- **Return every configured camera on every call.** Repeat the last frame on a dropout (as `WebCam` does); never skip
  a key.
- Shapes must equal `camera_configs[...]` exactly (asserted). Channel order is BGR, because BaseCamera converts to RGB.
  If the capture is already RGB, either flip it or override. Don't double-convert.
- Timestamp in the `time.monotonic()` (CLOCK_MONOTONIC) domain, e.g. the Argus/V4L2 buffer timestamp converted to
  monotonic, at or near exposure time. **Don't subtract latency in the subclass** (unlike `iphone.py:90`). One
  timestamp covers both cameras, so sync the two CSI sensors in hardware or pair frames by nearest timestamp.
- Run the loop at least at 60 Hz (or set config `fps` to the real rate): the selection tolerance is `2/fps` (33 ms at
  60 fps), and `fps % agent_hz == 0` is asserted.
- Rectify and crop to iPhone-equivalent intrinsics (§3) **inside** the peripheral, so frames are drop-in identical in
  geometry.
- Optionally override `reset()`.

**Config** (`RE/configs/peripherals/jetson_wrist.yaml`):
```yaml
_target_: real_env.peripherals.jetson_camera.JetsonCamera
name: right_wrist_camera_0          # robologger naming convention; must equal the task's logger_endpoints key
server_endpoint: tcp://localhost:16700   # must equal the task's wrist_camera_endpoint
logger_endpoint: tcp://localhost:55554   # must equal task logger_endpoints[right_wrist_camera_0]
camera_latency_s: 0.0               # residual after true capture timestamps; measure with a QR timecode
display_resolution_WH: null         # headless
camera_configs:
  main:      {type: rgb, width: 960, height: 720, fps: 60, sensor_id: 0, ...device-specific keys}
  ultrawide: {type: rgb, width: 960, height: 720, fps: 60, sensor_id: 1, ...}
```
- Device-specific keys (sensor id, calibration file, undistortion maps) can live inside `camera_configs`; extra keys
  are ignored by BaseCamera and VideoLogger.
- Shared-memory ring: 0.5 GB per camera with 1 s retention. 960x720x3 at 60 fps = 124 MB/s, which fits.
  1920x1440 at 60 fps is about 500 MB/s, which would overflow 0.5 GB, so stay at 960x720.
- The task process must run on the **same host** as the camera server (shm).
- **VideoLogger codec:** the default is `h264_nvenc`. The **Jetson Orin Nano has no NVENC**, so recording episodes
  (`c` → `start_recording`) would make ffmpeg fail silently (stderr goes to DEVNULL). Either subclass or patch to pass
  `codec="libx264"`, or don't list the camera in `logger_endpoints`.

**Task side** (new `umi_franka_task.py`):
- Copy `iPhumiARX5Task.get_observations` (images from `get_downsampled_camera_images`, FK and width from the Franka
  client, `crop_image: null`, `draw_gripper_mask: false`).
- Set `self.task_name`.
- Add prompting: `policy_agent.send_prompt_request(task_name, prompt_index)` before the first `c`, copying
  `iphumi_arx5_bimanual_task.py:395-410`. `task_name` must be one of the task names in the prompt dataset passed to
  `--dataset-path`.
- `policy_agent.agent_update_freq_hz: 20` for training-rate execution.

**Controller side:** `FrankaCartesian(BaseController)`, CARTESIAN_JOINT mode, as described in §1.2.
- TCP = the iPhUMI TCP (z-forward convention like ARX5's), reached by setting libfranka `F_T_EE` or by converting in
  the controller.
- Publishes monotonic timestamps; accepts absolute-time trajectories; does its own interpolation and lookahead.
- `RobotCtrlLogger`s `right_arm` / `right_end_effector`.

---

## 7. Test and mock hooks; dependencies

**What exists:** essentially nothing for running without hardware.
- No mock camera, mock controller, or mock spacemouse. `grep` for mock/dummy/fake/sim in real-env finds only comments.
- Open-loop tools:
  - `detached_inference_action_replay.py` (needs a robot; bimanual-only keys);
  - `IP/deployment/test_iphone_usb_streaming.py` / `..._ethernet_...` (camera only; QR latency);
  - the training-time `UmiPromptingRunner` env runner (offline replay of dataset observations; it produces the
    `results/.../videos`).

**Cheap mocks for an end-to-end test without hardware (all small):**
1. `FakeCamera(BaseCamera)`: plays back 4:3 frames from a raw demo's `*_rgb.mp4` / `*_ultrawidergb.mp4`, resized to
   960x720, stamped with `time.monotonic()`. This also gives a ground-truth check: the server's 224 image should
   equal the replay-buffer image.
2. `FakeCartesianController(BaseController)`, CARTESIAN_JOINT: integrates the scheduled trajectory, using
   `real_env.common.interpolator.PoseTrajInterpolator` / `JointTrajInterpolator` the way ARX5 does, and publishes the
   resulting pose as "FK".
3. A fake spacemouse server publishing 7 zeros (float64 bytes) on topic `spacemouse_state` at
   `tcp://localhost:15557`. `SpacemouseClient` crashes without one.
4. `detached_inference.py --device cpu|cuda` with the run_001 checkpoint and `--dataset-path` set to the
   firstreal-single zarr.

That exercises every RMQ hop, timestamping, frame selection, preprocessing, prompting, and relative→absolute
conversion.

**Python deps to import and run real-env** (`RE/../env.yaml`):
- python 3.10, numpy 2.2.6, scipy, zarr 2.18.3, hydra-core 1.3.2, omegaconf 2.3.0, opencv-python 4.12, msgpack;
- robotmq 0.1.14, robologger 0.1.7 (pulls in loguru, used by `policy_agent.py`), robot-utils 0.1.4 (pose/time/image
  utils, spacemouse client), transforms3d (spacemouse agent), pynput (task keyboard; needs X/uinput access), the
  ffmpeg binary (VideoLogger);
- optional: pypeertalk (iPhone only), arx5-interface (ARX5 only), ur-rtde (UR5), open3d (pose visualizer, lazily
  imported), spnav (spacemouse server), flask-socketio (Ethernet test).
- Import chain for a task: `base_task` → `policy_agent` (robotmq, loguru, robot_utils), `spacemouse_agent`
  (robot_utils.teleop_utils, transforms3d), `base_controller_client`, `base_camera` (cv2, zarr, robotmq, robot_utils,
  robologger), robologger `MainLogger`. Controllers import their SDKs at module top (e.g. `arx5_interface`), so a
  Franka module should import franky/panda-py at module top only in its own file.
- **aarch64:** robotmq 0.1.14 and pypeertalk 0.1.2 have `cp310 manylinux_2_28_aarch64` wheels on PyPI (verified with
  `pip download --platform manylinux_2_28_aarch64 --python-version 3.10`). This contradicts details.md's "may need
  building from source". robologger and robot-utils are pure Python.
- Inference server deps (behavior_prompting): torch, hydra, dill, diffusers, timm, transformers, click, imageio,
  robotmq.

---

## 8. Issues found (ranked)

1. **No single-arm prompting path.** `UmiARX5Task` / `iPhumiARX5Task` never call `send_prompt_request`. A
   prompting policy would run unprompted (probably an error or garbage output). Fix: add it to the new Franka task.
2. **`iPhumiARX5Task.get_observations` uses the unset `self.task_name`** (`iphumi_arx5_task.py:64`), and its launcher
   wants `iphumi_arx5_<name>.yaml`, which doesn't exist (`:90`).
3. **Rate mismatch:** the default `agent_update_freq_hz: 10` vs the trained 20 Hz (actions, proprio, main images
   spaced 50 ms; UW 100 ms). At 20 Hz the iPhone config's `fps: 30` trips the `fps % 20` assert; use 60 fps.
4. **Double latency subtraction** in the iPhone path (`iphone.py:90` + `base_camera.py:221`).
5. **Hard 5 s inference timeout** (`policy_agent.py:207`). Fine on a Jetson GPU, fatal on Pi CPU.
6. **Cropping trap:** the `umi_*` base `get_observations` center-crops to 256² and draws a UMI mask (both defaults in
   `umi_*.yaml`). Training pads instead. Only the `iphumi_*` overrides are correct.
7. **The action-replay server is bimanual-only** (needs `gripper_left_*`).
8. **The VideoLogger defaults to `h264_nvenc`**, which the Orin Nano doesn't have.
9. `calibrate_camera_latency.py` imports a symbol that no longer exists.
10. Camera packet misalignment if a subclass skips a camera (iPhone decode-error path); a peek race in
    `BaseCameraClient`.
11. The iOS app defaults to Low (320x240) streaming, so it has to be switched to High to match the configs.

---

## Open questions

- Which prompt `task_name`(s) and `prompt_index` should be used for block_raise? They must match a task name in the
  firstreal-single zarr (`meta/task_names`).
- What are the gripper width scale and range in training (`*_gripper_range.json` max-min)? Does the Franka or custom
  gripper width match 0 = closed in finger-gap meters?
- Does the iPhone ultrawide in ARKit (training) have the same FOV and distortion correction as the AVCapture
  ultrawide (deployment)? This is unverified. For the Jetson rig, the UW target is ARKit's pinhole K (f=259 @ 640x480).
- Can the Jetson's ~75° main and ~120° ultrawide lenses (plus their mounting) reproduce the iPhone 16 Pro main (68.6°
  HFOV) and UW (102° HFOV pinhole) views and the camera-to-TCP offset? If not, retrain on rig-collected data.
- Should the deployment use `agent_update_freq_hz: 20` (training rate) or deliberately slower? And `exec_h` 8 or 12?
- Hardware sync between the two CSI cameras (a common timestamp is assumed per packet).
