# Deployment: the rig as the policy's wrist camera

The rig replaces the iPhone at deployment through a real-env camera peripheral, `RigCamera`. It publishes exactly what
the iPhone peripheral published, so the task, `PolicyAgent` and `detached_inference.py` stay unchanged and the policy
sees the same images it was trained on.

| piece | file |
|---|---|
| logic (capture, pairing, remap, timestamps, stats), no real-env imports | `sensor_rig/sensor_rig/deploy/frame_source.py` (`RigFrameSource`) |
| real-env adapter (`RigCamera(BaseCamera)`, launcher `run_rig_camera`) | `behavior_prompting/deps/real-env/real_env/peripherals/rig_camera.py` |
| hardware config | `real_env/configs/peripherals/rig_wrist_right.yaml` |
| no-hardware config | `real_env/configs/peripherals/rig_wrist_mock.yaml` |
| console script | `rig_camera` (one line in real-env `pyproject.toml`) |

## 1. Running it on the Jetson

One-time setup, in the real-env Python environment on the Jetson:

```bash
pip install -e /path/to/Capstone/sensor_rig                       # RigCamera imports sensor_rig
pip install -e /path/to/Capstone/behavior_prompting/deps/real-env  # re-run so the new rig_camera script is registered
```

The camera server and the task have to run on the **same host**, because robotmq passes images through POSIX shared
memory.

1. Calibrate the rig (`docs/calibration.md`) and check it with `python -m sensor_rig rig-validate <rig.yaml>`. There
   should be no placeholder warnings and coverage should be at least 0.98 for both virtual cameras. Use the same
   rig.yaml (same `argus` exposure/white-balance block) that recorded any rig training data.
2. Point the config at it: edit `rig_config_path` in `rig_wrist_right.yaml` (absolute path), or override it on the
   command line.
3. Start the camera server instead of `iphone wrist_right`:

   ```bash
   rig_camera wrist_right rig_config_path=/home/jetson/rig/rig_calibrated.yaml
   # equivalently: python -m real_env.peripherals.rig_camera wrist_right ...
   ```

   The first argument picks `configs/peripherals/rig_<name>.yaml`. Everything after it is a Hydra override.
4. Start the arm, the spacemouse server, inference and the task exactly as for the iPhone. The name
   (`right_wrist_camera_0`), the server endpoint (`tcp://localhost:16702`) and the logger endpoint
   (`tcp://localhost:55556`) are the same as `iphone_wrist_right.yaml`, so the task config doesn't change. Do set
   `policy_agent.agent_update_freq_hz: 20` in the task (see E3 below).

Every `stats_period_s` (5 s by default) it prints a status line like this:

```
rig: packets 300 fps main 30.0 uw 29.9 | skew ms p50 0.3 p95 16.1 max 16.7 | drops main 0 uw 0 | repeats 0 | remap ms main p50 1.57 p99 2.35
```

A RESET command from the client prints the full stats as JSON and clears them.

### Jetson checklist (hardware, not yet verified)

- [ ] Both Argus streams run at the rig.yaml fps (`fps main`, `fps uw` in the status line). `drops` stays near 0.
- [ ] Remap time on the Orin Nano: main p99 ≤ 4 ms at 1640x1232 → 960x720. Try `num_cv_threads: 2` if it's slow.
      Upgrade path if it's still too slow: VPI/CUDA remap.
- [ ] The publish loop holds 60 Hz (BaseCamera's `FPS:` line) while inference runs on the same board.
- [ ] Skew p95 ≤ ½ ultrawide period (≈ 17 ms at 30 fps). `repeats` stays 0.
- [ ] Measure the residual latency with a QR/LED timecode (see E9). Put it in rig.yaml `cameras.*.time.offset_prior_ns`,
      **not** in `camera_latency_s`.
- [ ] Look at one frame of each camera (fingers and tags in the same image positions as in iPhone training images).

## 2. Mock mode (no hardware)

```bash
rig_camera wrist_mock                                                   # synthetic moving checkerboard, rig_default.yaml
rig_camera wrist_mock backend_opts.episode_dir=/data/rig_raw/2026-10-20/<NAME>   # replay a raw rig episode
```

- `backend: mock` uses `sensor_rig.drivers.MockCamera`. With an `episode_dir` it replays that raw episode's JPEGs
  (`MockCamera.replay`, looping, with stamps re-based to `time.monotonic_ns()`). Without one, it generates a synthetic
  pattern at each physical camera's resolution.
- `rig_config_path: null` means: use `<episode_dir>/rig.yaml` (the snapshot recorded with the episode), or
  `rig_default.yaml` when there's no episode.
- `publish_mode: fixed_rate` publishes at 60 Hz whatever the replayed rate is.
- Without robotmq/robologger installed (e.g. on the laptop), the launcher can be run with the test stubs:
  `PYTHONPATH=sensor_rig/tests:behavior_prompting/deps/real-env python -c "import realenv_stubs; realenv_stubs.install(); import sys; sys.argv=['rig_camera','wrist_mock']; from real_env.peripherals.rig_camera import run_rig_camera; run_rig_camera()"`.
  Measured on the laptop: a steady 60.4 Hz publish rate, main remap p50 1.6 ms.
- Limitation: when you replay with `restamp: true`, each camera is re-based to its own first frame. The recorded offset
  between the main and the ultrawide is lost, so skew numbers in replays aren't meaningful. The offline D10d test
  uses `preserve_timestamps: true, realtime: false` with `sync: lockstep` instead, which is exact.

## 3. The contract (same as the iPhone peripheral, R3 §6)

| item | value |
|---|---|
| keys | exactly `{"main", "ultrawide"}`, on **every** call (no partial packets) |
| frames | `(720, 960, 3)` uint8 BGR, C-contiguous. BaseCamera converts them to RGB before publishing |
| geometry | virtual iPhone 16 Pro cameras at the deployment scale (half-pixel scaling convention): main f=703.1714, c=(476.3752, 361.3600); ultrawide f=388.5231, c=(480.6903, 359.7618), both pinhole with no distortion. Built by `rig.remap(name, (960, 720))`, the **same** `RemapTable` code the export uses at 1920x1440 / 640x480 |
| timestamp | one float, CLOCK_MONOTONIC seconds = capture time of the **main** frame: `rig.capture_time_ns("main", frame.t_ns) / 1e9` |
| name / endpoints | `right_wrist_camera_0`, `tcp://localhost:16702`, logger `tcp://localhost:55556` |
| fps (config) | 60 for both cameras, so `fps % 20 == 0` at the trained 20 Hz and the frame-selection tolerance is 2/fps = 33 ms |
| `camera_latency_s` | `0.0` |
| `display_resolution_WH` | `null` (headless) |
| video logger codec | `libx264`. RigCamera sets `video_logger.codec` because the Orin Nano has no NVENC |

**Pairing.** The two CSI cameras free-run and aren't hardware-synchronised. The main camera drives the loop: there is
one packet per new main frame. The ultrawide is the frame among the last `pairing.history` (4) whose capture time is
nearest to the main's capture time (`ultrawide: nearest`). The worst case is half an ultrawide period, ≤ 17 ms at 30 fps, well under the
policy's 100 ms ultrawide spacing. If `|t_main − t_uw|` > `max_skew_s` (34 ms), a warning is logged (rate-limited).
If it's > `stale_after_s` (0.25 s), the pair is refused. Each ultrawide frame is remapped once (cached by its seq).

**Failures.** On a main timeout (`frame_timeout_s`, 0.5 s) or a stale ultrawide, the previous packet is served again
with the same timestamp. After `max_consecutive_timeouts` (10) failures in a row it raises `RuntimeError`, which
stops the server. A frozen camera is never served silently for long. At startup it waits up to 5 s for the first
pair.

**Publish modes.**
- `per_main_frame`: one packet per new main frame.
- `fixed_rate`: packets at `camera_configs.main.fps`, each holding the newest pair. Duplicate frames keep their own
  main timestamp, so the client's selection error is at most one physical frame period.
- `auto` (the default in `rig_wrist_right.yaml`): `fixed_rate` when the physical main fps in rig.yaml is below the
  configured 60, otherwise `per_main_frame`. The default rig's main is 1640x1232 at 30 fps (full FOV), so `auto`
  resolves to `fixed_rate` at 60 Hz.

No IMU is opened at deployment. Pose comes from robot FK (R3 §5).

## 4. Latency policy (applied exactly once)

- The backend's `Frame.t_ns` is the raw CLOCK_MONOTONIC stamp, exactly what the recorder writes to `data.csv`.
  Argus MONOTONIC_RAW is converted inside the backend.
- Raspberry Pi 5: `backend: picamera2` (in the peripheral config / `RigFrameSource(..., backend="picamera2")`) is passed
  through to `make_camera` unchanged and reads each camera's `picamera2:` block from rig.yaml. Its `t_ns` is the
  libcamera `SensorTimestamp` (CLOCK_MONOTONIC start of frame). Deploy with the same rig.yaml (sensor mode, locks,
  focus) that recorded the training data. See [raspberry_pi.md](raspberry_pi.md).
- `RigFrameSource` adds the per-camera rig.yaml offset (`cameras.<c>.time`: `offset_prior_ns` + Kalibr
  `timeshift_cam_imu`) through `rig.capture_time_ns`. Export uses the same function, so training and deployment
  timestamps mean the same thing.
- `RigCamera` returns that time unchanged and does **not** subtract latency itself (unlike `iphone.py:90`).
- The config sets `camera_latency_s: 0.0`, so BaseCamera's `timestamp - camera_latency_s` (`base_camera.py:221`) does
  nothing. If it is non-zero, RigCamera prints a warning and treats it as an extra measured residual.

This avoids the iPhone path's double subtraction (E4) without touching `base_camera.py` or `iphone.py`.

## 5. Verification (no hardware), `sensor_rig/tests/`

| test | what it checks |
|---|---|
| `test_frame_source.py` | nearest/latest pairing (live and lockstep), main capture timestamps, stale ultrawide and timeout handling, ultrawide remap cache, catch-up and drop stats, fixed-rate packets, output contract, coverage guard, the real `MockCamera` backend |
| `test_rig_camera_contract.py` | Hydra composes the configs as drop-ins for `iphone_wrist_right`; `get_images_bgr_hwc_and_timestamp` contract; `BaseCamera.run()` publishes one RGB frame per camera and then the main capture time; libx264; config validation |
| `test_deploy_equivalence.py` | D10d. Part 1: real iPhone frame → default rig (inverse remap) → export path (1920x1440 → iphumi pad+224) vs deploy path (960x720 → BGR→RGB → behavior_prompting pad+224). Part 2: a simulated raw episode from a real iPhone demo (WP-C sim, `sim_rig.yaml`, 90 frames) replayed by `MockCamera.replay` into `RigCamera`, compared with the export path, the original iPhone frames, and the exported H.264 video |

Measured (2026-10-15, demo `…21-39-22.626Z_49242…`, MAE in grey levels at 224x224, ECC translation in px):

| comparison | main | ultrawide |
|---|---|---|
| unit: deploy vs export path | 0.063 (max 5), ECC 0.0008 | 0.161 (max 11), ECC 0.0009 |
| unit: deploy vs original iPhone (export path vs original) | 0.119 (0.114) | 0.431 (0.406) |
| unit: 1 px shift at 960x720 (sensitivity check) | MAE 1.02, ECC 0.263 (expected 0.233) | |
| integration: deploy vs export path (30 / 15 frames) | 0.085, ECC max 0.0012 | 0.195, ECC max 0.0011 |
| integration: deploy vs original iPhone (export path vs original) | 0.562 (0.556) | 0.718 (0.711) |
| integration: deploy vs exported MP4 (export path vs MP4) | 2.200 (2.190) | 2.223 (2.204) |

Integration timestamps equal the simulator's true capture times to the nanosecond, and every packet pairs the
globally nearest ultrawide frame. The MP4 difference (~2.2) is H.264 compression and is the same for the export path.
The deployment path adds ≤ 0.02 on top of it. The tolerances (P2 §7) are: deploy vs export ≤ 0.5 MAE and ≤ 0.05 px
ECC; versus the original or the MP4, within the export path's own MAE + 1.0 and ≤ 4.0.

## 6. Upstream deployment issues (R3 §8): proposed fixes, **not applied, need approval**

None of these is needed for RigCamera to work. The only rig-local mitigations in place are `camera_latency_s: 0`
(latency applied once via `capture_time_ns`) and the libx264 override inside RigCamera. Nothing in
`behavior_prompting/` or real-env was changed beyond the new `rig_camera.py`, the two rig yaml files and one line in
real-env `pyproject.toml`.

| # | issue | proposed minimal fix (file → change) | status |
|---|---|---|---|
| E1 | No single-arm prompting: `UmiARX5Task` / `iPhumiARX5Task` never call `send_prompt_request` | The new Franka task copies `iphumi_arx5_bimanual_task.py:395-410` (`policy_agent.send_prompt_request(task_name, prompt_index)` in `start_episode`). Don't patch `UmiARX5Task` | not applied — needs approval |
| E2 | `iPhumiARX5Task` uses an unset `self.task_name`. Its launcher wants a non-existent `iphumi_arx5_<name>.yaml` | `RE/tasks/base_task.py` `__init__`: `self.task_name = task_name` (1 line). `iphumi_arx5_task.py:90`: compose `iphumi_arx5` when there's no argv (2 lines) | not applied — needs approval |
| E3 | `agent_update_freq_hz: 10` vs the trained 20 Hz, and `fps: 30` trips `fps % 20` | Task config override `policy_agent.agent_update_freq_hz: 20` (config only). The rig side is already done: fps 60 + `fixed_rate` | rig part done; task part not applied — needs approval |
| E4 | Double latency subtraction in the iPhone path (`iphone.py:90` + `base_camera.py:221`) | `RE/peripherals/iphone.py:90`: `timestamp = time.monotonic()`, then re-tune `iphone_wrist.yaml camera_latency_s` (the effective latency changes from 0.16 to 0.08 s) | not applied — needs approval (RigCamera avoids it by design) |
| E5 | Hard 5 s `predict_actions` RMQ timeout | `RE/agents/policy_agent.py`: new ctor arg `inference_timeout_s: float = 5.0` used at `:207`, plus a `policy_agent.yaml` entry. The first CUDA call on an Orin Nano can take more than 5 s | not applied — needs approval |
| E6 | `umi_*` `get_observations` center-crops to 256² and draws a UMI mask (training pads instead) | No code change. The new task must use the `iphumi_*` `get_observations` with `crop_image: null`, `draw_gripper_mask: false`, plus an assert | not applied — needs approval |
| E7 | Action-replay server is bimanual-only (KeyError on single-arm buffers) | `BP/deployment/detached_inference_action_replay.py:134-159`: derive the sides from the keys present | not applied — needs approval |
| E8 | VideoLogger defaults to `h264_nvenc` (no NVENC on the Orin Nano; ffmpeg fails silently) | Handled inside RigCamera (`video_logger.codec = "libx264"`). Optional upstream change: a `video_codec` arg on BaseCamera passed to VideoLogger. libx264 at 2×960x720@60 probably can't keep up on the CPU, so keep the camera out of `logger_endpoints` | rig part done; upstream part not applied — needs approval |
| E9 | `scripts/calibrate_camera_latency.py` imports the removed `IPHONE_INPUT_STREAMS_CONFIG` | Use `BaseCameraClient.camera_names` instead. For the rig, measure `offset_prior_ns` with an LED/QR timecode | not applied — needs approval |
| E10 | Packet misalignment if a subclass skips a camera; peek race in `BaseCameraClient` | Skipping: RigCamera never skips. Race: in `base_camera.py:351-391` peek the camera topics first and the timestamps last, then keep the last `min(len)` of each (≈6 lines) | not applied — needs approval |
| E11 | iOS app defaults to Low (320x240) streaming | N/A for the rig | — |
| E12 | VideoLogger sends ffmpeg stderr to DEVNULL, which hides encoder failures | Upstream robologger (not vendored). No workaround needed if the camera isn't in `logger_endpoints` | not applied — needs approval |

## 7. Known limitations / next steps

- `MockCamera.replay(..., restamp)` re-bases each camera separately (see §2). A shared-origin option in the mock
  would make live replays keep the recorded inter-camera offset.
- Argus timestamps are best-effort (appsink delivery time mapped to CLOCK_MONOTONIC, `offset_prior_ns` ≈ −30 ms
  placeholder). A libargus SOF-timestamp capture is the upgrade path (R4).
- Geometry is matched, photometry is not: IMX219 colour, noise, rolling shutter and auto-exposure differ from the
  iPhone. Use the same `argus` block for recording and deployment.
- A remap can't move the optical centre. The rig lenses must sit where the iPhone's main and ultrawide lenses were
  (within ~3 mm), or an iPhone-trained policy sees a shifted viewpoint (P2 §11.5, §13 R2).
