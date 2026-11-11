# `dataproc sim`: synthetic recordings with ground truth

`dataproc.sim` renders raw episodes from a known trajectory, so calibration, processing, SLAM and the
gripper-width stage can be tested without hardware. Each one is first rendered in dataproc's own format
([CONTRACT.md](CONTRACT.md), `dataproc.raw.v1`; it passes `RawEpisode.validate(decode_videos=True)` with zero
problems unless faults are injected on purpose), plus `truth.npz`.

**By default `dataproc sim` then writes it in the Pi's format** (`dataproc/pi_format.contract_to_pi`, what the Pi 5
puts in `/data/incoming`: docs/PI_ADAPTER.md) as `episode_NNNNNN`, with `truth.npz` carried over and listed in the
manifest, so it goes through `calibrate` → `run --slam oracle` → `package` exactly like real data. `--contract`
writes the contract episode itself instead (for intake/process tests).

```bash
export DATAPROC_ROOT=/tmp/sim_root                                 # a separate scratch root, never the real one
dataproc sim --out /tmp/sim_src --episodes 1 --calibration --fast  # episode_000000: a calibration
dataproc sim --out /tmp/sim_src --episodes 5 --fast                # episode_000001..000005: demonstrations
dataproc calibrate block_raise episode_000000 --source /tmp/sim_src --rig rig_v1_fast
dataproc run block_raise episode_000001 episode_000005 --source /tmp/sim_src --slam oracle
```

**Do not run `dataproc sim` against the real data on the Jetson** (`/data/incoming`, `~/data`): `sim` copies the
(nominal) rig yaml it rendered with into `rigs/` if that `<rig_id>.yaml` is missing there. Always point
`DATAPROC_ROOT` (or `--rigs-dir`) and `--out` at scratch directories.

| Flag | Meaning |
|---|---|
| `--out DIR` | where the recording directories go (created) |
| `--episodes N` | number of recordings (default 1) |
| `--calibration` | calibration recordings (fixed pose + wobble, width swept over the full range) |
| `--seed S` | base seed; recording number `n` uses seed `S * 1000003 + n` (default 0) |
| `--duration SEC` | recording length incl. pre-roll and 0.5 s post-roll (default 15; `--fast`: 5) |
| `--rig PATH` | rig yaml (default `configs/rigs/rig_v1.yaml`) |
| `--fast` | low-resolution rig `<rig_id>_fast` (main x0.25 = 320x180, ultrawide x0.5 = 410x308) and 2 s pre-roll; pass `--rig rig_v1_fast` to `dataproc calibrate` |
| `--imu-gap SEC` | remove SEC of IMU samples from the middle of each recording (an IMU dropout: `run` rejects it at validate with `imu_gap`, so it lands in `failed/`) |
| `--contract` | write dataproc's own format (`<session>_episode_NNNNNN`, with `.verified`) instead of the Pi's |
| `--session ID` | session id (default `sim-<UTC yyyymmddTHHMMSSffffff>`); names the `--contract` directories; in Pi format only recorded in `truth.npz` (the adapter's session id is `pi-<date>`) |
| `--rigs-dir DIR` | the rig yaml used is copied there as `<rig_id>.yaml` **if missing** (default `$DATAPROC_ROOT/rigs`); an existing, different file is left alone with a warning |
| `--unverified` | with `--contract`: do not write `.verified` (as if the Jetson had not checked the transfer yet) |

Numbering: `NNNNNN` (Pi `meta.json` `episode`, or contract `episode_number`) continues after the largest
`episode_NNNNNN` / `..._episode_NNNNNN` number already in `--out`, so calibration and demonstration runs can share
one `--out` and every recording stays unique (the adapter names it `pi-<date>_episode_NNNNNN`). The Pi-format copy
carries the frame `exposure_ns` column (the sim knows it) and the whole recording is the button span (the adapter has
no button presses). The wall anchor is the real time of generation.

Python API (`dataproc.sim.episode`):

```python
generate_episode(out_dir, rig, kind="demonstration" | "calibration", seed=0, duration_s=15.0, preroll_s=2.5,
                 session_id="sim", episode_number=0, faults=None, *, verified=True, imu_rate_hz=416.0,
                 imu_noise=True, noise_sigma=1.0, ss=2, scene_seed=0, texels_per_m=400.0, t0_ns=None,
                 unix_anchor_ns=None, crf=10, preset="veryfast", extra_quads=()) -> SimEpisode
fast_rig(rig_yaml_path) -> (yaml_dict, RigConfig)        # the --fast rig
read_truth(episode_dir) -> dict                          # truth.npz
run_sim(..., pi_format=False, imu_gap_s=None)             # the CLI driver (pi_format=True unless --contract)
```

## Time-stamp convention

- **True time** is the rig's IMU time base (what processing calls time). The recording starts at `t0_ns` (default
  `1e12 + 1e11 * episode_number` ns on `CLOCK_MONOTONIC`), at the first IMU sample.
- Camera frame `k` of role `r` is rendered at its **true mid-exposure time**
  `t_mid = t0 + phase_r + round(k * 1e9 / fps)` (phase 3 ms main, 7 ms ultrawide; constant exposure 4 ms main,
  8 ms ultrawide). Its CSV `t_ns` is the inverse of `RigConfig.camera_time_ns`:
  `t_ns = t_mid - s * (exposure_ns // 2) - time_offset_ns` (`s = +1` start_of_exposure, `-1` end_of_exposure),
  so `rig.camera_time_ns(role, t_ns, exposure_ns) == t_mid` exactly (tested, also with start_of_exposure and
  non-zero offsets). There is no motion blur or rolling shutter: the frame is the instant `t_mid`.
- IMU CSV `t_ns = t_true - imu.time_offset_ns`, so `rig.imu_time_ns(t_ns)` is the true sample time.
- `button_start_ns = t0 + preroll_s`, `button_stop_ns = t0 + duration_s - 0.5 s`; every stream covers it.
- `wall_anchor = {clock_ns: t0 - 0.5 s, unix_ns: ...}` (the CLI uses the real time).

## `truth.npz`

Inside the episode directory and listed in `MANIFEST.sha256` (listed extra files are allowed by the contract).
Byte-deterministic (fixed zip timestamps). Read with `read_truth(dir)` or `np.load`.

| Key | Shape / type | Meaning |
|---|---|---|
| `version` | str | `dataproc.sim.truth.v1` |
| `rig_id`, `kind`, `session_id` | str | |
| `seed`, `scene_seed`, `episode_number` | int64 | |
| `t_ns` | (N,) int64 | dense true times, 1 kHz, `t0 .. t0 + duration` |
| `T_W_imu` | (N, 4, 4) float64 | IMU pose in the gravity-aligned z-up room world (metres) |
| `width_m` | (N,) float64 | finger-tag centre separation `right.x - left.x` in the **virtual** ultrawide frame, i.e. exactly what the ArUco + PnP stage measures (iPhUMI's identity `aruco_measured_width == aruco_actual_width` calibration). **Not** the exported width: the zarr `gripper_width` is the measured separation minus the calibration `min_width` (iPhUMI `get_gripper_width_offset`), i.e. ≈ `width_m - 0.030` in sim, so end-to-end comparisons must use `width_m - min_width` |
| `cam0_t_ns`, `cam1_t_ns` | (n,) int64 | true mid-exposure times of the frames in each MP4 (after dropped frames) |
| `imu_t_ns` | (m,) int64 | true times of the samples in `imu.csv` |
| `T_imu_cam0`, `T_imu_cam1` | (4, 4) | extrinsics used (from the rig) |
| `T_virt_tag_left0`, `T_virt_tag_right0` | (4, 4) | tag poses in the virtual ultrawide frame at width 0 (add `-+width/2` to x) |
| `button_start_ns`, `button_stop_ns` | int64 | as in `meta.json` |

`T_W_cam = T_W_imu @ T_imu_cam`; the SLAM camera is the physical ultrawide (`cam1`), so the true TCP is
`T_W_imu @ T_imu_cam1 @ rig.T_cam_tcp`. The truth world is *not* the SLAM world: compare up to a rigid transform
(yaw + translation for gravity-aligned VIO).

## Trajectory (`trajectory.py`)

The IMU body moves in a 3 x 3 x 2.5 m room (`x, y in [-1.5, 1.5]`, `z in [0, 2.5]`, z up). Every channel (x, y, z,
yaw about world z, pitch about world y, roll about world x, width) is a sum of terms with analytic first and second
derivatives: min-jerk segments (quintic, C2) between waypoints and windowed sums of sines. The IMU is computed
**exactly** from them (gyro = body rate, accel = `R^T (a - g)`, `g = (0, 0, -9.80665)`), then the vendored
`imu_synth.add_noise` adds the rig's noise: white noise at the rig's noise densities (`density * sqrt(rate)`), plus a bias
random walk starting from a seeded constant bias whose per-axis std is 2e-3 rad/s (gyro) and 0.05 m/s^2 (accel,
`imu_synth` defaults), at 416 Hz on `imu_synth.imu_times`. A test checks the exact IMU against the vendored
`synth_imu` on the same poses.

- 0 - 0.3 s: motionless (gravity / bias).
- 0.3 s - pre-roll end (start press): **excitation** on all six axes, +-3-5 cm and +-5-8 deg at 0.5-1.1 Hz,
  peaks ~0.3-0.4 m/s and ~1 rad/s (ORB-SLAM3 mono-inertial initialisation needs translation + rotation).
- Demonstration: reach (~0.5 m forward and down towards the block on the table, pitching down), grasp pause, lift
  (~16 cm), hold, sideways move (~20 cm), settle; peak speed ~0.25-0.35 m/s (< 0.5 m/s), segments time-scaled to
  fit the button interval (short episodes scale the motion down instead of speeding up). Width: closed (34 +- 2 mm)
  -> open (78 +- 5 mm) during the reach -> closed at the grasp, then held (the block is raised).
- Calibration: no demonstration motion, width swept 30 <-> 90 mm in ~1 s half-cycles, ending closed.
- A slow +-1 cm / +-1.5 deg wobble (0.15-0.5 Hz) runs from 0.3 s to the end (hands are never still; keeps parallax).
- Base pose: ultrawide at ~(-0.2, 0, 1.15) m looking along +x, pitched down 30 +- 4 deg; seeded randomisation of
  start and waypoints.

## Renderer (`render.py`)

- **Scene** (`build_scene(seed)`, cached; the same room for every episode by default): the room, a table
  (0.75 m high), the "eraser" block on it, a cabinet, a crate, a box on the table and one on the floor, all
  axis-aligned boxes, so depth varies from ~0.2 m to ~3 m. Each face has its own seeded RGB texture at 400
  texels/m (2.5 mm), packed into one atlas (~4096 x 3220).
- **Texture choice (ORB-SLAM3):** multi-octave band-limited value noise (2 to 256 texels, pink-ish weights; the
  finest octave is 2 texels so it does not alias at distance), a per-face base colour and tint, then
  ~90 random anti-aliased shapes per m^2 (circles, ellipses, polygons, lines, rotated rectangles, 1.2 to 25 cm,
  random colours), plus a fine grain layer. Every face is different and nothing repeats, so ORB descriptors are
  unique (no tiled aliasing that breaks matching / loop closure), shapes give sharp corners at many scales (FAST
  fires at every pyramid level), and the noise keeps flat areas from being featureless. Check on a full-res
  episode: ORB (1500 features, 8 levels) saturates at 1500 keypoints in every ultrawide frame, and consecutive
  frames have a median of ~1165 (min ~950) fundamental-matrix RANSAC inliers.
- **Cameras:** rays are the vendored model's `unproject` of a 2 x 2 supersampled pixel grid (computed once per
  camera), so the image geometry is exactly `PinholeRadtan` (main) / `FisheyeEquidistant` (ultrawide); supersample
  `k` of pixel `u` sits at `u + (k + 0.5) / 2 - 0.5` and `cv2.resize(INTER_AREA)` box-averages them (anti-aliasing;
  ArUco decodes reliably). A numba `parallel` kernel finds the nearest hit (boxes and quads, nearest wins) and
  outputs atlas coordinates; `cv2.remap` samples the atlas bilinearly. Rays outside the model's valid range or
  hitting nothing are black. Uniform lighting. Optional per-frame Gaussian noise (`noise_sigma`, default 1 DN,
  from a seeded bank of 8 patterns).
- **Finger tags:** two ArUco `DICT_4X4_50` markers (`aruco.left_id` / `right_id`, black square
  `aruco.marker_size_m`, plus a 1-cell white quiet zone), rigid in the ultrawide frame. In the **virtual**
  ultrawide frame (the physical frame is `R_phys_virt @ virtual`): centres at `(-+width/2, 0.045, aruco.nominal_z_m)`,
  orientations `rot_y(+-15 deg) @ rot_x(150 deg)` (left `+`), facing the camera and tilted like iPhUMI's fingers
  (sensor_rig `TAG_POSES`). No fixed x offset: the separation *is* `width_m`. They are rendered in every camera
  (in main only if they project into it, which for rig_v1 they do not). Only the tags are drawn (no finger bodies).
- **Video:** PyAV libx264, `crf 10` (near-lossless), `preset veryfast`, `yuv420p`, fps from the rig. Output is
  byte-deterministic for a given seed on a given machine (x264 is deterministic for a fixed thread count).

## Faults

`faults={...}` (or `Faults(...)`), for downstream tests:

| Fault | Effect | Validator |
|---|---|---|
| `drop_frames: {cam: [capture indices]}` | frames neither encoded nor listed; `sensor_seq` gaps; `dropped_frames` counts them | `sensor_seq_gap` warning |
| `imu_gaps: [(start_s, dur_s)]` | samples in `[t0 + start, t0 + start + dur)` removed; `sample_idx` gap; `dropped_samples` counts them | `imu_sample_gap` warning |
| `count_mismatch: [cam]` | the last CSV row removed (video and `frame_count` keep it) | `count_mismatch` + `decoded_count_mismatch` errors |

## Speed (laptop, 20 threads)

| Episode | Time |
|---|---|
| full res, 15 s (main 1280x720 @ 60 = 900 frames, ultrawide 820x616 @ 30 = 450 frames) | ~95 s (main ~79 ms/frame, ultrawide ~53 ms/frame; ~80 % ray casting) |
| `--fast`, 5 s (300 + 150 frames) | ~5-7 s |

First use in a process also builds the scene (~1.5 s) and the main ray table (~1.4 s full res); numba compiles once
and caches to `__pycache__`. Sizes for a full-res 15 s episode: 17 MB per MP4, 0.6 MB `imu.csv`, 1.4 MB
`truth.npz`.

## Limitations

- No motion blur, rolling shutter, auto-exposure, lens vignetting or lighting changes; exposure is constant.
- The scene is static and axis-aligned; the "eraser" does not move when grasped (only the width and motion follow
  the task).
- Only the tags are rendered, not the finger bodies; tags (and their white zone) are fixed in the ultrawide image,
  which a real SLAM front end should mask.
- `--fast` episodes use a different `rig_id` (`<rig_id>_fast`), so they need that rig yaml in `rigs/` (the CLI
  copies it there if missing).
