# dataproc: Pi recordings → policy training data

`dataproc` runs on the Jetson. It takes the episodes the Pi 5 records into `/data/incoming/` and turns them into one
training file per task, `<task>.zarr.zip`. That file trains both the **behavior prompting** policy
(`umi_policy_dunetp`) and the **regular diffusion** policy (`umi_policy_dunet`) unchanged.

- **This README** explains *what* goes in, *what* comes out and *how* each piece of data is converted. It also covers
  one-time setup.
- **[DIRECTIONS.md](DIRECTIONS.md)** is the step-by-step "I have episodes in `/data/incoming`, now what?" guide.

```
 Pi 5 rig ──rsync──▶ /data/incoming/episode_NNNNNN/          (Pi format; read only, never changed)
                            │
                            │  dataproc run --task <task> ...
                            ▼
   1. adapt     convert a copy into dataproc's format         ~/data/incoming/<copy>/  (brief staging)
   2. intake    file it under the task                        ~/data/tasks/<task>/episode_N/raw/
   3. process   calibration + 8 stages per episode            ~/data/tasks/<task>/episode_N/processed/
   4. export    every good episode into one zip               ~/data/tasks/<task>/output/<task>.zarr.zip
                            │
                            │  copy to the cluster (DIRECTIONS.md, step 7)
                            ▼
        ~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip  ──▶  train (dunetp or dunet)
```

---

## 1. What one Pi episode contains (the input)

One folder per recording, e.g. `/data/incoming/episode_000004/`. Each recording is a fixed **15 s** clip.

| File | What it is | Details |
|---|---|---|
| `cam0.mp4` | **Ultrawide camera** (IMX219, ~160° fisheye) | 1280×960 at 30 fps, H.264. Sees the gripper fingers and the scene. |
| `cam1.mp4` | **Main camera** (IMX519) | 1600×900 at 60 fps, H.264. The forward view. |
| `cam0_frames.csv`, `cam1_frames.csv` | One row per video frame: `frame_index,t_ns` | `t_ns` = when the frame was captured (libcamera `SensorTimestamp`, ns on the Pi's boot clock). |
| `imu.csv` | `sample_index,t_ns,gx,gy,gz,ax,ay,az` | LSM6DSOX at 104 Hz, **raw int16 counts** (gyro and accelerometer). Same clock as the cameras. |
| `meta.json` | The Pi's notes | Episode number, `valid` / `problems`, each camera's model, size, fps and frame count, IMU scale factors (`accel_g_per_lsb`, `gyro_dps_per_lsb`), start/end time. |
| `MANIFEST.sha256` | Checksums of the files above | Proves nothing was corrupted in transfer. |
| `COMPLETE` | Empty marker | The Pi finished writing the episode. |

**Not in a Pi episode, but needed for training:**

| Missing | How dataproc fills the gap |
|---|---|
| Which task it belongs to | You name it: `dataproc run --task <task>` |
| Which recording is the gripper calibration | You name it: `--calibration-episode <dir>` |
| Gripper position and rotation | Computed by SLAM (ORB-SLAM3) from the ultrawide video + IMU |
| Gripper opening width | Measured from the finger ArUco tags in the ultrawide video, scaled by the calibration |
| Start/stop of the demonstration | None recorded: the **whole clip** is used |
| Per-frame exposure time | Assumed 5 ms (the Pi doesn't record it yet) |
| Camera/IMU geometry (lens model, offsets, gripper-tip position) | The rig file `configs/rigs/rig_pi_v1.yaml` (estimated values until calibrated) |

---

## 2. What training needs (the output)

One `.zarr.zip` per task (iPhUMI's format, the same one the existing `firstreal-single` dataset uses). Per
**step** (60 per second of recording):

| Array in the zip | Shape (per step) | Meaning | Comes from |
|---|---|---|---|
| `data/camera_right_main_rgb` | 224×224×3 image | Main camera view | `cam1.mp4` (IMX519) |
| `data/camera_right_ultrawide_rgb` | 224×224×3 image, **10 per second** | Ultrawide view, undistorted to the iPhone ultrawide's lens | `cam0.mp4` (IMX219) |
| `data/gripper_right_eef_pos` | 3 floats (m) | Gripper tip position, gravity-aligned world | SLAM + rig geometry |
| `data/gripper_right_eef_rot_axis_angle` | 3 floats | Gripper tip rotation (axis-angle) | SLAM + rig geometry |
| `data/gripper_right_gripper_width` | 1 float (m) | How open the gripper is | Finger tags + calibration |
| `meta/episode_ends`, `meta/episode_names` | per episode | Where each episode ends; its Pi name | Bookkeeping |
| `meta/task_names` (+ `task_lengths`, `task_data_ends`, ...) | per episode | The task label (`--task`), one task per episode | `--task` |
| `meta/*_camera_right_ultrawide_rgb` | index maps | Which 10 Hz ultrawide image goes with each 60 Hz step | Frame timing |

How the policies read it (both use the same file):

- **Observations:** 2 recent main images (taken every 3rd step, i.e. 20 Hz), 2 recent ultrawide images, and the
  gripper pose (made relative to the current pose; rotation converted to a 6-number form) and width.
- **Actions:** not stored. The training code builds them from future poses and widths
  (position 3 + rotation 6 + width 1, predicted in chunks).
- **Behavior prompting (`dunetp`) only:** pairs each training sample with *another whole demo of the same
  `task_names` value* as its "prompt". So it needs **at least 2 good demos per task**; more is much better
  (`firstreal-single` had 45).
- **Regular diffusion (`dunet`):** ignores the task names and samples per episode. The cluster training script
  defaults to `dunetp`; switching to `dunet` is a one-line change in `scripts/cluster/train.sbatch`.

---

## 3. Step by step: from Pi episode to training file

Everything below runs from one command, `dataproc run` (see [DIRECTIONS.md](DIRECTIONS.md)). Each step saves its
result, so a later run continues where the last one stopped instead of starting over.

| # | Step | Reads | Produces | Stops the episode when |
|---|---|---|---|---|
| 1 | **adapt** (`dataproc/pi_adapter.py`) | `/data/incoming/<dir>/` | A converted copy: cameras renamed by sensor (IMX519 → `cam0` = main, IMX219 → `cam1` = ultrawide), IMU in rad/s and m/s², `meta.json` in dataproc's format, new checksums | No `COMPLETE`, a checksum mismatch, the Pi marked it invalid, or it was already used (`~/data/pi_sources.csv`) |
| 2 | **intake** | the converted copy | `tasks/<task>/episode_N/raw/` (or `calibration/<name>/raw/`), a row in `episodes.csv` | Format errors, duplicate, wrong rig: moved to `~/data/quarantine/` with `reason.json` |
| 3 | **calibration** (once per task) | the calibration episode's ultrawide video | `gripper_range.json`: tag spacing at fully closed and fully open | Both finger tags never seen together. Every episode then stops at step 8. |
| 4 | **check** | the whole episode, every video frame decoded | `check.json` | A file problem, or resolution/fps doesn't match the rig file |
| 5 | **slam** | ultrawide video (grayscale) + IMU | `slam/trajectory.tum`: the IMU's path through space, gravity-aligned | No ORB-SLAM3, or it never initialised (the camera didn't move enough) |
| 6 | **grid** | the recording's time span | `grid.npy`: one step every 1/60 s over the whole clip | Clip shorter than 2 steps |
| 7 | **pose** | SLAM path + rig geometry | `pose.npz`: gripper-tip position and rotation at each step | (Steps with no SLAM pose within 100 ms are marked invalid) |
| 8 | **width** | every ultrawide frame + `gripper_range.json` | `width.npz`: gripper width at each step | No calibration (or it failed) |
| 9 | **validate** | everything above | `validate.json`: the usable span of the episode | Too fast (> 3 m/s), SLAM lost track (< 95% tracked), tags visible in < 80% of frames, an IMU gap > 20 ms, an ultrawide gap > 100 ms, or < 2 s usable |
| 10 | **images** | both videos | the selected frames (in memory) | A step has no main frame within 25 ms |
| 11 | **export** | images, poses, widths | `episode.zarr`: this episode's training data | — |
| 12 | **zip** (task level) | every exported `episode.zarr` | `output/<task>.zarr.zip` + `output/report.json` | No episode exported |

A failed episode never blocks the others. The zip contains every episode that reached export.

---

## 4. Per data source: what happens to each file

```
Pi file                    used in steps                       ends up as
───────────────────────    ────────────────────────────────    ─────────────────────────────────────────────
cam0.mp4 (IMX219 fisheye)  calibration, check, slam, width,    camera_right_ultrawide_rgb
                           images                              + (via slam) eef_pos / eef_rot_axis_angle
                                                               + (via tags) gripper_width
cam1.mp4 (IMX519 main)     check, images                       camera_right_main_rgb
cam*_frames.csv            every step that aligns in time      the timing that ties images, poses, widths together
imu.csv                    slam, validate                      (via slam) eef_pos / eef_rot_axis_angle; not stored itself
meta.json                  adapt, check                        camera roles, sizes, IMU scale factors, episode name
MANIFEST.sha256, COMPLETE  adapt, intake, check                (integrity only)
rig_pi_v1.yaml (not Pi)    every geometric step                lens models, camera↔IMU offsets, gripper-tip offset
calibration episode        calibration                         the scale for gripper_width
```

**Ultrawide video (`cam0.mp4`, IMX219).** The workhorse.
1. *SLAM:* frames are written as grayscale images with their timestamps and fed to ORB-SLAM3 together with the IMU.
   ORB-SLAM3 tracks features to recover how the camera moved.
2. *Width:* each frame is remapped from the physical fisheye to a virtual 640×480 pinhole view (the iPhone
   ultrawide's lens, so data matches the iPhone pipeline). The two finger ArUco tags are detected in it. The
   horizontal distance between them, mapped through the calibration range, gives the width.
3. *Images:* every 1/10 s, the nearest frame is taken, remapped the same way, letterboxed to a black square and
   resized to 224×224. Each 60 Hz step points at the latest of these at or before it.

**Main video (`cam1.mp4`, IMX519).** Only used for images: for every 60 Hz step, the frame nearest in time
(at most 25 ms away) is letterboxed and resized to 224×224. No undistortion.

**Frame timestamps (`cam*_frames.csv`).** Each `t_ns` marks the end of exposure. dataproc shifts it back half an
exposure to the middle of the exposure. That makes images, IMU and poses line up in time. Gaps between stamps
reveal dropped frames.

**IMU (`imu.csv`).** Converted from raw counts to rad/s and m/s² using the scale factors in `meta.json`.
ORB-SLAM3 uses it for real-world scale (metres) and gravity direction. dataproc uses it to point the world's z axis
up and to check SLAM's scale. Validate rejects an episode with an IMU gap over 20 ms. The IMU readings themselves are
not stored in the zip.

**Poses.** SLAM gives the IMU's pose over time. Each step's gripper-tip pose is that pose, times the
IMU→ultrawide-camera offset, times the camera→gripper-tip offset (both from the rig file), in a gravity-aligned
world.

**Gripper width.** The calibration episode gives the tag spacing at fully closed and fully open, plus a mapping
between the two tags' positions. In demos, width = tag spacing (or the predicted spacing when one tag is hidden),
clipped to that range, with fully closed reading 0.

---

## 5. Command-line options and why they exist

`dataproc run` is the one command for Pi recordings. Its options:

| Option | Default | Why it exists |
|---|---|---|
| `--task NAME` | required | The Pi doesn't know what task you're recording. The name is the task folder and the label written into the zip (`task_names`), which behavior prompting uses to pair demos. Letters, digits, `_`, `-`. |
| `--calibration-episode DIR` | — | The gripper width needs one calibration recording per task, and the Pi can't mark which recording that is. Required the first time, unless `--no-calibration`. A newer one replaces the old one. |
| `--no-calibration` | off | To process recordings before a calibration exists (e.g. to test whether SLAM tracks). Everything up to the width step runs and is kept. No zip until you add a calibration; then only width onward reruns. |
| `--episodes DIR ...` | every new one | By default every unused episode in `/data/incoming` is taken as a demo of `--task`. This picks just a few, e.g. for a quick test, or when the folder holds recordings of different tasks. |
| `--dry-run` | off | Prints what would be taken (and what's skipped, with reasons) and changes nothing. Taking an episode marks it as used, so check first. |
| `--source DIR` | `/data/incoming` | Where the Pi delivers. Only ever read. |
| `--rig RIG_ID` | `rig_pi_v1` | Which rig file describes the cameras, IMU and gripper (`~/data/rigs/<id>.yaml`, copied in automatically). Change it when the hardware or its calibration changes. |
| `--slam auto\|orbslam3\|none\|oracle` | `auto` | `auto` uses ORB-SLAM3 when it's built, else `none` (stop at SLAM, exit code 3, and resume later). `oracle` is only for fake `dataproc sim` episodes. |

Exit codes: `0` = zip built, `3` = stopped at SLAM because none is installed, `1` = anything else (including "no
episode got through": read the table it prints).

Other commands:

| Command | Use it for |
|---|---|
| `dataproc status --task NAME` | A table of every episode: last stage reached, and why it stopped. Read only. |
| `dataproc process --task NAME [--episode N] [--force]` | Re-run processing without taking new episodes. `--force` redoes every stage, e.g. after editing the rig file by hand. |
| `dataproc intake ...`, `dataproc sim ...` | Lower level: for a Pi that sends dataproc's own format directly ([docs/CONTRACT.md](docs/CONTRACT.md)), and for fake test data. |

Settings from the environment:

| Variable | Default | Effect |
|---|---|---|
| `DATAPROC_ROOT` | `~/data` | Where all output goes. **If it's still set from the practice run, output goes to `~/dataproc_practice`.** `dataproc run` prints the folder it uses on its first line. |
| `ORB_SLAM3_DIR` | `~/ORB_SLAM3` | Where ORB-SLAM3 is built. |

---

## 6. One-time setup

Written for this Jetson: JetPack 7 (Ubuntu 24.04), Python 3.12. Run everything as your normal user unless it says
`sudo`. **On the current Jetson all of this is already done.**

**A1. System packages.** `build-essential` is needed to compile one Python package (`numcodecs`).

```bash
sudo apt update
sudo apt install -y git python3-venv build-essential
```

**A2. Code** (branch `rnoro/data-work`).

```bash
cd ~ && git clone git@github.com:rubennoro/Capstone.git
cd ~/Capstone && git checkout rnoro/data-work
```

**A3. Python environment.** Takes several minutes. Versions are pinned in `pyproject.toml` (newer OpenCV, SciPy and
numcodecs releases break it).

```bash
python3 -m venv ~/envs/env_dataproc
~/envs/env_dataproc/bin/pip install --upgrade pip
cd ~/Capstone && ~/envs/env_dataproc/bin/pip install -e "data-processing[test]"
```

**A4. Turn the environment on** in every new terminal (or add the line to the end of `~/.bashrc`):

```bash
source ~/envs/env_dataproc/bin/activate
```

**A5. Check the install.** The tests take ~7 minutes and should end with only passes and skips.

```bash
dataproc --help
cd ~/Capstone && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest data-processing/tests -q
```

**A6. Data folder.** `mkdir -p ~/data`. Everything dataproc writes goes under it. `dataproc run` creates the rest
(`incoming/`, `tasks/`, `rigs/`, `quarantine/`) and copies the rig file in.

**A7. ORB-SLAM3.** Built natively in `~/ORB_SLAM3` (done on 2026-11-07). Recipe and patches:
[docs/ORBSLAM3.md](docs/ORBSLAM3.md). `dataproc run` finds it automatically.

---

## 7. Practice run on fake data (optional)

`dataproc sim` makes fake episodes with a known answer, so you can see every command work without hardware. It
uses its own folder so nothing mixes with real data.

```bash
source ~/envs/env_dataproc/bin/activate
export DATAPROC_ROOT=~/dataproc_practice        # this terminal only
mkdir -p $DATAPROC_ROOT/incoming
dataproc sim --out $DATAPROC_ROOT/incoming --calibration --episodes 1 --fast --seed 1
dataproc intake --task practice --calibration
dataproc sim --out $DATAPROC_ROOT/incoming --episodes 3 --fast --seed 2
dataproc intake --task practice
dataproc process --task practice --slam oracle   # oracle = the fake data's known answer instead of ORB-SLAM3
dataproc status --task practice
```

You should see `calibration: done`, `exported` for each episode, and
`~/dataproc_practice/tasks/practice/output/practice.zarr.zip`. A `note: rig ... still has nominal sections` is
expected (the rig isn't calibrated).

**When done, run `unset DATAPROC_ROOT` (or close the terminal).** Otherwise your real runs also go to
`~/dataproc_practice`.

---

## 8. Troubleshooting

| Message | Meaning | Fix |
|---|---|---|
| `data root: ~/dataproc_practice (from $DATAPROC_ROOT ...)` | Output is going to the practice folder | `unset DATAPROC_ROOT`, then run again |
| `slam: ... never initialised (0 keyframes ...)` | ORB-SLAM3 couldn't start: the camera didn't move enough, or the scene has no texture | Record with motion (start with ~2 s of gentle side-to-side movement) in front of a textured scene |
| `calibration: calibration_failed ... finger tags 0/1 never both detected` | Tags not visible in the calibration recording | Re-record calibration with both tags in the ultrawide's view, then `dataproc run --task T --calibration-episode <new dir>` |
| `width: no_calibration` | The task has no calibration (e.g. `--no-calibration`) | Add one with `--calibration-episode` |
| `validate: tag_coverage` / `slam_tracking` / `speed` | Tags hidden too often / SLAM lost track / moved too fast | Better recordings: tags in view, textured scene, smoother motion |
| `skip episode_X: no COMPLETE` | Still being copied from the Pi | Wait, run again |
| `skip episode_X: already taken (...)` | Used in an earlier run | Nothing to do. To reprocess it: `dataproc process --task T --episode N --force`. (A *quarantined* one can be retried by deleting its row from `~/data/pi_sources.csv`.) |
| `--episodes X cannot be used: ...` | A named episode is missing, used or invalid | Read the reason; pick another |
| `incoming/ is not empty` | Leftovers from an interrupted run | `dataproc intake --task <task>` to finish them, or move them away |
| `quarantined: <reasons>` | Failed a file-level check | `~/data/quarantine/<name>/reason.json`; codes are in [docs/CONTRACT.md](docs/CONTRACT.md) |
| `dataproc: command not found` | Environment not on | `source ~/envs/env_dataproc/bin/activate` |

---

## More detail

| File | What's in it |
|---|---|
| [DIRECTIONS.md](DIRECTIONS.md) | Step by step: from episodes in `/data/incoming` to a training run |
| [docs/PI_ADAPTER.md](docs/PI_ADAPTER.md) | The exact Pi → dataproc field mapping and its assumptions |
| [docs/CONTRACT.md](docs/CONTRACT.md) | The format dataproc reads directly (what the Pi should eventually send) |
| [docs/PROCESS.md](docs/PROCESS.md) | Every processing stage, its checks and thresholds |
| [docs/EXPORT.md](docs/EXPORT.md) | Images and the zarr export in detail |
| [docs/GEOMETRY.md](docs/GEOMETRY.md) | Frames, poses and how gripper width is measured |
| [docs/DESIGN.md](docs/DESIGN.md) | Folder layout, the output zarr schema, conventions |
| [docs/INTAKE.md](docs/INTAKE.md), [docs/SIM.md](docs/SIM.md), [docs/ORBSLAM3.md](docs/ORBSLAM3.md) | Intake rules, the fake-data generator, the ORB-SLAM3 build |
