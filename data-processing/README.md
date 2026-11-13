# dataproc: Pi recordings → policy training data

`dataproc` runs on the Jetson. It takes the recordings the Pi 5 makes into `/data/incoming/` and turns them into a
training file, `<dataset>.zarr.zip`. That file trains both the **behavior prompting** policy
(`umi_policy_dunetp`) and the **regular diffusion** policy (`umi_policy_dunet`) unchanged.

- **This README** explains *what* goes in, *what* comes out and *how* each piece of data is converted. It also covers
  one-time setup, a practice run on fake data, and troubleshooting.
- **[DIRECTIONS.md](DIRECTIONS.md)** is the step-by-step "I have recordings in `/data/incoming`, now what?" guide.

Three commands, in order, once per recording session:

```
 Pi 5 rig ──rsync──▶ /data/incoming/episode_NNNNNN/          (Pi format; read only, never changed)
                            │
   dataproc calibrate TASK EPISODE_DIR                         fit + check the gripper calibration on one recording;
                            │                                  on success create ~/data/tasks/TASK/ (or TASK_2, ...)
   dataproc run TASK_FOLDER START_DIR END_DIR                  every recording START..END, one at a time:
                            │                                    passed every stage → tasks/<folder>/episode_N/
                            │                                    failed somewhere   → tasks/<folder>/failed/<name>/
   dataproc package DATASET FOLDER [FOLDER ...]                the finished episodes of any folders → one zip:
                            │                                  ~/data/datasets/<task>/<DATASET>/<DATASET>.zarr.zip
                            │  rsync to the cluster (it prints the commands)
                            ▼
        ~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip  ──▶  train (dunetp or dunet)
```

- **One task folder = one calibration = one session** on one hardware setup. A second `calibrate` with the same
  task name makes `block_raise_2`, then `block_raise_3`, ...; all of them are labelled `block_raise` in the zip.
- `run` numbers the episodes of a folder from **`episode_1`**, gap-free, in the order they passed. A recording that
  fails never gets a number.
- `package` can combine several sessions (folders) of one task, or several tasks, into one dataset.

---

## 1. What one Pi recording contains (the input)

One folder per recording, e.g. `/data/incoming/episode_000004/`. Each recording is a fixed **15 s** clip.

| File | What it is | Details |
|---|---|---|
| `cam0.mp4` | **Ultrawide camera** (IMX219, ~160° fisheye) | 1280×960 at 30 fps, H.264. Sees the gripper fingers and the scene. |
| `cam1.mp4` | **Main camera** (IMX519) | 1600×900 at 60 fps, H.264. The forward view. |
| `cam0_frames.csv`, `cam1_frames.csv` | One row per video frame: `frame_index,t_ns` | `t_ns` = when the frame was captured (libcamera `SensorTimestamp`, ns on the Pi's boot clock). |
| `imu.csv` | `sample_index,t_ns,gx,gy,gz,ax,ay,az` | LSM6DSOX at 104 Hz, **raw int16 counts** (gyro and accelerometer). Same clock as the cameras. |
| `meta.json` | The Pi's notes | Episode number, `valid` / `problems`, each camera's model, size, fps and frame count, IMU scale factors (`accel_g_per_lsb`, `gyro_dps_per_lsb`), start/end time. |
| `MANIFEST.sha256` | Checksums of the files above | Proves nothing was corrupted in transfer. |
| `COMPLETE` | Empty marker | The Pi finished writing the recording. |

**Not in a Pi recording, but needed for training:**

| Missing | How dataproc fills the gap |
|---|---|
| Which task it belongs to | You name it: `dataproc calibrate TASK ...` |
| Which recording is the gripper calibration | You name it: `dataproc calibrate TASK EPISODE_DIR` |
| Which recordings are this session's demonstrations | You name the range: `dataproc run TASK_FOLDER START_DIR END_DIR` |
| Gripper position and rotation | Computed by SLAM (ORB-SLAM3) from the ultrawide video + IMU |
| Gripper opening width | Measured from the finger ArUco tags in the ultrawide video, scaled by the calibration |
| Start/stop of the demonstration | None recorded: the **whole clip** is used |
| Per-frame exposure time | Assumed 5 ms (the Pi doesn't record it yet) |
| Camera/IMU geometry (lens model, offsets, gripper-tip position, gripper opening) | The rig file `configs/rigs/rig_pi_v3.yaml` (default: gripper-tip and IMU offsets from CAD, 80 mm opening; `rig_pi_v2.yaml` has the same cameras with estimated offsets and a 110 mm opening; `rig_pi_v1.yaml` for the old zoomed-in main camera; lens and IMU noise values estimated until calibrated) |

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
| `meta/task_names` (+ `task_lengths`, `task_data_ends`, ...) | per episode | The task label, one task per episode | `TASK` of `dataproc calibrate` |
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

## 3. Step by step: from Pi recording to training file

Each step saves its result, so a later command continues where the last one stopped instead of starting over.

| # | Step | Reads | Produces | Stops the recording when |
|---|---|---|---|---|
| 1 | **adapt** (`dataproc/pi_adapter.py`; calibrate and run) | `/data/incoming/<dir>/` | A converted copy: cameras renamed by sensor (IMX519 → `cam0` = main, IMX219 → `cam1` = ultrawide), IMU in rad/s and m/s², `meta.json` in dataproc's format, new checksums | No `COMPLETE`, a checksum mismatch, the Pi marked it invalid, or it was already taken (`~/data/pi_sources.csv`): skipped and listed |
| 2 | **calibration** (`dataproc calibrate`, once per folder) | the calibration recording's ultrawide video | `gripper_range.json`: tag spacing at fully closed and fully open | Wrong rig, tags not seen, or a measured opening (`max_width - min_width`) off the rig's `gripper.max_opening_m` (0.110 m ± 0.010 m on `rig_pi_v1`): **nothing is created** |
| 3 | **intake** (run) | the converted copy | `tasks/<folder>/.staging/<name>/raw/`, a row in `episodes.csv` | A duplicate: `~/data/quarantine/` with `reason.json` |
| 4 | **check** | the whole recording, every video frame decoded | `check.json` | A file problem, or resolution/fps doesn't match the rig file |
| 5 | **slam** | ultrawide video (grayscale) + IMU | `slam/trajectory.tum`: the IMU's path through space, gravity-aligned | ORB-SLAM3 never initialised (the camera didn't move enough). No ORB-SLAM3 installed: the recording **waits** in `.staging/` (exit 3) |
| 6 | **grid** | the recording's time span | `grid.npy`: one step every 1/60 s over the whole clip | Clip shorter than 2 steps |
| 7 | **pose** | SLAM path + rig geometry | `pose.npz`: gripper-tip position and rotation at each step | (Steps with no SLAM pose within 100 ms are marked invalid) |
| 8 | **width** | every ultrawide frame + `gripper_range.json` | `width.npz`: gripper width at each step | — (the folder always has a calibration) |
| 9 | **validate** | everything above | `validate.json`: the usable span of the recording | Too fast (> 3 m/s), SLAM lost track (< 95% tracked), tags visible in < 80% of frames, an IMU gap > 20 ms, an ultrawide gap > 100 ms, or < 2 s usable |
| 10 | **images** | both videos | the selected frames (in memory) | A step has no main frame within 25 ms |
| 11 | **export** | images, poses, widths | `episode.zarr`: this recording's training data. Then `.staging/<name>/` is renamed to `episode_N/` | — |
| 12 | **package** (`dataproc package`) | every exported `episode_N/processed/episode.zarr` of the folders named | `datasets/<task>/<DATASET>/<DATASET>.zarr.zip` + `package.json` | No exported episode, or the folders used different rigs |

A recording that stops at steps 3–11 is moved to `tasks/<folder>/failed/<name>/` with `reason.json` (stage, reason
codes, time). It never blocks the others and is never packaged.

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

## 5. Commands and options

```
dataproc calibrate TASK EPISODE_DIR [--source /data/incoming] [--rig rig_pi_v3]
dataproc run TASK_FOLDER START_DIR END_DIR [--source /data/incoming] [--slam auto|orbslam3|oracle|none]
                                          [--retry-failed] [--force] [--dry-run]
dataproc package DATASET FOLDER [FOLDER ...] [--overwrite]
dataproc status TASK_FOLDER
dataproc sim --out DIR [--episodes N] [--calibration] [--fast] [--imu-gap SEC] [--contract] ...
```

| Command / option | Why it exists |
|---|---|
| `calibrate TASK EPISODE_DIR` | The gripper width needs one calibration recording per session, and the Pi can't mark which recording that is. It fits and checks it **before** anything is created: on failure it prints the tag statistics and likely causes, creates nothing, and the recording can be retried. On success it creates the first free folder `TASK`, `TASK_2`, ... and prints the next command. `TASK` (letters, digits, `_`, `-`) is the label written into the zip (`task_names`), which behavior prompting uses to pair demos. |
| `--rig RIG_ID` (calibrate) | Which rig file describes the cameras, IMU and gripper (`~/data/rigs/<id>.yaml`, copied from `configs/rigs/` if missing). Recorded in the folder's `task.yaml`; `run` uses that. |
| `run TASK_FOLDER START_DIR END_DIR` | Processes every recording from START_DIR to END_DIR (directory names in `--source`, inclusive, natural sort), one at a time. Recordings that are still syncing, invalid, a calibration, already episodes of this folder, taken by another folder, or failed earlier are **skipped and listed**. Re-running the same range is safe: it only picks up what is new (e.g. recordings that finished syncing). |
| `--slam auto\|orbslam3\|oracle\|none` | `auto` (default) uses ORB-SLAM3 when it's built, else `none`: recordings stop at SLAM and wait in `.staging/` (exit 3); the same command resumes them later. `oracle` is only for fake `dataproc sim` recordings. |
| `--retry-failed` | Moves the range's `failed/<name>/` back to `.staging/` and processes them again. Stages whose inputs changed (rig file, `--slam` backend, dataproc version) re-run; with nothing changed the same failure comes back, so add `--force` to re-run every stage. |
| `--force` | Re-runs every stage of the selected range, e.g. after editing the rig file by hand. |
| `--dry-run` | Lists what would happen to each recording, changes nothing. |
| `--source DIR` | Where the Pi delivers (default `/data/incoming`). Only ever read. |
| `package DATASET FOLDER ...` | Combines every exported `episode_N` of the folders into one dataset, in the cluster's layout. Refuses an existing dataset name unless `--overwrite`; warns when a label has fewer than 2 episodes (prompting needs ≥ 2). Prints the `rsync` and training commands. |
| `status TASK_FOLDER` | A table of every episode, `failed/` and `.staging/` entry (last stage, why it stopped) and the calibration. Read only. |
| `sim` | Fake recordings with a known answer (section 7, [docs/SIM.md](docs/SIM.md)). |

Exit codes: `calibrate` 0 = folder created, 1 = failed (nothing created). `run` 0 = done (also when some recordings
failed: they are listed), 3 = some recordings wait for SLAM, 1 = it could not run. `package` 0 = written, 1 = error.

Settings from the environment:

| Variable | Default | Effect |
|---|---|---|
| `DATAPROC_ROOT` | `~/data` | Where all output goes. **If it's still set from the practice run, output goes to `~/dataproc_practice`.** |
| `ORB_SLAM3_DIR` | `~/ORB_SLAM3` | Where ORB-SLAM3 is built. |

The lower-level `intake` and `process` modules still exist (`dataproc/intake.py`, `dataproc/process.py`), but are no
longer commands: `calibrate` and `run` call them.

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

**A6. Data folder.** `mkdir -p ~/data`. Everything dataproc writes goes under it. `dataproc calibrate` creates the rest
(`incoming/`, `tasks/`, `rigs/`, `datasets/`, ...) and copies the rig file in.

**A7. ORB-SLAM3.** Built natively in `~/ORB_SLAM3` (done on 2026-11-07). Recipe and patches:
[docs/ORBSLAM3.md](docs/ORBSLAM3.md). `dataproc run` finds it automatically (`--slam auto`).

---

## 7. Practice run on fake data (optional)

`dataproc sim` makes fake recordings in the Pi's format, with a known answer, so you can see every command work
without hardware. Everything goes to its own folder so nothing mixes with real data. About 2 minutes.

```bash
source ~/envs/env_dataproc/bin/activate
export DATAPROC_ROOT=~/dataproc_practice        # this terminal only
rm -rf ~/dataproc_practice                      # start clean (sim numbers after what is already there)
P=$DATAPROC_ROOT/pi                             # the fake Pi's /data/incoming
dataproc sim --out $P --episodes 1 --calibration --fast         # episode_000000: a calibration
dataproc sim --out $P --episodes 4 --fast                       # episode_000001..000004: demonstrations
dataproc sim --out $P --episodes 1 --fast --imu-gap 0.5         # episode_000005: an IMU dropout (will fail)
dataproc sim --out $P --episodes 1 --calibration --fast         # episode_000006: a second session's calibration
dataproc calibrate block_raise episode_000000 --source $P --rig rig_v1_fast
dataproc run block_raise episode_000001 episode_000005 --source $P --slam oracle
dataproc package practice block_raise
dataproc calibrate block_raise episode_000006 --source $P --rig rig_v1_fast
dataproc run nope a b
```

What you should see:
- `calibrate`: `calibrate: OK -> tasks/block_raise (label block_raise, rig rig_v1_fast)` and `measured opening
  0.060 m (expected 0.060 m ± 0.010 m)` (the fake rig's gripper; the real `rig_pi_v1` expects 0.110 m). The warning
  `only 1 open/close cycle(s) detected` is expected for the fake calibration.
- `run`: `episode_000001..4 -> episode_1..episode_4` and `episode_000005 -> failed (validate: imu_gap)`, then the
  status table with `failed/pi-<date>_episode_000005` under "not episodes (never packaged)".
- `package`: `wrote ~/dataproc_practice/datasets/block_raise/practice/practice.zarr.zip (... 4 episodes ...)`, the
  failed recording listed as excluded, and the cluster commands.
- the second `calibrate`: `OK -> tasks/block_raise_2` (a second session of the same task).
- `run nope a b`: `task folder nope does not exist (or has no successful calibration): run dataproc calibrate
  <task> <episode_dir> first`, exit 1.

`--rig rig_v1_fast` is the fake rig `sim --fast` renders with (it copies it to `$DATAPROC_ROOT/rigs/`). Real
recordings use the default `--rig rig_pi_v3` (main camera 1600x1200 @ 30, geometry from CAD, gripper opening
limited to 80 mm); recordings from the old zoomed-in main camera setting (1600x900 @ 60) need `--rig rig_pi_v1`.

**When done, run `unset DATAPROC_ROOT` (or close the terminal).** Otherwise your real runs also go to
`~/dataproc_practice`.

---

## 8. Troubleshooting

| Message | Meaning | Fix |
|---|---|---|
| `calibrate: FAILED ... measured opening 0.071 m, expected 0.110 m ± 0.010 m` (`opening_out_of_range`) | The fit worked but the measured full opening is off | Open and close the gripper **fully** during the recording; check `aruco.marker_size_m` (the opening scales with it) and the ultrawide intrinsics in the rig file; check `gripper.max_opening_m` is this gripper's opening |
| `calibrate: FAILED ... the finger tags (0/1) were never seen` | No tag detected at all | Both fingers in the ultrawide view (lens cap, camera roles); `aruco.dict` / `left_id` / `right_id` match the printed tags |
| `calibrate: FAILED ... tags are seen in only N% of frames` | Detection rate too low | More light, slower open/close (motion blur), lens focus |
| `calibrate: FAILED ... the tags are not at the expected depth` | Tags far from `aruco.nominal_z_m` | Fix `aruco.nominal_z_m` in the rig file, or the camera was remounted |
| `calibrate: FAILED ... wrong --rig` (`rig_resolution_mismatch`, `rig_fps_mismatch`, `rig_mismatch`) | The recording's cameras don't match the rig file | Pass the rig this recorder uses (`--rig`) |
| `calibrate: FAILED ... already taken` | That recording is already a calibration or demo somewhere | Record a new calibration for a new session |
| `... rigs/rig_pi_v1.yaml has no gripper section, but the repo's ... has one` | `~/data/rigs/` holds a copy from before the gripper check existed | Add the printed `gripper:` section to `~/data/rigs/rig_pi_v1.yaml`, or delete that file (the repo copy is then used) |
| `task folder X does not exist (or has no successful calibration)` | `run` on a folder that `calibrate` didn't create | `dataproc calibrate <task> <episode_dir>` first; use the folder name it prints |
| `START_DIR ... comes after END_DIR` / `... is not a directory in /data/incoming` | Bad range | Check `ls /data/incoming`; swap or fix the names |
| `-> failed (<stage>: <reasons>)` in the run summary | The recording stopped at that stage; it's in `tasks/<folder>/failed/<name>/` with `reason.json` | Read the reason (below). After fixing the cause (e.g. the rig file): `dataproc run <folder> <start> <end> --retry-failed` |
| `skip (failed earlier at <stage>: <reason>; --retry-failed to try again)` | A re-run of the range leaves failed recordings alone | Add `--retry-failed` if you changed something |
| `-> waiting for SLAM`, exit code 3 | No ORB-SLAM3 found: the recordings wait in `.staging/` | Build it ([docs/ORBSLAM3.md](docs/ORBSLAM3.md)), then run the same command again; nothing before SLAM is redone |
| `slam: ... never initialised (0 keyframes ...)` | ORB-SLAM3 couldn't start: the camera didn't move enough, or the scene has no texture | Record with motion (start with ~2 s of gentle side-to-side movement) in front of a textured scene |
| `validate: tag_coverage` / `slam_tracking` / `speed` / `imu_gap` | Tags hidden too often / SLAM lost track / moved too fast / IMU dropout | Better recordings: tags in view, textured scene, smoother motion |
| `skip (... still being recorded or transferred)` | No `COMPLETE` yet | Wait, run the same range again |
| `skip (is a calibration (of task folder ...))` | The range includes a calibration recording | Nothing to do |
| `incoming/ is not empty` | Leftovers in `~/data/incoming/` from an interrupted older command | Move them away |
| `another dataproc calibrate/run holds .../.run.lock` | Another command is running | Wait for it |
| `dataset ... already exists` (package) | That dataset name is taken | Pick another name, or `--overwrite` |
| `quarantined: <reasons>` | A duplicate recording (same session and episode number) | `~/data/quarantine/<name>/reason.json`; codes are in [docs/CONTRACT.md](docs/CONTRACT.md) |
| `dataproc: command not found` | Environment not on | `source ~/envs/env_dataproc/bin/activate` |

---

## More detail

| File | What's in it |
|---|---|
| [DIRECTIONS.md](DIRECTIONS.md) | Step by step: from recordings in `/data/incoming` to a training run |
| [docs/PI_ADAPTER.md](docs/PI_ADAPTER.md) | The exact Pi → dataproc field mapping and its assumptions |
| [docs/CONTRACT.md](docs/CONTRACT.md) | The format dataproc reads directly (what the Pi should eventually send) |
| [docs/PROCESS.md](docs/PROCESS.md) | Every processing stage, its checks and thresholds |
| [docs/EXPORT.md](docs/EXPORT.md) | Images and the zarr export in detail |
| [docs/GEOMETRY.md](docs/GEOMETRY.md) | Frames, poses and how gripper width is measured |
| [docs/DESIGN.md](docs/DESIGN.md) | The commands, folder layout, the output zarr schema, conventions |
| [docs/INTAKE.md](docs/INTAKE.md), [docs/SIM.md](docs/SIM.md), [docs/ORBSLAM3.md](docs/ORBSLAM3.md) | Intake rules, the fake-data generator, the ORB-SLAM3 build |
