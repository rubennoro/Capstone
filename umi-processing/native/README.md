# native: UMI's ORB-SLAM3 fork, our patches, and the `umiproc_slam` front-end

umiproc runs UMI's SLAM recipe with UMI's own ORB-SLAM3 fork, [cheng-chi/ORB_SLAM3](https://github.com/cheng-chi/ORB_SLAM3)
at `b741dca`, built natively (no Docker) on the laptop and the Jetson. This folder holds what we add to it:

| Path | What |
|---|---|
| `build.sh` | clones and pins the fork, applies `patches/`, builds everything (laptop and Jetson) |
| `patches/0001…0012-*.patch` | our fixes to the fork, applied in order (each file starts with what, why and the behaviour change) |
| `umi_slam/umiproc_slam.cc` | the front-end: runs the fork on a umiproc EuRoC folder (the header comment documents it) |
| `umi_slam/CMakeLists.txt` | builds the front-end against the fork |

The Python side is `umiproc/slam/`: `runner.py` runs the binary, `log.py` parses its output, `trajectory_csv.py`
reads the trajectory.

## The patches

All twelve apply with `git apply` to a clean `b741dca`, in order. Each one also applies (or reverse-applies) on its
own, which is how `build.sh` tells "not yet applied" from "already applied".

| # | File | Fix | Behaviour change |
|---|---|---|---|
| 0001 | Tracking.cc | When a map is loaded, its last keyframe is put one frame period before t = 0. That period was hard-coded to 1/59.97 s (GoPro); it is now `1.0 / Camera.fps`. Also removes a dead block that re-declared `dt` in an inner scope (the preintegration's dT was never used). | Gap of 1/30 s instead of 1/60 s |
| 0002 | System.cc | `SaveTrajectoryCSV` crashed when no map had a keyframe (`pBiggerMap` uninitialised). It now writes the header only: every frame lost. | Only in that case |
| 0003 | Tracking.cc | After every successful relocalization against a loaded map (`INIT_RELOCALIZE`), the new keyframe gets no IMU preintegration. Before, it copied the running preintegration (IMU since the start of the run, or since the keyframe before the loss), and the next keyframe then shared that same object. So the optimiser linked the relocalized keyframe to the previous keyframe through IMU from the wrong interval. Integration restarts from the relocalized keyframe. The optimisers already skip a keyframe without a preintegration (the fork does exactly that when relocalization succeeds on frame 0). | Localization runs that relocalize after frame 0 |
| 0004 | KannalaBrandt8.cpp | The tag-based initialisation estimated the table tag's pose with uninitialised distortion coefficients, although the corners were already undistorted. They are now zeros. | Tag initialisation, when that memory was not zero |
| 0005 | Tracking.cc | `mnFramesToResetIMU` was set only by the legacy settings parser. With `File.version "1.0"` settings (ours and UMI's) it was indeterminate: 0 on the laptop, 3 of 3 runs. It is now `Camera.fps`, the value ORB-SLAM3's older settings reader sets. **A deliberate difference from UMI** (decided 2026-11-16): UMI's runs most likely had 0. | Yes: 1 s of visual-only tracking after each relocalization or map reset. On the existing demos this is what made localization hold (20/362 → 340–362 tracked frames on the same map) |
| 0006 | System.h/.cc | Adds a read-only `System::MapSummary()`: per map, the id, keyframes, map points, whether the IMU is initialised, and whether VIBA 1 and VIBA 2 have run. Used by checks M1 and M2. | None |
| 0007 | Tracking.cc | Before a localization run's first relocalization there is no keyframe of the run yet, so the relocalized keyframe (and the IMU integration from it) started from a zero IMU bias, with nothing tying it to the map's. Frames waiting for that first relocalization now carry the bias of the map's last keyframe. | Localization runs |
| 0008 | Tracking.cc | After a relocalization, `mnLastKeyFrameId` held a keyframe id; everywhere else it is a frame id, and `NeedNewKeyFrame` compares it with frame ids. Localization frame ids continue from the mapping run (733+), keyframe ids are the map's (100–180). | Small: keyframe insertion right after a relocalization, mostly on small maps |
| 0009 | Tracking.cc, Tracking.h | The loaded map's keyframe times were shifted on every relocalization, the run's own keyframes included. That made the CSV's `is_keyframe` stop matching and inflated LocalMapping's `mTinit` (VIBA 2 ran inside a 12 s demo). Now they are shifted only on the first relocalization. | Localization runs that relocalize more than once |
| 0010 | LocalMapping.cc | When Tracking starts a new map while LocalMapping still has keyframes of the old map queued, the points triangulated from them went into the NEW map (`mpAtlas->GetCurrentMap()`), with observations only in the old one. The new map's first IMU initialisation (`FullInertialBA`) then dereferenced a NULL keyframe vertex: the SIGSEGV of the episode-128 map run. Points now go to their keyframe's map. | Only keyframes processed after a new map was started |
| 0011 | LocalMapping.cc | `InitializeIMU` (VIBA 1/2, seconds long) ended with `mpTracker->mState = OK` unconditionally. If Tracking had started a new map meanwhile, that forced it to track an empty map with no reference keyframe (`SearchByBoW(NULL)`, SIGSEGV; seen once under gdb on episode 128). Now only if the atlas's current map is still the one being initialised. | Only when a new map is started during an IMU initialisation |
| 0012 | System.h | Adds a read-only inline `System::GetAtlas()` (and the macro `UMIPROC_SYSTEM_GET_ATLAS`): the atlas pointer, for `umiproc_slam --save_points` after `Shutdown()`. It replaces the front-end's earlier read of the private `mpAtlas` through the explicit-instantiation access idiom. A front-end built against a variant without it refuses `--save_points` (exit 1). | None |

0001 and 0009 are exported with 1 line of context, because 0009 adds lines right after 0001's.

## Build

```bash
umi-processing/native/build.sh
```

Results (default `ORB_SLAM3_UMI_DIR=~/ORB_SLAM3_umi`):

- `~/ORB_SLAM3_umi/umiproc_build/umiproc_slam`: the binary umiproc runs.
- `~/ORB_SLAM3_umi/Vocabulary/ORBvoc.txt`: the vocabulary.
- `~/ORB_SLAM3_umi/umiproc_build/BUILD_INFO`: fork and Pangolin commits, a hash of each patch, the OpenCV used, the
  machine.
- `~/ORB_SLAM3_umi/umiproc_build/logs/`: one log per step. On a failure, the script prints the tail of the failing
  step's log.

What it does, in order:

1. Checks that the tools and libraries are there, and names the packages to install if not.
2. Chooses an OpenCV. The fork needs the contrib `aruco` module with the pre-4.7 API (`DetectorParameters::create()`,
   `Ptr<Dictionary>`). The system OpenCV is used when it has that. Otherwise the script builds OpenCV 4.5.4 + contrib
   (the laptop's apt version, pinned by commit, only the modules the fork uses) into
   `Thirdparty/opencv-4.5.4/install`.
3. Clones the fork at `b741dca`, or checks that an existing checkout is at that commit (it never moves one).
4. Fetches Pangolin `d484494` over https (the fork's `.gitmodules` has an ssh URL).
5. Applies each patch, or confirms it is already applied. It stops if a patch does neither, or if the fork has
   changes in files no patch touches.
6. Builds DBoW2, g2o, Sophus (no tests or examples) and Pangolin (in place; no examples, tools, Python or FFmpeg).
7. Unpacks the vocabulary.
8. Builds the fork's library (Release, its `-march=native`) and then the front-end.
9. Checks that the binary's libraries resolve and that `--help` runs.

Running it again only rebuilds what changed (a few seconds when nothing did).

Options (environment variables):

- `JOBS`: parallel compile jobs. Default: 2 below 10 GB of RAM (the Jetson), else RAM/2 GB, capped at the core
  count.
- `ORB_SLAM3_UMI_DIR`: build somewhere else.
- `UMIPROC_OPENCV=auto|system|build`: force the OpenCV choice.
- `PATCHES_UPTO=NNNN`: build an A/B variant with only patches 0001…NNNN (see below).
- `OpenCV_DIR`: with `system`, which OpenCV to use.

### Laptop (Ubuntu 22.04, x86_64)

```bash
sudo apt install build-essential cmake git libeigen3-dev libgl1-mesa-dev libopengl-dev libglew-dev \
    libboost-serialization-dev libssl-dev libopencv-dev libopencv-contrib-dev
umi-processing/native/build.sh        # uses apt's OpenCV 4.5.4 (has contrib aruco)
```

### Jetson Orin Nano (JetPack, aarch64)

```bash
sudo apt install build-essential cmake git libeigen3-dev libgl1-mesa-dev libopengl-dev libglew-dev \
    libboost-serialization-dev libssl-dev
umi-processing/native/build.sh        # first run: builds its own OpenCV 4.5.4 + contrib (most of the time)
```

How the Jetson differs from the laptop:

- **OpenCV.** JetPack's OpenCV (4.8 on JetPack 6) cannot build the fork. Its aruco API is the 4.7+ one:
  `getPredefinedDictionary` returns a `Dictionary`, and `DetectorParameters::create()` is gone. It most likely has
  no contrib `opencv2/aruco.hpp` either. So `build.sh` builds OpenCV 4.5.4 + contrib under the fork folder, the same
  version as the laptop, so both machines run the same OpenCV code. JetPack's OpenCV stays installed and untouched.
  The fork's libraries find the private one through their RUNPATH. Expect roughly 30–60 min with 2 jobs and about
  1.5 GB of disk.
- **Pangolin.** The Jetson has Pangolin v0.6 in `/usr/local` (for the upstream ORB-SLAM3 at `~/ORB_SLAM3`).
  `build.sh` points every CMake step at the fork's own Pangolin (`-DPangolin_DIR=…`), and the front-end's
  `find_package` ignores default paths, so the two never mix. `~/ORB_SLAM3` itself is not touched.
- **Memory.** 7.3 GB of RAM plus 8 GB of swap, hence 2 jobs (some fork files need about 2 GB each to compile). If
  the compiler is killed (out of memory), use `JOBS=1`.
- **`-march=native` (the fork's flag).** Each machine builds its own binaries; do not copy them between machines.
  GCC on JetPack 5 and 6 supports `-march=native` on aarch64.
- **Time.** The first build takes about an hour, mostly OpenCV. A rebuild after a patch change takes minutes.

Not run on a Jetson yet: the script was run here on the existing checkout (system OpenCV). Its private-OpenCV path
was run here on a fresh clone with `UMIPROC_OPENCV=build` (see the step's report).

### A/B variant: only some of the patches

To compare builds on the same recordings (for example the patches up to 0006 against all of them), build a variant:

```bash
PATCHES_UPTO=0006 umi-processing/native/build.sh
```

- **Output:** `~/ORB_SLAM3_umi/umiproc_build_0006/`, with its own `umiproc_slam`, `BUILD_INFO` and `logs/`.
- **Sources:** its own copy of the fork's sources in `fork/`, a git worktree at `b741dca` with only 0001–0006
  applied.
- **Separation:** its own `fork/lib/libORB_SLAM3.so`. The main build is never touched. DBoW2, g2o, Pangolin,
  OpenCV and the vocabulary are shared, since none of them is patched. `build.sh` checks with `ldd` that each binary
  loads its own `libORB_SLAM3.so`. Both binaries find their library through RUNPATH, so do not set
  `LD_LIBRARY_PATH` to either `lib/`.
- **Cost:** a full compile of the fork (about 3 min on the laptop, roughly 20–30 min on the Jetson) and about
  65 MB of disk.

Run umiproc with it in one of two ways:

- set `UMIPROC_SLAM_BINARY=~/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam` (read by `umiproc.slam.runner`);
- or pass `binary=` to `run_umi_slam`.

Maps saved by one build can be loaded by the other: no patch changes the map file format.

Remove the variant with:

```bash
rm -rf ~/ORB_SLAM3_umi/umiproc_build_0006 && git -C ~/ORB_SLAM3_umi worktree prune
```

## Checking the build

```bash
~/ORB_SLAM3_umi/umiproc_build/umiproc_slam --help        # the options
cat ~/ORB_SLAM3_umi/umiproc_build/BUILD_INFO             # commits, library, patch hashes, OpenCV
git -C ~/ORB_SLAM3_umi diff --stat                       # exactly the files the patches touch
cd umi-processing && env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_slam_*.py -q
```

The last command includes `tests/test_slam_real_binary.py` (marked `slow`, about 1.5 min). It maps a real recording,
then localizes a second one against that map, and checks that the summary and the trajectory line up with the
inputs. It is skipped when the binary or the recordings are absent. Point it elsewhere with
`UMIPROC_TEST_EUROC_MAP`, `UMIPROC_TEST_EUROC_LOC` and `UMIPROC_TEST_MASK`.

## Running it by hand

```bash
B=~/ORB_SLAM3_umi
$B/umiproc_build/umiproc_slam -v $B/Vocabulary/ORBvoc.txt -s SEQ/orbslam3_settings.yaml -i SEQ \
    -o OUT/trajectory.csv --save_map OUT/map.osa --mask_img mask.png --init_tag_id 13 --init_tag_size 0.16  # map
$B/umiproc_build/umiproc_slam -v $B/Vocabulary/ORBvoc.txt -s SEQ2/orbslam3_settings.yaml -i SEQ2 \
    -o OUT2/trajectory.csv --load_map OUT/map.osa --mask_img mask.png                                    # localize
```

`--save_points OUT/map_points.csv` (needs patch 0012) also writes the saved map's points, keyframes
(`map_points_keyframes.csv`) and `map_points_meta.json` (gravity up) in the trajectory's world, for `umiproc slam-3d`;
with `--load_map` and `--max_frames 0` it exports a saved atlas without feeding a frame (the header comment of
`umiproc_slam.cc` has the format).

The last line of the output is `UMIPROC_SUMMARY {json}` (format `umiproc.slam_summary.v2`). Its `pacing` object
says whether the machine kept up with the camera: `track_ms` (median, p95, max of the per-frame tracking time),
`frames_over_period` (frames whose tracking took longer than their frame period), `feed_rate_hz` against
`recording_rate_hz`, and `feed_wall_s`; the line before it prints the same in words. Exit codes:

- 0: finished.
- 3: stopped at `--max_lost_frames`; the trajectory is still saved.
- 1: bad input.

After a non-zero exit, the `--save_map` file is removed.
