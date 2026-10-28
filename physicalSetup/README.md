# physicalSetup

C++17 code for the Jetson sensor rig that replaces the iPhone on the iPhUMI gripper: capture, live checks,
episode packages, the offline pipeline, and (later) policy deployment. Design: [docs/PIPELINE.md](docs/PIPELINE.md).
Plan and task ownership: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md). Progress:
[docs/STATUS.md](docs/STATUS.md).

**State (2026-10-28):** Waves 0–1 built, reviewed and committed.
- **Built:** IMU driver + simulator, camera backends + focus lock, capture/shm/package/config, camera models + image
  prep + `rigpy`.
- **Not built yet:** the recording programs (`ec-record`, `ec-calibrate`, `ec-viz`, `ec-slam-live`, `ec-doctor`:
  Wave 2) and the offline `ec-process` (Wave 3).
- **Apps today:** `ec-imu-probe`, `ec-cam-probe`, `ec-live` (live window: both cameras + IMU readout and plots).
- **Hardware:** the Argus backend is compile-checked against L4T r36.4.3 headers, but nothing has run on the Jetson
  yet. **When the hardware arrives:** [docs/HARDWARE_ARRIVAL.md](docs/HARDWARE_ARRIVAL.md).

Hardware: Jetson Orin Nano (JetPack 6.2 / L4T 36.4.3), Arducam IMX519 main camera, Arducam IMX219 160° ultrawide,
ST LSM6DSOX IMU (Modulino Movement) on I2C bus 7. Example config:
[episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml](episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml).

## Layout

```
physicalSetup/
  CMakeLists.txt  cmake/          top-level build, helpers (RigHelpers.cmake), version header template
  common/                         static library `rigcommon` (shared by collection and deployment)
    include/rig/                  PUBLIC, FROZEN interface headers (T0); see "Contracts" below
    <module>/                     sources of each module: time/ config/ drivers/{camera,imu,i2c}/ capture/ shm/
                                  camera_model/ imageprep/ package/ slam/ (+ python/ bindings, T4)
  episode_collection/             programs that run on the Jetson + offline pipeline
    include/ec/  record/ calib/ viz/ offline/   static library `rigec`
    apps/<name>/                  one executable per directory (ec-record, ec-calibrate, ec-imu-probe, ...)
    configs/                      rig.yaml examples
  deployment/                     placeholder (Wave 5), see deployment/README.md
  tests/<area>/                   GoogleTest; test_<name>.cpp -> executable test_<area>_<name>
  third_party/CLI11/              CLI11 v2.4.2 single header (BSD-3-Clause)
  scripts/check.sh                configure + build + ctest (the gate)
  docs/                           PIPELINE.md, IMPLEMENTATION_PLAN.md, STATUS.md, worklogs/
```

## Build

Dependencies (Ubuntu 22.04 laptop and JetPack 6.2 apt): CMake >= 3.22, Ninja, gcc 9 or 11, OpenCV 4 (core,
imgproc, imgcodecs, calib3d, features2d, highgui, videoio, objdetect; aruco optional), Eigen 3.3+, yaml-cpp,
spdlog, nlohmann_json, libjpeg, GoogleTest, GStreamer 1.0 dev (optional).

```bash
# Always build outside the repo (laptop: the session scratch dir).
physicalSetup/scripts/check.sh /path/to/build                              # default: RelWithDebInfo
physicalSetup/scripts/check.sh /path/to/build-asan --sanitize address,undefined
physicalSetup/scripts/check.sh /path/to/build-tsan --sanitize thread
physicalSetup/scripts/check.sh /path/to/build -- -DRIG_WITH_ARGUS=ON       # extra CMake args after --
physicalSetup/scripts/check.sh /path/to/build --jobs 4                     # laptop: cap jobs (several builds OOM)
physicalSetup/scripts/check.sh /path/to/build-py -- -DRIG_WITH_PYTHON=ON -DPython3_EXECUTABLE=<venv>/bin/python
```

Notes:
- `RIG_WITH_PYTHON` picks pybind11 from that interpreter (`python -m pybind11 --cmakedir`) and requires ≥ 2.12
  (numpy 2).
- TSan tests run under `setarch -R` (handled by `check.sh`).
- `RIG_T4_EXTENDED_DIR=<dir>` enables the extended real-frame image goldens; they are skipped otherwise.

Or by hand: `cmake -S physicalSetup -B <build> -G Ninja && cmake --build <build> && ctest --test-dir <build>`.

### Options

| Option | Default | Effect |
|---|---|---|
| `RIG_WITH_ARGUS` | OFF | libargus camera backend; needs `RIG_ARGUS_ROOT` (default `/usr/src/jetson_multimedia_api`). Sources in `**/argus/`. |
| `RIG_WITH_GST` | ON if GStreamer dev packages are found | GStreamer fallback camera backend. Sources in `**/gst/`. |
| `RIG_WITH_ORBSLAM3` | OFF | ORB-SLAM3 backends (Jetson). Needs `cmake/orbslam3.cmake` (T6) defining `rig_orbslam3`. Sources in `**/orbslam3/`. |
| `RIG_WITH_CERES` | OFF | Ceres-based stages. Sources in `**/ceres/`. |
| `RIG_WITH_PYTHON` | OFF | `rigpy` pybind11 module from `common/python/` (T4). |
| `RIG_SANITIZE` | "" | `address`, `undefined`, `thread`, or `address,undefined`. Adds `-fno-sanitize-recover=all`. |
| `RIG_BUILD_TESTS` | ON | Unit tests and the header self-sufficiency check. |
| `RIG_IGNORE_PATH_PREFIXES` | ON | Do not derive package prefixes from `$PATH` (keeps anaconda's GTest/yaml-cpp/json out). |

Every target gets `-Wall -Wextra -Wpedantic -Werror`. Third-party headers enter as SYSTEM includes. Compile
definitions `RIG_HAVE_{ARGUS,GST,ORBSLAM3,CERES,PYTHON}` (0/1) are PUBLIC on `rigcommon`.

## Adding code (conventions for every task)

- **No CMakeLists edits are needed** to add sources, tests or apps; everything is globbed (`CONFIGURE_DEPENDS`):
  - `common/<module>/**/*.cpp` -> `rigcommon`;
  - `episode_collection/{record,calib,viz,offline}/**/*.cpp` -> `rigec`;
  - `episode_collection/apps/<name>/*.cpp` -> executable `<name>` (links rigec, rigcommon, CLI11, spdlog, json, yaml-cpp);
  - `tests/<area>/test_<name>.cpp` -> gtest executable `test_<area>_<name>`; other `.cpp` files in the area are
    helpers linked into all tests of that area. Tests see `RIG_SOURCE_DIR`, `RIG_REPO_ROOT`, `RIG_TEST_TMP_ROOT`.
  - Files under a directory named `argus/`, `gst/`, `orbslam3/` or `ceres/` compile only with that option ON.
    Headers that need those dependencies are private (next to their sources), never in `common/include/rig/`.
- **Public headers** in `common/include/rig/` must be self-sufficient (checked by the generated header-check TUs).
- **Errors:** return `rig::Status` / `rig::Result<T>` (`rig/status.hpp`); no exceptions across module boundaries.
- **Time:** `rig::TimeNs` = CLOCK_MONOTONIC_RAW ns everywhere (`rig/time/clock.hpp`).
- **Style:** `.clang-format` (Google, 120 columns).

## Contracts (frozen in T0)

| Header | What | Implemented by |
|---|---|---|
| `rig/status.hpp`, `rig/span.hpp`, `rig/types.hpp` | Status/Result, Span, TimeNs, ImageFrame, ImuSample, SummaryStats | T0 (inline) |
| `rig/time/clock.hpp` | now_raw/mono/unix_ns, ClockAnchor, ClockBracket, ChipClockMapper | T0 (clocks), T1 (mapper) |
| `rig/drivers/i2c_bus.hpp` | I2cBus, make_linux_i2c_bus | T1 |
| `rig/drivers/imu_backend.hpp` | ImuConfig, ImuStats, ImuBackend, make_imu | T1 |
| `rig/drivers/camera_backend.hpp` | CameraConfig, SensorMode, CameraStats, CameraBackend, make_camera, select_sensor_mode | T2 |
| `rig/config/rig_config.hpp` | RigConfig (1:1 with rig.yaml), load/parse/to_yaml/content_hash | T3 |
| `rig/capture/stream_hub.hpp`, `capture_service.hpp` | StreamHub (Lossless/DropOk), CaptureService | T3 |
| `rig/shm/ring.hpp`, `messages.hpp`, `status_channel.hpp` | seqlock rings (ABI), wire formats, status PODs + typed channels | T3 (channels inline in T0) |
| `rig/package/package.hpp` | naming (iPhUMI parity), Manifest/StageRecord, EventLog, Session/Package | T3 |
| `rig/camera_model/*.hpp` | model kinds, CameraModel, RemapTable | T4 |
| `rig/imageprep/imageprep.hpp` | letterbox_square_resize (cv_util.py parity), ImagePrep | T4 |
| `rig/slam/slam_backend.hpp` | SlamSettings, SlamBackend, make_slam_backend | T6 |

Additive public headers added in Wave 1 (each owned by its task):
- `rig/drivers/imu/*` (T1);
- `rig/drivers/camera/*` (T2: sensor modes, frame tracker, sensor clock, capture plan, IMX519 focus, probe report);
- `rig/package/{layout,csv,atomic_file,sha256}.hpp`, `rig/shm/publisher.hpp`, `rig/capture/thread_config.hpp` (T3);
- `rig/camera_model/coverage.hpp`, `rig/imageprep/color.hpp` (T4).

Additive fields:
- `FrameMeta::color_range`, `CameraStats` (+4 fields) (T2);
- `ImuConfig`/`ImuStats`/`ChipClockConfig` fields (T1; yaml keys by T3).

Changing a frozen public declaration requires a note to the orchestrator (IMPLEMENTATION_PLAN section 0). Adding a
defaulted field to a config/stats struct, or a new header in one's own module, is additive: record it in the task
WORKLOG. Private members (pimpl `Impl`, private data) belong to the implementing task.
