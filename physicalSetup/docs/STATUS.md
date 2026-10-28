# physicalSetup status

The single source of truth for implementation progress. Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Pause
and resume protocol: plan section 8. Design: [PIPELINE.md](PIPELINE.md).

Last gate commit: see `git log` (Wave 0 = T0, Wave 1 = T1–T4).

Wave 1 gate (2026-10-28, sequential, `--jobs 4`): default 397/397, ASan+UBSan 397/397, TSan 398/398 (+ `FromRigConfigWithMockBackends` ×5), Python 398/398. The orchestrator also fixed a pybind11 build trap: CMake now takes pybind11 from the chosen interpreter and requires ≥ 2.12, because the system 2.9 breaks with numpy 2. Sensors arrive 2026-10-29.

## Tasks

| Wave | Task | State | Worklog | Notes |
|---|---|---|---|---|
| 0 | T0 build skeleton + frozen interfaces | done | [worklogs/T0.WORKLOG](worklogs/T0.WORKLOG) | gate passed 2026-10-27: 14/14 tests default, ASan+UBSan, TSan |
| 1 | T1 LSM6DSOX driver + simulator | done (R1 verified) | [worklogs/T1.WORKLOG](worklogs/T1.WORKLOG) | executor done: 61/61 IMU tests default/ASan+UBSan/TSan; clock map ≤ 28 µs under ±2 % drift; awaiting R1 |
| 1 | T2 camera backends + IMX519 focus | done (R1 verified) | [worklogs/T2.WORKLOG](worklogs/T2.WORKLOG) | executor done: 375/375 whole tree default/ASan+UBSan/TSan; Argus compile-checked against r36.4.3 headers; awaiting R1 |
| 1 | T3 time/config/capture/shm/package | done (F17–F21 fixed; gate verified) | [worklogs/T3.WORKLOG](worklogs/T3.WORKLOG) | executor done: 196/196 T3 tests in default/ASan+UBSan/TSan; whole tree 369/369 default; awaiting R1 |
| 1 | T4 camera models + image prep + bindings | done (R1 verified) | [worklogs/T4.WORKLOG](worklogs/T4.WORKLOG) | executor done: bit-exact vs iPhUMI cv_util + camera_model.py; 188/188 default, 194/194 with Python; awaiting R1 |
| 1 | R1 reviewer | done: GO after F21 (2026-10-28) | [worklogs/R1.WORKLOG](worklogs/R1.WORKLOG) | started 2026-10-28 |
| 2 | T5 ec-record | todo | | |
| 2 | T6 ec-viz + ec-slam-live | todo | | |
| 2 | T7 ec-calibrate + tag detector | todo | | |
| 2 | T8 bring-up kit + ec-doctor | todo | | |
| 2 | R2 reviewer | todo | | |
| H | hardware day H1–H7 | todo | | |
| 3–5 | T9–T12, calibration, deployment | todo | | |

## Open reviewer findings

All Wave 1 findings are closed. F1–F16 were verified by the R1 re-check; F17–F21 were fixed by T3 and verified at the gate; F22 was fixed by the orchestrator.

### Earlier R1 findings (all fixed)

R1 results: all configs 375/375; 300 s integration run clean. Verdict: not ready until the majors are fixed.

| id | sev | owner | defect |
|---|---|---|---|
| F1 | major | T3 | backends are started on the caller's thread, so the IMU poll and Argus threads are never pinned or prioritised |
| F2 | major | T1 | the sim keeps truth records forever (~190 MB/h) |
| F3 | major | T2 | the clock resolver rejects slightly negative latency, so start() can fail on r36 |
| F4 | major | T1 | the probe rate check ignores the FREQ_FINE trim, a false H1 fail |
| F5 | minor | T1 | `--record DIR` doesn't create DIR |
| F6 | minor | T1 | probe sample-meta CSV name/columns ≠ layout.hpp |
| F7 | minor | T3 | the new IMU fields have no yaml keys / hash |
| F8 | minor | T3 | poll budget assumes 2 words per ODR (3 with timestamps) |
| F9 | minor | T1 | yaml accel noise is the ±2 g value; should be 80 µg/√Hz |
| F10 | minor | T2 | `--focus-sweep` runs on mock / misreports |
| F11 | minor | T2 | TSan suppressions too broad (`race:libgst*`) |
| F12 | minor | T2 | the mock doesn't count drops when behind real time |
| F13 | nit | T1+T3 | "DS13345" references remain |
| F14 | nit | T3+T2 | `frames/` isn't EuRoC `data/`; rename to `data/` |
| F15 | nit | T3 | duplicate sensor-mode table; use T2's `known_sensor_modes()` |
| F16 | rec→do | T2+T4 | colour path: Argus NV12_ER (full range) + FrameMeta colour-range flag + imageprep honours it |

## Items for R1 / later tasks (from executor reports)

- T4: live NV12 → BGR uses BT.601 limited range, while stored JPEG decodes as full range, so training and deployment colours can differ. Decide on one deployment colour path (Wave 5 / R1).
- T4: the main camera is stored at 1280x720 while captured at 1920x1080. Deployment must downscale exactly as the recorder does (shared function).
- T4: possible aarch64 FMA ±1 ULP in the remap maps. Build the maps on the machine that uses them; verify on H.
- T4: ASan run showed T3's `ShmRingMultiProcess` failing while T3 was still in progress; T3 reports it passing now. R1 to confirm.
- T3: whole-tree sanitizer failures are T2's `GstCamera.*` (a 16 KiB glib leak under ASan) and `ReplayTest.*` (TSan). T2 was still running at the time.
- T3: cross-check T3's nominal sensor-mode tables (`common/config/sensor_modes.cpp`) against T2's. Both are unverified until H.
- T3 layout choices to confirm in R1: EuRoC headers verbatim; `chip.csv` = clock brackets; per-sample chip ticks in an optional `imu0/sample_meta.csv`.
- T1: AN5272 could not be downloaded. The user should fetch it by hand (browser) so R1 can confirm: the timestamp trim formula 25 µs/(1+0.0015·FREQ_FINE); burst-read wrap 0x7E→0x78; timestamp-word ordering + TAG_CNT pairing; temperature-word layout; FIFO size 512 words. Scratch downloads are in `$SCRATCH/t1dl/`.
- T1: the datasheet is **DS12814** (the plan's "DS13345" is wrong). Fix the docs.
- T1 → T3: the poll-budget check assumes 2 words per ODR; with timestamp batching it is 3. Fix in T3's config validation.
- T1: rig.yaml accel noise at ±8 g = 80 µg/√Hz (yaml has the ±2 g figure, 70).
- T1 → T5/T3: call IMU `start()` from the pinned thread (the poll thread inherits affinity and priority).
- T1 → BRINGUP: leave INT1 unconnected (INT1 high at power-up = I3C-only mode); i2c-7 must run ≥ 400 kHz; Tegra must accept 224-byte reads.
- T1's whole-tree ASan run showed 1 `ImagePrep` failure, although T4 reported all-pass. R1 to reproduce (possibly concurrent builds or fixture paths).
- T2 vs T3: T3's mode table marks IMX519 1920x1080 as not full-FOV, which would make the configured main mode unselectable. The tables must agree (R1).
- T2: IMX519 1280x720 is a crop (~2/3 the width of 1080p). Capture 1920x1080, store 1280x720; PIPELINE §3/D5 to be updated after R1.
- T2: `getSensorTimestamp` (r36.4.3) = first data from the sensor, so the code uses t_sof = ts − exposure and t_mid = ts − E/2 + R/2 (deviates from the plan's formula; R1 to confirm against the header).
- T2: the Argus timestamp clock domain is unsettled (MONOTONIC_RAW vs REALTIME in NVIDIA forum replies). `SensorClockResolver` picks per frame; verify on H.
- T2 → T5: Replay reads `frame_meta.csv` by column name. It must match T3's `layout.hpp` headers and what T5 writes (R1).
- T2: CAM0/CAM1 I2C bus numbers (9/10) are guesses; verify on H.
- T4 → T5: RawWriter must JPEG-encode straight from the full-range NV12 planes (libjpeg raw-data input, 4:2:0, JFIF), never via BGR, so live `frame_to_bgr` matches decoded training images (q90: mean 1.2 LSB, max 19).
- R1 re-check: T1's `DriverFaults.HostQueueOverflowCounted` failed once under load (T4's Python run). Check it for timing flakiness. T4 added an ASan-only LSan suppression for libtbb (system OpenCV's thread pool) in the imageprep test helper; confirm it's narrow.
- R1 re-check (T3): the accel_noise test assertion was weakened to "> 0" instead of the exact 7.845e-4; the exact value is preferred. `FromRigConfigWithMockBackends` skips the steady-state assertion under TSan (mock under TSan ~10 fps); judge whether that's OK. One unreproduced TSan flake in `StreamHub.StressManyConsumersBothPolicies` (load ~24, maybe memory).
- R1 re-check (T2): F11 narrowing revealed a real GstCamera start/callback ordering race. It was fixed with a hand-over plus TSan annotations, and a TSan death test proves injected races are caught. Verify the annotations don't paper over a real race. Argus BGR/RGB output copies via limited NV12 but frames report kFull. Check the `color_range` semantics for already-converted BGR frames.
- T2 → T5: tests needing exact sequences must set the mock's `driver.buffer_count` large (e.g. 256).
- T3 → T5: subscribe lossless consumers before `start()`; T5 owns the shm publishers and the CaptureStatus channel.

## Contract notes from T0 (binding for T1–T4)

- Factories take the config: `make_camera(cfg)`, `make_imu(cfg, bus=nullptr)`; `start()` takes no argument.
- Sources, tests (`tests/<area>/test_*.cpp`) and apps (`apps/<name>/`) are globbed, so no CMakeLists edits are needed. `argus/`, `gst/`, `orbslam3/` and `ceres/` dirs build only with their option on.
- Session names are `[A-Za-z0-9-]+` (iPhUMI group.py:36-37 splits on `_`); the package kind is `rigcalibration`.
- `ChipClockMapper` (rig/time/clock.hpp) is implemented by **T1**, not T3.
- Adding a field with a default, or a new header in your own module: worklog note. Changing a frozen declaration: tell the orchestrator.

## Next actions

1. ~~T0~~ done. T1–T4 executors done (2026-10-28).
2. ~~Wave 1~~ done and committed (2026-10-28).
3. **Waiting for the user to confirm Wave 2** (T5–T8 in parallel → R2 → gate → commit). The Wave 2 briefs must carry the T5/T6/T8 notes from "Items for R1 / later tasks" (JPEG from full-range NV12 planes; lossless subscribers before start(); mock `buffer_count`; INT1 unconnected; root/rtprio for SCHED_FIFO; i2c-7 ≥ 400 kHz).
4. The user may fetch AN5272 (LSM6DSOX app note) by hand to settle T1's unverified FIFO details.
5. **Hardware arrival directions:** [HARDWARE_ARRIVAL.md](HARDWARE_ARRIVAL.md) (what works today vs after Wave 2).
6. The open todos (Wave 2 go-ahead, AN5272, Kalibr discussion, hardware-day checks, laptop disk) are tracked in the repo-root [TODO.md](../../TODO.md), section "physicalSetup/".

No agents are running (T4 was stopped 2026-10-28). Nothing is left in the scratch build dir.
