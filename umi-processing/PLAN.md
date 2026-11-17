# umiproc: plan

The work to get from the Pi rig's recordings to checked task episodes with UMI's SLAM recipe. How it works:
[docs/DESIGN.md](docs/DESIGN.md). How to use it: [DIRECTIONS.md](DIRECTIONS.md). Update the status column as steps land.

## Goal

On the Jetson, `umiproc map` → `umiproc calibrate` → `umiproc run` turns one session of recordings from
`/data/incoming` into `~/data/umi/tasks/TASK/episode_N/` folders that passed every check and are ready to package.
Packaging itself comes later.

## Decisions (2026-11-16, with the user)

| Topic | Decision |
|---|---|
| Code | New folder `umi-processing/` (package `umiproc`), self-contained: whatever it uses from `data-processing/` is copied into `umiproc/dp/` (never imported) |
| Where | Jetson; reads `/data/incoming` (read only), writes `~/data/umi/` with its own list of processed recordings |
| Session order | UMI's: map → gripper calibration → demos. `run` without the gripper calibration still localizes and checks, then waits for it |
| SLAM | UMI's ORB-SLAM3 fork (`b741dca`) with our bug-fix patches, built natively; real per-frame timestamps; frames fed at camera speed |
| Gripper mask | UMI's exact mechanism (pixels set to 0 in every frame fed to ORB-SLAM3, mapping and localization) with our traced polygons |
| Table tag | UMI's tag 13 (0.16 m): start-up fallback while mapping, and a map check |
| Lost frames | UMI's rules, including splitting a demo at its gaps, converted to the same time at 30 fps: stop after 30 lost, > 5 lost → drop, < 30 tracked → drop, pieces < 12 frames dropped |
| Quality | No shortcuts: fix only real bugs or mismatches, never relax a check to make data pass |
| Sensor values | Lens: optional 1-minute ChArUco calibration (never blocks map/run; nominal otherwise, labelled). Camera→IMU: CAD. Time offset: 0 (measured offset reported as a diagnostic). IMU noise: datasheet (accel 75 µg/√Hz at ±4 g), random walks documented as assumptions |
| Checks | UMI's, plus the map checks UMI lacks (M1–M5) and the copied validate checks (docs/DESIGN.md "Checks") |
| Execution | Executor agents + one reviewer per step, a local commit per step, no push without asking |

## Steps

| Step | What | Who | DIRECTIONS.md | Status |
|---|---|---|---|---|
| Setup | Folder skeleton, `docs/DESIGN.md`, this plan, `configs/slam/rig_pi_v3.yaml`; copy of dataproc into `umiproc/dp/` + `tests/dp/` + `configs/rigs/`, tests passing in an environment without dataproc; front-end moved to `native/umi_slam/` | me | outline | done (`6187da7`) |
| 0 | `umiproc lens-board` (printable board PDF) and `umiproc lens-calibrate` (fit, checks, lens file); tested on a synthetic board recording | agent + review | §1 Lens calibration | done (reviewed, fixed) |
| 1 | `native/`: fork patches 0001–0011 (0007–0009, the relocalization fixes, A/B on the real session; 0010–0011, two upstream crashes when a second map starts), `build.sh` for laptop and Jetson; data root `~/data/umi`, organization (map / calibration / demonstration kinds), rig + SLAM config + lens file loading, ORB-SLAM3 settings writer, mask PNG, SLAM runner and log/CSV parsing; `umiproc status` | agents + review | §2 Jetson setup | done (reviewed, fixed); not yet built on the Jetson |
| 2 | `umiproc map`: creates the task folder, builds the map, re-localizes the mapping clip, table-tag pose, checks M1–M5 with hints | agent + review | §3 Prove the environment (mapping clip, `map`, reading the checks, what to change) | built, reviewed, fixes in. Real mapping clip 128: nominal lens = 34 resets (fails); with an experimental measured lens = a 31 s map with the IMU fully initialised, then lost track (3 maps, M1 fails) and an intermittent SLAM crash when a second map starts (fixed: patches 0010–0011). Needs a passing lens calibration |
| 3 | `umiproc run` (localization, UMI's lost-frame rules and splitting, pose checks, widths, `episode.zarr`, waiting for gripper calibration) and `umiproc calibrate` | agents + review | §4 Prove the demos, §5 Gripper calibration and finishing | to do |
| 4 | Smoke test on the existing recordings (expected: `map` fails with clear messages), fixes, final review | agents + me | §3–5 adjusted | to do |
| Real test | You record a session (mapping clip, gripper calibration, 1–3 demos) following DIRECTIONS.md §3–5. The first mapping clip is run twice, as is and with `imu.vio_inflation: {noise_density: 10, random_walk: 10}` in the data-root rig copy, compared on checks M2–M4 alongside the 0006-vs-0009 patch comparison (docs/DESIGN.md "SLAM") | you | — | waiting on steps 2–3 |

## Later (not now)

- Packaging the episodes (the copied `umiproc/dp/package` is a starting point).
- Camera→IMU and time-offset calibration with a board (OpenICC), and the IMU's random walks from a long still recording.
- Deployment reading the same rig and lens files (the ultrawide image is undistorted with the lens model).
