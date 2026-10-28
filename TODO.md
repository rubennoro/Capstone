# TODO: Jetson sensor rig → trained policy → Franka

Ordered roughly by dependency. Big picture: [ROADMAP.md](ROADMAP.md) ("Status"). What to buy: [PURCHASING.md](PURCHASING.md).
All software below is built and tested with mocks; nothing has run on real hardware yet.

## Decisions (blocking)

- [ ] **D1. Panda or FR3?** Pins libfranka/panda-py (`controlSW/franka_deploy/scripts/build_libfranka_pandapy.sh --robot`); `check-config --config deploy.yaml` (no `--profile`) fails until set. Record the robot system version from Desk.
- [ ] **D2. Gripper actuator:** (a) Franka Hand with printed fingers: what the sim evaluator, profile `real-iphone` and `hardware_day.md` §11-13 assume (needs finger adapters, the width map, `hand_server`); or (b) the servo-driven iPhUMI gripper, Feetech STS3215 (recommended, cheap) or Dynamixel XM430 (PURCHASING §2). Undecided; the docs present both.
- [x] **D3. Cameras/lenses:** main = Arducam B0180 + 2.1 mm low-distortion lens; ultrawide = Arducam B0179 (175° fixed fisheye, IR-cut version). Decided 2026-10-19 (PURCHASING §1).
- [ ] **D4. Where the Franka 1 kHz loop runs:** the Jetson first, gated by the go/no-go load test; a small RT PC is the fallback.
- [ ] **D5. Evaluation settings:** v_max for reported results (0.25 vs 0.5 m/s), the success rule (lift 3 cm here, 5 cm in `hardware_day.md` §10 and `criteria/block_raise.yaml`, held ≥ 2 s, 60 s timeout), which prompt indices, start-pose buckets, N ≥ 20 per condition.
- [ ] **D6. Commit?** Uncommitted: W10B (sim TCP derivation, judge/eval fixes), newer `sensor_rig/`, `scripts/regression/`. Run the regression guard first (`scripts/regression/README.md`).
- [ ] **D7. Powering the handheld rig:** tether or battery (PURCHASING §1).
- [ ] **D8. Pending upstream real-env fixes** (ROADMAP §0, `sensor_rig/docs/deployment.md`): stay unapplied (franka_deploy works around them) unless you approve them.
- [ ] **D9. Franka Hand tooling geometry:** printed fingers about 10 cm longer than stock, iPhone on the -y side (sim Stage 1: 43/45 demos feasible on Panda and FR3; stock length hits links 5/6). With the real UMI finger kit (A9, 2026-10-28) the sim recommends an 8 mm jaw adapter, phone -y: 43/45 on the Panda, +39 mm clearance; but the kit's tag positions do not match the training video yet and its Stage 1b lifts regress (sim TODO: `controlSW/docs/sim_plan/TODO.md` section 1). Confirm before printing; sets `tooling.finger_adapter.extension_m` / `tcp_x_along_hand` (still 0.0 / "+y"). Not yet re-run with the 30 mm-thick rig housing. Caveats (A3): 37/45 demos leave the 60 degree bridge cone (the SafetyFilter would clamp them); joint speeds within limits for only 19/43 FR3 and 36/43 Panda.
- [ ] **D11. Is the real TCP the pad centre?** Station `grasp_tcp_height_m` 0.01 puts the 30 mm pads 5 mm into the table if so; the sim uses 0.018. If yes, set 0.018. Also: ~0.5 m/s approaches hit the table at ~40 N per finger and trip the 20 N reflex (real): keep v_max <= 0.25 and do not raise the reflex threshold; do not gate on the sim's `slip` metric until A7 (it measures arm overshoot today).
- [ ] **D10. Open sim-evaluator gates:** Stage 2b live-vs-offline p50/p95/n_pairs, Stage 3 `obs_motion` 3 mm (arm speed x image age, 5-9 mm at demo speeds; `--match-camera-latency` is off by your decision), `/home` work_root on the laptop (99 % full).

## Purchasing

- [ ] Order PURCHASING §1–3 (check stock and compatibility first; verify the IMU breakout exposes SPI, and the M12 lenses fit the B0180 holder).
- [ ] Print and mount the Aprilgrid and ChArUco boards; measure the real tag/square sizes.
- [ ] Design and print the camera + IMU bracket with **lens centres within ±3 mm of the iPhone 16 Pro lens positions**, the gripper servo coupling, and the Franka wrist mount.

## Software you can do now (no hardware)

- [ ] Build the ORB-SLAM3 image (`sensor_rig/docker/orbslam3/`) on a machine with ≥ 20 GB free. The laptop disk is ~90 % full; free space or use an external SSD.
- [ ] Run `sensor-rig process-session --vio orbslam3-docker` on a simulated episode, and measure accuracy against the sim ground truth (T4 criterion c, still open).
- [ ] Map-then-localize for ORB-SLAM3 (UMI-style), plus a mask-capable build that uses the finger mask (docs: `sensor_rig/docker/orbslam3/README.md`).
- [ ] Resolve the VERIFY-marked URLs and versions in `controlSW/franka_deploy/scripts/` (RT kernel apt line, JetPack 6 torch wheel, libfranka/panda-py versions) once D1 is decided.
- [ ] Add the regression-guard paragraph to DIRECTIONS.md (drafted in `scripts/regression/README.md`).
- [ ] Optional: a sim-rig 30 fps dataset from all 45 `firstreal` demos → `process_dataset.sbatch` (its first real run) → training vs an iPhone baseline (ROADMAP A2; needs cluster approval).

## Hardware day 1: data collection — follow [sensor_rig/docs/first_plugin.md](sensor_rig/docs/first_plugin.md)

- [ ] Flash JetPack 6 to the NVMe; jetson-io: enable dual IMX219 + SPI1; run `sensor_rig/scripts/jetson_setup.sh`.
- [ ] `sensor-rig doctor` all PASS (both cameras, IMU WHO_AM_I and gravity, disk, clock/NTP).
- [ ] `sensor-rig bench-encode`: no drops at 2 × 1640×1232 @ 30 fps; 10-min thermal check.
- [ ] `sensor-rig stamp-offset` (LED on pin 7) under full load → put the camera offsets into rig.yaml. **If jitter > 2 ms rms, build the argus_sof helper** (`sensor_rig/native/argus_sof/build.sh`) and switch to it.
- [ ] Kalibr: camera intrinsics + camera–IMU (run twice; the time shifts must agree ≤ 1 ms); Allan variance for the IMU. Then `sensor-rig calib-import`, `calib-align`, `rig-validate` (coverage ≥ 0.99).
- [ ] Measure the TCP for the real mount (`sensor-rig calib-tcp`) and replace the `"JetsonRig1,1"` entry in `gripper_calibration.yaml` (currently a copy of the iPhone's).
- [ ] Pilot: `sensor-rig collect --session rig<date>` with a calibration + 5–10 demos, plus a 60–90 s mapping video.
- [ ] `sensor-rig process-session --vio orbslam3-docker` on the pilot: ≥ 90 % of demos tracked, gravity check < 3°, gyro residuals OK.
- [ ] Sanity check: export → iPhUMI viewer (`interactive_trajectory_viewer.py`), widths vs the iPhone calibration, no black borders.

## Hardware day 2: training on rig data

- [ ] Copy the raw session from the Jetson (`rsync --exclude '.*'`), then `process-session` → run the printed `scripts/run_pipeline.sh all ...` line (first real `process_dataset.sbatch` run).
- [ ] Smoke run on 10–20 demos; then ~60 epochs at ≥ 45 demos. Compare with run_001 (val MSE, attention videos).
- [ ] Decide whether to mix rig and iPhone data (only after the widths and views are confirmed to match).

## Hardware day 2b: the iPhone + Franka Hand path (sim evaluator inputs) — follow [controlSW/franka_deploy/docs/hardware_day.md](controlSW/franka_deploy/docs/hardware_day.md) §11-15

- [ ] **iPhone latency:** `python -m franka_deploy wireless-latency measure ...` (11.5). `clock.latency_s` 0.08 is an unvalidated placeholder; rig `residual_latency_s` 0 too (11.9). Apply the printed `--set` lines with provenance.
- [ ] **Stream intrinsics:** `fdcal capture stream` + `fdcal stream-intrinsics` (11.6): deployment K vs station K vs the two training variants.
- [ ] **Reconnect check** (11.7) and the app settings (High = 960x720, 30 fps; the app default Low fails).
- [ ] **Franka Hand semantics** (`panda_py_checklist.md` §7, `hardware_day.md` 12): GIL, `stop()`, `read_once` rate, homing ≤ 20 s, grasp without an object, force per finger vs total (gauge at 30 N), eraser epsilons (`eps_inner` 0.015), hysteresis 0.064 / 0.070.
- [ ] **Width map** (13): `fdcal capture width --hand-endpoint` + `fdcal width --linear`. `a` = 1.0 is a critical placeholder, `b` = 0.0431 is derived. Check which eraser side the demos grip (20 mm).
- [ ] **panda-py `CartesianImpedance`** (14): read the source; the sim law is under-damped (15-34 % overshoot) and unverified. Steps at 600/800/1000 N/m vs `python -m franka_sim characterize`.
- [ ] After the day: re-run Stage 1 / 2b / 3 with the measured values (`hardware_day.md` §15).

## Hardware day 3: deployment — follow [controlSW/franka_deploy/docs/hardware_day.md](controlSW/franka_deploy/docs/hardware_day.md)

- [ ] `controlSW/franka_deploy/scripts/jetson_deploy_setup.sh` (RT kernel, isolated core, static IP, torch); `tools/jetson_verify.py` all PASS.
- [ ] Build libfranka/panda-py for the chosen robot; pin the versions in the config.
- [ ] Implement the `FeetechGripper` (or Dynamixel) driver in `franka_deploy/gripper/driver.py`; servo → width map (`fdcal width`).
- [ ] cyclictest pre-test, then the 30-min go/no-go (0 reflexes, success rate ≥ 0.99). On fail: move the bridge to an RT PC.
- [ ] Bridge alone: home, small steps, sine; measure lag τ and bias e.
- [ ] TCP: CAD, then hand-eye with the ChArUco board (≤ 3 mm / 1°); pivot test (≤ 2 mm).
- [ ] Open-loop demo replay (`run --replay-episode`): ≤ 1 cm / 3° over 5 s.
- [ ] Copy the prompt zarr (845 MB) to the Jetson; measure Jetson inference latency and memory (continuous mode only if capture→reply < 0.42 s).
- [ ] Closed loop in pause mode (exec_h 8, v_max 0.25, prompt 0): 10 trials, then the full evaluation protocol → `franka_deploy annotate` + `eval`.
- [ ] Check commanded vs reached gripper width (`width_proprio`); the measured-vs-target anchor sweep; micro-clamp rate < 1 %.

## Known risks to watch (from the final reviews)

1. ORB-SLAM3 on real ultrawide footage (never run).
2. Camera timestamp quality (appsink vs argus_sof).
3. The perception gap between rig and iPhone images (viewpoint, coverage, 30 fps, colour). The fallback is rig data plus fine-tuning.
4. RT + libfranka + GPU sharing one Orin Nano.
5. The gripper reaching its commanded width; the squeeze offset must stay inside the driver.
6. TCP / hand-eye error.
