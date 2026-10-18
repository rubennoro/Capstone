# Hardware checklist: franka_bridge with the panda-py backend

Everything in `franka_deploy/bridge/backends.py::PandaPyBackend` is unexecuted until these steps pass on the real
arm. Do them in order, with the external activation device (user stop) in hand and the zone clear.

## 0. Before the arm (B0)

- [ ] Robot model and system version: Panda (system 4.x) or FR3 (system 5.x). FCI is enabled in Desk.
- [ ] Choose libfranka per Franka's compatibility table (Panda → 0.9.x, FR3 → ≥ 0.10). Build panda-py against
      exactly that libfranka. On the Jetson (aarch64) both are source builds.
- [ ] Record the versions in `configs/bridge.yaml` (`backend.panda_py_version`, `backend.libfranka_version`).
- [ ] Re-check the panda-py names used by `PandaPyBackend` against the pinned version:
  - `Panda(hostname)`, `get_state()` (`O_T_EE` column-major, `q`, `dq`, `O_F_ext_hat_K`,
    `control_command_success_rate`, `robot_mode`);
  - `start_controller`, `stop_controller`, `recover`, `move_to_joint_position(q, speed_factor)`;
  - `get_robot().set_collision_behavior`, `get_robot().set_EE`;
  - `controllers.CartesianImpedance(impedance, damping_ratio, nullspace_stiffness, filter_coeff)`,
    `set_control(position, orientation_xyzw)`.
- [ ] Wiring: a direct Ethernet link to the control box, nothing else on that NIC. Development access goes over
      Wi-Fi or USB-Ethernet.

## 1. Host real-time (PLAN_B §2.6 go/no-go)

- [ ] PREEMPT_RT kernel (JetPack `nvidia-l4t-rt-kernel`). Confirm both Argus cameras and CUDA still work.
- [ ] Run the bridge's libfranka thread on an isolated core (`isolcpus=5 nohz_full=5 rcu_nocbs=5`), with the NIC IRQ
      pinned to that core. Then `nvpmodel` max and `jetson_clocks`.
- [ ] libfranka `communication_test`: success rate ≥ 0.99.
- [ ] 30-minute co-load: bridge holding the home pose, with RigCamera, back-to-back inference and the task running.
  - Pass: zero reflexes, per-second `min_success` ≥ 0.99 in the bridge stats lines, 1 kHz loop p99.9 < 0.5 ms.
  - On failure, move only the bridge to an x86 PREEMPT_RT PC (`endpoint: tcp://<rt-pc>:5565`).

## 2. Bridge alone (no task, no policy)

- [ ] `backend.type: panda_py`, low impedance (600 N/m, 30 Nm/rad). Start with `start`, check that `get_state`
      matches Desk.
- [ ] Measure the workspace box and the orientation cone around the table and enter them in `bridge.yaml`.
  - Test each rule with deliberate bad targets sent through `BridgeClient`: NaN → hold, out of the box → clamped,
    outside the cone → clamped, a 10 cm jump → executed at ≤ 0.25 m/s.
- [ ] Watchdog: stop sending targets. Expect `watchdog_hold` within 0.2 s and `watchdog_stop` within 2 s; the arm
      stops. `start`/`recover` is the only way back.
- [ ] Reflex: push the arm past the collision thresholds. Expect bridge state `reflex`, the controller `fault` on
      "error", and no automatic recovery. Recover only through `recover`.
- [ ] `setCollisionBehavior` thresholds (about 20 N / 10 Nm to start). Payload (gripper + rig + Jetson, about
      1–1.5 kg, plus CoM) set in Desk or with `setLoad`.
- [ ] Steps of 2 cm per axis at 0.05 m/s, then a 1 Hz sine of ±3 cm. Tune stiffness and filter. Record the
      steady-state error e and the lag τ.

## 3. Frames (B4)

- [ ] F_T_TCP: CAD first, then hand-eye (ChArUco, ≥ 20 poses, `cv2.calibrateHandEye`). Accept agreement within 3 mm
      / 1°. Enter it in `franka_cartesian.yaml`.
- [ ] Either `ee_is_tcp: false` (the bridge commands the flange and the controller composes F_T_TCP once), or set
      the EE in Desk / `NE_T_EE` and use `ee_is_tcp: true`. Never both.
- [ ] Pivot test: ±20° about TCP x/y/z. The fingertip centre stays within 2 mm of a marked point.

## 4. Controller and gripper

- [ ] `FrankaCartesian` with spacemouse teleop (task idle). Check `pose_orientation_types`. Status `hop_ms_p99` and
      `period_err_ms_p99` under load.
- [ ] Set `lookahead_time_s` to 0.1 + measured τ.
- [ ] Gripper driver (Feetech/Dynamixel stub → real implementation):
  - calibration table from a servo sweep plus the tag detector (monotonic, round trip < 0.1 mm);
  - current limit and squeeze δ on the eraser;
  - commanded vs reached width logged (decision b, B5).
- [ ] Hold keeps the grasp: during a grasp send `hold`. The commanded width must stay and the object must not drop.

## 5. Closed loop (B5)

- [ ] Task with `anchor: target`, pause mode, exec_h 8, inference timeout 2 s. Watch the per-call
      `info_anchor_dev_m` in the tap: > 2 cm → fallback + reseed, > 5 cm → hold.
- [ ] Stop paths: stop file or command-file key → hold ≤ 0.1 s, and the hardware user stop.
