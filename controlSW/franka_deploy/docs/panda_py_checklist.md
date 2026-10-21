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

## 6. Sim fidelity checks (A4: `franka_sim` SimCore / MuJoCoBackend)

The sim (`controlSW/franka_sim/franka_sim/simcore.py`) implements panda-py's CartesianImpedance from its documented
law; the panda-py source was not available offline. Verify each item against the pinned panda-py and the arm.

- [ ] CartesianImpedance law (verify-on-hardware): per 1 ms, `tau_task = J^T (-K e - D J dq)` with
      `D = 2 * damping_ratio * sqrt(K)` elementwise (not mass-normalised), orientation error `e_r = -R vec(q^-1 q_d)`
      (q_d sign-flipped to the same hemisphere), null space `(I - J^T pinv(J^T)) (k_ns (q_ns - q) - 2 sqrt(k_ns) dq)`
      with `q_ns` = q at controller start and an undamped pinv, plus Coriolis; torque rate limit 1 Nm/ms. If panda-py
      uses a damped pseudo-inverse (franka_example_controllers uses lambda 0.2), set `backend.mujoco.pinv_damping`.
- [ ] Target filter: `p_d <- fc p_t + (1 - fc) p_d` and slerp with `filter_coeff` every 1 ms (1.0 = no filter).
- [ ] Step response vs the sim baseline (`franka_sim/configs/characterization_baseline.json`, `python -m franka_sim
      characterize`): run §2's 2 cm steps at 600/800/1000 N/m and compare lag and overshoot. The sim gives 15-35 %
      overshoot on ideal 2 cm steps and a gonogo-steps lag of ~90-120 ms (effective damping ratio 1/sqrt(Lambda) with
      Lambda 4.6-8.3 kg); real joint friction will damp more.
- [ ] Error derivation in `PandaPyBackend.get_state` (`derive_panda_error`): a reflex gives `robot_mode` Reflex; check the
      exact `robot_mode` value (`.name` / `str`), that `current_errors` / `last_motion_errors` print as `[]` when empty
      (else adapt `_errors_text`), and that after a user stop or a dead controller thread `robot_mode` leaves Move
      (the "controller stopped unexpectedly" rule, 0.5 s grace after `start`). `recover()` clears the error after
      `automatic_error_recovery`.
- [ ] Desk end-effector setting = `robot.desk_end_effector` in the station (`none` -> the bridge commands the flange;
      `franka_hand` -> F_T_NE / NE_T_EE apply and `ee_is_tcp` / EE_T_TCP must follow).
- [ ] Collision thresholds: `robot.impedance.collision_thresholds` / `backend.panda_py.collision` match what
      `set_collision_behavior` received (the sim uses the upper torque/force thresholds, else the libfranka defaults
      tau [20,20,18,18,16,14,12] Nm, F [20,20,20,25,25,25]).
- [ ] Payload: Desk/`setLoad` mass and CoM vs the weighed tooling; the sim's `physics.payload_error_kg` (configured -
      true) shows a wrong payload as a z offset of about m g / K_z (3.7 mm per 0.3 kg at 800 N/m).
- [ ] Phone/cradle vs links 5/6: on both arms the sim's q7 rotation hits a self-collision before the q7 limit (the sim
      reflexes on it; MuJoCo itself does not collide the tooling with link 6). Check the real clearance before
      large q7 motions.

## 7. Franka Hand (A5: `gripper/hand_server.py` LibfrankaHand, `gripper/franka_hand.py`, sim `franka_sim/hand_sim.py`)

`LibfrankaHand` is unexecuted until these pass. Run `python -m franka_deploy.gripper.hand_server --endpoint
tcp://localhost:5566 --backend libfranka --hostname <fci>` alone first and drive it with a `BridgeClient`
(`hand_cmd` / `hand_state`); then the controller with `gripper.type: franka_hand`.

- [ ] Names in the pinned panda-py: `panda_py.libfranka.Gripper(hostname)`, `homing()`, `move(width, speed)`,
      `grasp(width, speed, force, epsilon_inner, epsilon_outer)`, `stop()`, `read_once()` -> `.width`,
      `.is_grasped`, `.max_width`.
- [ ] GIL: does panda-py release the GIL during the blocking `Gripper` calls? Check that `hand_state` replies stay
      < 5 ms while a `move` runs. If not, the REP loop stalls for the call time: the driver then counts
      `rpc_timeouts`/`stale_reads` (the controller is unaffected) and hand_server needs the device in a subprocess.
- [ ] `stop()` from another thread mid-`grasp` and mid-`move`: does the blocked call return at once (False)? A new
      `hand_cmd` while busy relies on it (latest wins); the worker waits for the stop before the next command.
- [ ] `read_once()` rate while idle and while a command runs (the state thread polls at 50 Hz; `gap_vel_m_s` is a
      finite difference of it).
- [ ] Homing after power-on: needed before `move`/`grasp` give correct widths? hand_server homes at start unless
      `--no-homing`; record how long it takes (the launcher waits 20 s for `hand_state`).
- [ ] Grasp semantics vs the sim: with no object, does `grasp` close to 0 or stop at `width - epsilon_inner` (the sim
      and FakeHand stop at `width - epsilon_inner`)? With an object narrower than `width - epsilon_inner`, does it
      still close onto it? This decides whether a small `epsilon_inner` can miss the eraser entirely.
- [ ] Grasp force: is libfranka's `force` per finger or total? The sim applies a tendon force of -2F (F per finger);
      measure the squeeze (force gauge between the pads) at `force_n` 30.
- [ ] Epsilons for the eraser: with `width_map.b` 0.0431 the policy grasps when the training width drops below 0.064
      (gap ~0.018-0.021) and the 20 mm side stops the pads at gap ~0.012 (sim pad offset 0.008), so
      `tooling.hand.grasp_epsilon_inner_m` must be >= ~0.009; the station uses 0.015 (no-object close stops >= 3 mm
      above 0). `grasp_epsilon_outer_m` 0.08.
- [ ] Width map: measure `train = a * gap + b` with `fdcal width` (tag separation from the ultrawide vs
      `read_once().width` over the stroke) and enter `tooling.width_map.a/b`. Current values: a 1.0 (placeholder),
      b 0.0431 (derived: = `train_closed_m`, the tag separation at gap 0). With them the sim holds the 20 mm eraser
      at train ~0.055, inside the data's hold widths (0.046-0.062). Also measure the pad offset (pad separation at
      gap 0; sim `model.PAD_OFFSET_M` 0.008, or `tooling.finger_adapter.pad_offset_m`).
- [ ] Re-grasp: while GRASPING, a hand that is idle and not holding (grasp_failed, hand_server restarted mid-hold)
      gets up to `policy.max_regrasps` (2) new grasps 0.25 s apart; check this does not cycle the fingers on hardware.
- [ ] Shutdown while holding: hand_server `close()` (SIGINT/SIGTERM) stops only a RUNNING command; a finished grasp
      that is holding is left alone (no `stop()`), so the eraser stays in the fingers rather than dropping. Check that
      the hand keeps its grasp force after the hand_server process exits, and use the Desk/`homing` to release.
- [ ] Hysteresis on hardware: `gripper.policy.close_below_train_m` 0.064 / `open_above_train_m` 0.070 (from the
      firstreal-single widths); no chatter while the policy holds, exactly one release.
