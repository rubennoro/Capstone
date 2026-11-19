# teleop/motion: design

Pure logic, no sockets, no threads. Called by teleop's main loop (`teleop_loop.py`, E3) once per 20 ms cycle. All new
code; nothing copied from fd:.

## What it does, and does not

- Does: turns the ui's key state into at most one `set_waypoints` per cycle (PLAN 5.1 step 3, 5.2), keeps the PLAN 5.3
  safety bound, and knows whether the arm is at home (PLAN 8).
- Does not: talk to the PC (E3 builds the C3 `SetWaypoints` from a `MotionCommand` and sends it), keep a pose of its
  own (every command starts from the PC's latest target), know the PC's box or cone (PLAN 13.3), or compute joints.

## Classes and data flow

```
U1 keys ──► KeyState.update ─► KeyState.current(now) ─► HeldKeys ─┐
newest C2 ─► BaseState(T_target, t_pc_ns, width, mode) ───────────┼─► MotionController.step ─► MotionCommand | None
PcClock ─► now_pc_ns ──────────────────────────────────────────────┘        │ twist_from_keys ─► Twist
U1 gripper ─► MotionController.gripper                                      └ WaypointPlanner.move/release/hold
every C2 ─► HomeReference.observe;  after a C3 home ─► HomeReference.set_from;  Start ─► at_home(q, T_target)
```

| File | Class / function | Job |
|---|---|---|
| `keys.py` | `HeldKeys`, `KeyState` | newest key state; deadman: no focus or older than `key_stale_ms` = no keys; `changes()` for keys.csv |
| `twist.py` | `Twist`, `twist_from_keys` | ±v_max per axis (base frame), twist about the TCP z, tilt about base x/y, slow, tilt normalised |
| `planner.py` | `MotionSettings`, `Waypoint`, `advance`, `WaypointPlanner`, `worst_step` | the settings and the margin rule; poses advanced from the base pose |
| `controller.py` | `BaseState`, `MotionCommand`, `MotionController` | the step rules below; commanded width |
| `home.py` | `HomeSettings`, `HomeReference` | the home reference and its tolerances |

**`advance(T, twist, tau)`**: `p = p + v tau`, `R = Rot_base([w_tilt_x, w_tilt_y, 0] tau) @ R @ Rot_z(w_twist tau)`,
re-orthonormalised by SVD. The tilt multiplies on the left (base axes) and the twist on the right (the TCP's own z);
both pivot at the TCP point, so a pure rotation keeps the position (PLAN 13.2). Tilt signs by effect: `tilt_fwd` moves
the TCP z axis (down at home) toward +x, `tilt_left` toward +y.

**`MotionController.step` rules, in order** (STRUCTURE 7.4):
1. not begun / stopped → None;
2. now_pc or base missing, base older than `state_max_age_ms`, or C2 mode not `running`/`holding` → None (pending
   release or finger command waits);
3. non-zero twist → `move`: K waypoints at `now_pc + kΔ`, advanced over `t_k − t_base` (a pending width rides on it);
4. twist became zero after a move → one `release`: one waypoint at `now_pc + Δ`, the base advanced by the old twist
   over the state's age, but not past the end of the last plan sent;
5. pending finger command → `fingers`: one waypoint at the base pose with the new width;
6. otherwise None.

## Decisions

- **The bound (PLAN 5.3).** A waypoint lies at most `speed × (K Δ + state_max_age)` past its base pose, per
  translation axis and in tilt of the z axis (`worst_step`). `MotionSettings` refuses settings above `margin_use` ×
  the assumed PC margins (defaults: 7.5 mm ≤ 8 mm, 1.5° ≤ 1.6°). Diagonal tilt is normalised to `w_max`.
- **Stale state sends nothing** (STRUCTURE 7.4 rule 2, refining PLAN 5.3's holding set): a set built on a stale
  state could exceed the margin. Rule 2 also refuses modes other than running/holding: the PC refuses
  `set_waypoints` there anyway, and right after C3 `start` the newest C2 may still say `idle`.
- **Release not past the plan end.** After its last waypoint the PC target stands still; advancing a later base
  by the old twist would creep. `WaypointPlanner.release` has an extra optional `until_pc_ns` for this.
- **Width always in [0, width_open_m]:** `begin(measured)` clips the C2 width it starts from (an open Franka Hand reads
  about 0.080-0.081 m plus the PC's offset, and the gate rejects widths above `gripper_width_max_m`: a fault). It never
  raises for the width and returns / exposes `width_clipped` for E3's log. N/M give exactly 0 and `width_open_m`.
- **Fingers (PLAN 13.10):** `close` = 0.0, `open` = `width_open_m` (E3 passes `limits.yaml gripper_width_max_m`). Every
  N/M press sends, even if the width is unchanged. Ignored before `begin`.
- **Home (PLAN 8):** reference = q and TCP target position after a home ending in `idle` (C3 home via `set_from`, or
  `homing → idle` on C2 via `observe`). `at_home` checks only q and position; the mode rule is E3's.
  **`observe` sets the reference only while none is known, and only if the arm moved during that homing** (a joint
  by more than `tolerance_rad` or the TCP target by more than `tolerance_m`, between the first `homing` C2 and the
  `idle` one). The PC's `g` (homing the hand alone) shows the same `homing → idle` on C2 wherever the arm is and moves
  neither, so it never counts, before or after a home. A PC `h` while the arm already stands at home looks the same
  and is not accepted either: Start stays refused until a C3 home (the page's Home). The fake PC's home moves the TCP
  target (its q stays 0), which is enough. The PC's home pose is fixed, so the first moving home is enough;
  `set_from` (after a C3 home teleop sent) always replaces it.

## Tests

`tests/motion/`: key deadman and changes; twist signs, cancelling, slow, normalisation; `advance` (identity at 0,
directions by effect, pivot, twist about own axis when tilted, rigid); settings values and the margin rule; planner
times and poses; every controller rule; the home reference (a `g` never counts, before or after a home; an `h` counts only if the arm moved); and the bound for all 4096 key combinations × slow × state ages 0 and the
guard, from a home and a tilted pose (the home case reaches the bound exactly). Helpers in `tests/core_helpers.py`.

## How to extend

- **A new motion key:** add its name to `MOTION_KEYS` (E1's contracts) and the ui's map, then give it an axis in
  `twist_from_keys`. If it can raise the speed of an axis, extend `worst_step` and the bound test.
- **A new command kind:** add it to `KINDS` and to `step` in its place in the rule order; keep rule 2 first.
