# fake_pc: design

The Python stand-in for the PC's `franka_control` under `--mock` (PLAN §3, STRUCTURE §10.2). This is laptop-side
code: it runs on the laptop, on 127.0.0.1, and speaks only the PC's existing C2/C3 contracts (`deploy_contracts`). It
is not PC code, and nothing on the PC changes.

## What it does, and what it does not

- `python -m teleop_collect.fake_pc [--station S] [--session D] [--ready-file F] --mock` binds C2 (PUB) and C3 (REP) on
  127.0.0.1 with the `link.yaml` ports. It is ready (ready file) once both are bound and the first C2 is out. A bind
  failure is `FATAL fake_pc: ...`. SIGINT or SIGTERM stop it with exit 0. The launcher starts it first under `--mock`
  (with `launcher.fake_pc: true`) and stops it last.
- It answers C3 per mode as franka_control does (`idle`, `holding`, `running`, `stopped`, `homing`, `fault`), with the
  real refusal texts:
  - `hello` reports `mock: true`, `schemas=SHARED_SCHEMAS`, `F_T_TCP` and the limits from `deployment/contracts/`, the
    watchdog from `link.yaml`, and `homing_timeout_s` / `max_lead_s` from `config/fake_pc.yaml`.
  - A laptop-only schema (R1, U1, U2) is `unknown_schema`, as on the PC.
- C2 at `state_rate_hz` on a **drifting, offset clock** (`clock_offset_s`, `clock_drift_ppm`), so teleop's PC clock map
  is always exercised.
- It has the PC's watchdog (`holding` / `running` without a request for `watchdog_limit_s` → `stopped`).
- **A C3 `home` blocks the REP socket** until the move (`home_duration_s`, to `home_pose`) ends, as the PC does.
- Its gate has the time and width rules only: a past waypoint is skipped; beyond `max_lead_s`, or a width outside
  `[0, gripper_width_max_m]`, is rejected and gives `fault`. A set whose waypoints are all skipped freezes the target
  and goes `holding`.
- **Not modelled:**
  - libfranka, dynamics, joints;
  - the workspace box and the tilt cone (`pulled_back` and `shortened` only through the test hooks);
  - gripper physics beyond a width ramp;
  - a real-time loop;
  - bind addresses and the firewall.

**Its C2 `q` and `dq` are always zero.** A reference pose built only from joints would therefore always match. That
is why teleop's start pose check also uses the TCP target position (`teleop.yaml start_pose.tolerance_m`), and why the
fake PC's own start pose checks use the TCP target (below).

## The start pose (2026-11-20)

final-deployment's start-pose work (`927a5ba`) gave the PC a key `p` and a start pose check. The fake PC
models the PC with `homing.q_start_rad` set, by the TCP target instead of the joints:

- `p_pose` (`config/fake_pc.yaml`: the same as `start_pose`, so a mock session starts at the start pose, as if `h`,
  then `p` were pressed). `null` models `q_start_rad` unset: `p` is refused and a C3 `start` never checks the pose.
- `keyboard_start_pose()` is the PC's `p`, with its refusals in its order: `p_pose` unset (`refused:
  homing.q_start_rad is not set: ...`), then the mode (the StateMachine text, as `h`), then away from `home_pose`
  (`refused: not at home: press h first`). Else a home-like move (`home_duration_s`, mode `homing`, then `idle`) to
  `p_pose`, with the hand opening to `gripper_width_max_m`. A stop during it aborts it as a home does.
- A C3 `start` away from `p_pose` (after the mode check): `refused: not at the start pose: press h, then p on the PC`.
- "At a pose": the target position within `at_pose_tol_m` (0.002) and its rotation within 1 deg (`AT_POSE_TOL_DEG`).
- The operator stand-in, `auto_h_p_after_stop_s` (1.0; `null`: off): that long after an episode's stop (`holding` /
  `running` → `stopped` by a C3 `stop`, the PC's `s` or the watchdog), it presses `h`, and when that home ends in
  `idle`, `p`. A stop during either cancels the rest; a start before the `h` cancels it. Under `--mock` nobody can
  press the PC's keys, so every episode would otherwise end the session.
- The real PC's other checks around `p` (the move's duration against `homing.timeout_s`, the box and the cone at its
  end) are not modelled, as for `h`.

## Classes and threads

| File | Class | Job |
|---|---|---|
| `__main__.py` | `main` | process entry: settings, bind, signals, ready file |
| `settings.py` | `FakePcSettings` | `config/fake_pc.yaml` + station `fake_pc:` section, checked; plus the endpoints, watchdog, limits and `F_T_TCP` from `deployment/contracts/` (never from config) |
| `fake_pc.py` | `FakePc` | the modes, the C3 handlers, the C2 tick, the test hooks, `commands.jsonl` in its session folder |
| `drifting_clock.py` | `DriftingClock` | `t_pc = laptop_ns × (1 + drift) + offset`, with `shift()` |
| `fake_gate.py` | `FakeGate`, `GateResult` | the skip / lead / width rules |
| `plan_follower.py` | `PlanFollower`, `interpolate` | the target follows the accepted plan (linear position, slerp rotation), and the width ramps toward the commanded width |

There are two threads. `fake-pc-state` calls `tick()` at `state_rate_hz`: it advances the modes and the plan, then
publishes. `fake-pc-commands` serves C3 through `RpcServer` with `schemas=SHARED_SCHEMAS`. Every request that reaches
a handler feeds the watchdog. A broken request never does.

**Test hooks** (methods, for E3's and E7's tests):

- `inject_fault`, `restart` (seq from 1);
- `stall` (C2 and/or C3), `burst_after_stall`, `delay_states`, `drop_states`, `repeat_t_ns`;
- `set_mode`, `hold_forever`, `shift_clock`, `report_shortened`, `report_pulled_back`;
- the PC keyboard: `keyboard_home` (`h`), `keyboard_start_pose` (`p`), `keyboard_hand_home` (`g`), `keyboard_stop`
  (`s`);
- `handle(msg)`, which answers one request in-process.

## Decisions

- **Copied whole, unchanged in behaviour but for the plan-replacement fix and the start pose** (PLAN §9, STRUCTURE
  §10.2). It already did everything teleop needs: the keyboard hooks, the drifting clock, the shared schemas. Changing
  it more would make `--mock` differ from what policy_deploy was tested against. The blocking C3 home stays although
  teleop no longer sends one (the real PC still answers it).
- The station section is `fake_pc:`, and `config/fake_pc.yaml` is fd:'s with only the start pose block appended
  (`p_pose`, `at_pose_tol_m`, `auto_h_p_after_stop_s`).

## Copied from fd: and changed

Every file is from `fd:policy_deploy/policy_deploy/fake_pc/` @ 2ffe641:

- `drifting_clock.py`, `fake_gate.py` and `plan_follower.py`: imports only.
- `settings.py`: imports, and the start pose keys.
- `fake_pc.py`: imports, the start pose (above), and one behaviour fix found by the integration tests: an accepted
  `set_waypoints` first moves the target to "now" (as the all-skipped branch and the real PC's `plan.at(now)` do) before
  the new plan replaces the old one. Without it, each plan started from the target of the last 100 Hz tick (up to 10 ms
  old) stamped as "now", so teleop's plan replaced every 20 ms lost up to half its motion: x+ held 1 s moved 33 to 47 mm
  instead of 50 (the real franka_control: 50.0 mm/s).
- `__init__.py` and `__main__.py`: also the module name in the docstrings and `(teleop_collect check)` in the bind
  hint.

The docstrings still cite the source's section numbers (`STRUCTURE.md 10.x`, `TOP 3.7.x` are
fd:policy_deploy/STRUCTURE.md and fd:STRUCTURE.md).

## Tests

- `tests/fake_pc/test_fake_pc.py` (copied, imports only): the mode table, the refusal texts, hello on the drifting
  clock, the keyboard hooks, the blocking home, the watchdog, seq and t_ns, the follower and the gate, the commands
  log.
- `tests/fake_pc/test_shared_examples.py`: every stored C2/C3 example from `deployment/contracts/examples/`
  (`deploy_contracts.load_index()`). A good request gets a valid reply of the right schema; a broken one gets
  `error/v1` naming its field and never reaches a handler; the R1/U1/U2 examples are unknown schemas. The fake PC's
  clock offset is 1e6 s there, so the stored example times are in the past on any machine. With the source's 1000 s,
  that needed an uptime above about 34 h.
- `tests/fake_pc/test_start_pose.py`: `p` (its refusals, texts and order; the move, the hand open; a stop during it),
  the C3 start at and away from `p_pose`, `p_pose: null` as before, the operator stand-in, the settings keys.
  `test_fake_pc.py` and `test_shared_examples.py` run with `p_pose` and the stand-in off.
- Helpers: `tests/launcher_helpers.py` (`make_fake_pc`, `FakeLaptopClock`, `wait_for`). They run it in-process on
  `tcp://127.0.0.1:*`.

## How to extend

- **A new fault or link condition for a test:** a method under "test hooks" in `fake_pc.py` that changes state under
  `self._lock`, plus a test in `test_fake_pc.py`.
- **A new C3 request:** this needs a new `deploy_contracts` schema, which PLAN §2 rule 2 forbids, so the PC would
  change. Don't.
