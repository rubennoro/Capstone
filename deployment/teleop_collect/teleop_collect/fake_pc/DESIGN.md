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

**Its C2 `q` and `dq` are always zero.** A home reference built only from joints would therefore always look "at
home". That is why teleop's home check also uses the TCP target position (`teleop.yaml home.tolerance_m`, STRUCTURE
§3.3).

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
- the PC keyboard: `keyboard_home` (`h`), `keyboard_hand_home` (`g`), `keyboard_stop` (`s`);
- `handle(msg)`, which answers one request in-process.

## Decisions

- **Copied whole, unchanged in behaviour but for the plan-replacement fix below** (PLAN §9, STRUCTURE §10.2). It already did everything teleop needs: the
  blocking C3 home, the keyboard hooks, the drifting clock, the shared schemas. Changing it would make `--mock` differ
  from what policy_deploy was tested against.
- The station section is `fake_pc:`, and `config/fake_pc.yaml` is fd:'s, unchanged.

## Copied from fd: and changed

Every file is from `fd:policy_deploy/policy_deploy/fake_pc/` @ 2ffe641:

- `settings.py`, `drifting_clock.py`, `fake_gate.py` and `plan_follower.py`: imports only.
- `fake_pc.py`: imports, and one behaviour fix found by the integration tests: an accepted `set_waypoints` first
  moves the target to "now" (as the all-skipped branch and the real PC's `plan.at(now)` do) before the new plan
  replaces the old one. Without it, each plan started from the target of the last 100 Hz tick (up to 10 ms old)
  stamped as "now", so teleop's plan replaced every 20 ms lost up to half its motion: x+ held 1 s moved 33 to 47 mm
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
- Helpers: `tests/launcher_helpers.py` (`make_fake_pc`, `FakeLaptopClock`, `wait_for`). They run it in-process on
  `tcp://127.0.0.1:*`.

## How to extend

- **A new fault or link condition for a test:** a method under "test hooks" in `fake_pc.py` that changes state under
  `self._lock`, plus a test in `test_fake_pc.py`.
- **A new C3 request:** this needs a new `deploy_contracts` schema, which PLAN §2 rule 2 forbids, so the PC would
  change. Don't.
