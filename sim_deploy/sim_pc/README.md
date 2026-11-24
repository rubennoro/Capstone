# sim_pc: the sim PC

The copied real `franka_control` (`deployment/control_deploy/franka_control/`) built with a MuJoCo arm and gripper.
With `--sim` the real gate, supervisor, watchdog, impedance controller, homing and C2/C3 codec run against physics.
Design: [../DESIGN.md](../DESIGN.md) sections 3, 4, 5, 8 and "E4: as built".

| File | What |
|---|---|
| `include/sim_pc/sim_config.h`, `src/sim_config.cpp` | the station's `sim:` block (strict) |
| `include/sim_pc/sim_world.h`, `src/sim_world.cpp` | the physics thread: 1 ms steps paced to `CLOCK_MONOTONIC`, torque/joint calls, joint hold, gravity compensation as libfranka, finger servo, reflexes |
| `src/sim_scene.cpp` | the io thread: state PUB :5650 (`sim.state/v1`), scene REP :5651 (`hello`, `set_object`, `ground_truth`, `set_arm`, `key`, `stats`) |
| `include/sim_pc/sim_parts.h`, `src/sim_parts.cpp` | `SimArm`, `SimGripper` (franka_control's `Arm`/`Gripper`), `makeSim` |
| `include/sim_pc/sha256.h`, `src/sha256.cpp` | the `model_sha256` check against `model.json` |
| `payload.py` | the `load` block (rig + finger kits − stock fingers) from the model |
| `merge_station.py` | P1: a base station + `stations/start_pose.yaml` → the one `--station` file (`build/<base>.merged.yaml`) |

## Build (once, and after every change or pull)

Needs the apt packages of the laptop's libfranka-less build (`control_deploy/STRUCTURE.md` 8.3) and the
`~/envs/sim_deploy` env (MuJoCo 3.3.0 wheel: `sim_deploy/setup_envs.sh sim`).

```bash
cd ~/Capstone-main-sim/deployment/control_deploy/franka_control
cmake -S . -B build-sim -DCMAKE_BUILD_TYPE=Release -DFRANKA_CONTROL_SIM=ON \
      -DFRANKA_CONTROL_WITH_LIBFRANKA=OFF -DFRANKA_CONTROL_BUILD_TESTS=OFF
cmake --build build-sim -j8
```

`MUJOCO_DIR` (default `~/envs/sim_deploy/lib/python3.10/site-packages/mujoco`) can be set on the cmake line. The
binary carries an rpath to it.

## Run

The model first (E6): `python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml`. Then the
station: franka_control takes one `--station` file, so the start pose (`stations/start_pose.yaml`, only
`franka_control.homing.q_start_rad`) is merged into the base station first (`run.sh` does this before every sim PC
start). Then, with ports 5620, 5630, 5650 and 5651 free (`ss -ltn | grep -E ':56(20|30|50|51) '`):

```bash
cd ~/Capstone-main-sim
~/envs/sim_deploy/bin/python -I sim_deploy/sim_pc/merge_station.py sim_deploy/stations/control_sim.yaml \
    sim_deploy/stations/start_pose.yaml sim_deploy/build/control_sim.merged.yaml
deployment/control_deploy/franka_control/build-sim/franka_control --sim --local --keyboard \
    --station sim_deploy/build/control_sim.merged.yaml [--ready-file FILE] [--session DIR]
```

- `--sim` needs `--local` (C2 and C3 on `127.0.0.1:5620/5630`) and refuses `--mock`. `hello.mock` is true, so the
  laptop runs as `policy_deploy run --mock` with `launcher.fake_pc: false`.
- Keys as on the PC: `h` home the arm, `p` start pose (from home only: joint move to `homing.q_start_rad`, the hand
  opens to `gripper_width_max_m`, 0.040), `g` home the hand, `s` stop, `q` quit. Ctrl-C / SIGTERM shut down cleanly. The same h/p/g/s also
  come in through the scene REP (`key`, below), with or without `--keyboard`.
- Once `q_start_rad` is set, C3 `start` is refused unless every joint is within `homing.at_pose_tol_rad` (0.05) of
  it (`refused: not at the start pose: press h, then p on the PC`): `h`, then `p`, then the laptop's `c`. The laptop
  has no `h` any more (the port removed it), so the eval driver homes with the scene key `h`.
- The log prints a `sim: stats` line every 10 s (real-time factor, slow steps, lag, step cost, state rate) and the
  run totals at the end.
- `control_deploy run|check|console` are not used for the sim: their strict station loader refuses `sim:`. The
  copy's console still works as a C3 client, with any station without `sim:` (the box comes from `config.yaml`); with
  a start pose set, press `h` then `p` (scene key) first, since its start is refused elsewhere:

```bash
cd ~/Capstone-main-sim/deployment/control_deploy && unset PYTHONPATH
~/envs/sim_policy_deploy/bin/python -c "import sys; sys.path.insert(0, '.'); from control_deploy.cli import main; sys.exit(main())" \
    console --mock --local --yes --station station.yaml     # writes to station.yaml's session_root
cd ../policy_deploy
~/envs/sim_policy_deploy/bin/python -m policy_deploy.manager.pc_probe --local --mock --seconds 10 --axes --start-pose \
    --station ../../sim_deploy/stations/policy_sim.yaml     # --start-pose: the q_start_rad line, height, tilt, width
```

## Scene control (REP :5651, JSON)

```bash
~/envs/sim_deploy/bin/python -c "import zmq, json; s = zmq.Context().socket(zmq.REQ); s.connect('tcp://127.0.0.1:5651'); \
s.send_string(json.dumps({'cmd': 'ground_truth'})); print(s.recv_string())"
```

`set_object {name, pose: [x, y, z, qw, qx, qy, qz]}` (any body with a free joint, velocity zeroed),
`ground_truth` (objects, `tcp`, `flange`, `finger_gap_m`, `gripper_width_m` (= gap + `width_offset_m`, the C2 width), `arm` {q, call, reflex,
tau_ext_nm}, `contacts` [{a, b, force_n}] without eraser–table), `set_arm {q}` (refused while a torque session or
homing runs, which in franka_control is always after start-up: reset the arm with the key `h` / C3 `home`), `stats`.

`key {key: "h" | "p" | "g" | "s"}` (sim only, P1): pushes the key into franka_control's own `KeyQueue`, the queue the
keyboard thread feeds, so it takes exactly the path of a key typed on the PC (the messaging thread drains it into
`Supervisor::onKey`, also without `--keyboard`). The reply waits (without blocking the state PUB) for the
supervisor's one log line about that key: refused → `{"ok": false, "key": "p", "error": "key p refused: ..."}`;
accepted → `{"ok": true, "key": "p", "log": "key p: start pose: 15.1 s joint move to homing.q_start_rad"}`; no line
within 1 s → `{"ok": true, "key": ..., "log": null}`. The outcome of an accepted move (`start pose reached`, or a
box/cone `fault`) shows later in C2 (`homing` → `idle` / `fault`) and the log. `q` is refused (not a queued key).

```bash
~/envs/sim_deploy/bin/python -c "import zmq, json; s = zmq.Context().socket(zmq.REQ); s.connect('tcp://127.0.0.1:5651'); \
s.send_string(json.dumps({'cmd': 'key', 'key': 'p'})); print(s.recv_string())"
```

## Payload

```bash
~/envs/sim_deploy/bin/python -I sim_deploy/sim_pc/payload.py      # prints the load: block for the station
```
