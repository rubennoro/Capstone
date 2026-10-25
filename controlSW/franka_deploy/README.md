# franka_deploy

Runs the trained block_raise policy (run_001 checkpoint) on a Franka arm, with the sensor rig (Jetson Orin Nano + 2
CSI cameras) as the wrist camera. It gives the policy the same observations it was trained on, turns its action
chunks into safe Cartesian + gripper commands, and logs every rollout for evaluation.

Processes, started in this order by one launcher:

1. `bridge`: the 1 kHz Franka loop (panda-py), with the safety filter and watchdog.
2. `controller`: `FrankaCartesian` + gripper.
3. `camera`: real-env `RigCamera`.
4. `inference`: the behavior_prompting server.
5. `spacemouse`.
6. `task`: `FrankaPromptTask`.

Status: everything is tested on a laptop with fakes and mocks. Nothing has run on the arm yet; see
[docs/hardware_day.md](docs/hardware_day.md).

## Environments

| Env | Used for |
|-----|----------|
| **env_bp** (behavior_prompting + torch; `requirements-inference.txt`, imagecodecs 2023.9.18) | inference server, `eval`, `reinfer`, `offline-eval`, `bench_inference` |
| **env_realenv** (real-env + robotmq/robologger/robot-utils, pyzmq, scipy) | launcher, bridge, controller, camera, task, `fdcal`, `gonogo`, `replay-server`, `jetson_verify` |

Run from the repo root with `PYTHONPATH=controlSW/franka_deploy`. The Jetson setup is
`scripts/jetson_deploy_setup.sh` (see `--help`). The libfranka/panda-py build is
`scripts/build_libfranka_pandapy.sh --robot panda|fr3`.

## The one command

```bash
python -m franka_deploy check-config --config configs/deploy.yaml --profile jetson   # all OK first
python -m franka_deploy run --config configs/deploy.yaml
```

Keys:

| Key | Action |
|-----|--------|
| `c` | start an episode (from home) |
| `y` / `n` | stop as success / failure |
| `s` | stop |
| `r` | home |
| `q` | quit |

Each session goes to `work_root/session_*`: config snapshot, logs, episodes and taps. The stop file and the
hardware user stop always work.

## Mock dry run (laptop)

```bash
python -m franka_deploy run --config <copy of configs/deploy.yaml with laptop paths> --mock
```

The mock profile uses FakeBackend + FakeGripper + a replayed rig episode + real inference.

- Add `--replay-episode <E> --replay-speed 0.5` to replay a demo instead of running the policy.
- For a scripted run, add `--no-tty --script "3:c,30:n,32:q"`.
- The checkpoint must be a symlink in a scratch directory, never a path under `results/`. The preflight refuses
  paths under `results/`.

## Other commands

`python -m franka_deploy --help`:

- `validate`, `annotate --annotator <name>`, `eval` (env_bp), `reinfer`, `offline-eval`;
- `fdcal handeye|pivot|width|capture|stream-intrinsics`:
  - `fdcal capture stream --out DIR --n 30` then `fdcal stream-intrinsics --data DIR --config deploy.yaml --profile real-iphone --out DIR/result`: ChArUco K of the deployed 960x720 stream vs the station K and the training variants;
  - `fdcal capture width --out DIR --hand-endpoint tcp://localhost:5566 --gaps 0.0:0.08:0.0025` then `fdcal width --linear --data DIR --deploy-config deploy.yaml --profile real-iphone`: the Franka Hand's `train = a * gap + b` (the servo path keeps `--controller-config --positions`);
- `wireless-latency measure|analyze|show|synth`: QR-clock latency of the iPhone or rig stream (`--set` snippet for `sensor.clock.*`);
- `gonogo run|evaluate`;
- `replay-server`;
- `policy-contract`;
- `run` and `check-config` take `--set group.path=value` (station override, e.g. `--set host.work_root=$S/runs`).

Tests live in `tests/`. Most run in env_realenv; the server, eval and light-workspace tests run in env_bp.

## Configs

- [configs/deploy.yaml](configs/deploy.yaml): launcher (paths, task settings, profiles).
- [configs/bridge.yaml](configs/bridge.yaml): safety box/cone, v_max, slack, watchdog, backend.
- [configs/franka_cartesian.yaml](configs/franka_cartesian.yaml): F_T_TCP, interpolator, gripper and width map.

Station configs: a deploy profile can carry `station:` / `run:` ids (see profile `station-mock`). The launcher then
derives bridge/controller values and the task flags from `controlSW/configs/` (see [../configs/README.md](../configs/README.md)),
writes `config/station_resolved.yaml` into the session and refuses real runs with critical placeholders. Profiles
without `station:` behave as before. The bridge REQ/REP endpoint is now `tcp://127.0.0.1:5565` (was 5555: the iPhone
app hardcodes 5555/5556 for its camera streams).

Host paths: a profile with a station fills `work_root`, `python`, `checkpoint`, `prompt_zarr` and `disk_budget_gb` from the
station `host` group (`controlSW/configs/station/host/`) when the profile does not set them itself; the `sim-*` profiles
therefore hold no machine paths. Profiles without `station:` never read `host`. `real-*` use `jetson_orin_nano`, whose
values equal the deploy.yaml base.

Values marked `HW:` must be measured on the arm. See [docs/hardware_day.md](docs/hardware_day.md) and
[docs/panda_py_checklist.md](docs/panda_py_checklist.md).
