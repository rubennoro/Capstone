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
- `fdcal handeye|pivot|width|capture`;
- `gonogo run|evaluate`;
- `replay-server`.

Tests live in `tests/`. Most run in env_realenv; the server, eval and light-workspace tests run in env_bp.

## Configs

- [configs/deploy.yaml](configs/deploy.yaml): launcher (paths, task settings, profiles).
- [configs/bridge.yaml](configs/bridge.yaml): safety box/cone, v_max, slack, watchdog, backend.
- [configs/franka_cartesian.yaml](configs/franka_cartesian.yaml): F_T_TCP, interpolator, gripper and width map.
- [configs/task_franka.yaml](configs/task_franka.yaml): anchor, pause/exec_h, width proprio.

Values marked `HW:` must be measured on the arm. See [docs/hardware_day.md](docs/hardware_day.md) and
[docs/panda_py_checklist.md](docs/panda_py_checklist.md).
