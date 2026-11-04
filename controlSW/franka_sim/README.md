# franka_sim

Phone-free closed-loop simulation of the station: MuJoCo Panda + Franka Hand + printed fingers (the real Onshape finger
kit by default, box fingers as a fallback) + the custom sensor rig, driven by the real deployment stack
(`../franka_deploy`: launcher, bridge, controller, task) and any policy that speaks the policy contract. Each episode
gets two separate results:

- **Safety** (the verdict): PASS, or FAIL on the first violation (non-finger contact, reflex, workspace box, approach
  cone, joint margin, EE speed / angular speed), measured at the EE site the bridge enforces. Rules:
  `../configs/run/criteria/pick_place.yaml` `safety` + `limits`, the scene `workspace`, the robot joint limits.
- **Task outcome** (reported, never a FAIL): none < grasped < lifted < placed (eraser on the green marker at
  `scene.place`, released, hand open, TCP retreated, held 1 s). Rules: the same criteria file, `task`.

## Quickstart: three runs, then read the reports

From a fresh clone on branch `rnoro/main-sim`, on a machine with a display:

```
cd controlSW/franka_sim
./setup_env.sh                                              # optional: run_viewer.sh does it on first use

./run_viewer.sh 3                                           # 1. kit fingers (default), 3 eraser poses
./run_viewer.sh 3 --box                                     # 2. box fingers (+10 cm), 3 poses
./run_viewer.sh 1 --set criteria.safety.joint_margin_deg=60 # 3. forced safety FAIL (proves the judge stops unsafe runs)
```

Each run opens the MuJoCo viewer and takes about 1.5–2 minutes for 3 poses. The window closes by itself at the end.
Each prints `report: ~/sim_runs/scripted_<time>/summary.html`. Open the reports with:

```
xdg-open "$(ls -td ~/sim_runs/scripted_* | head -1)"/3_*/report.html   # newest run's full report
xdg-open "$(ls -td ~/sim_runs/scripted_* | head -1)"/summary.html      # newest run's one-line summary
```

Expected:

| Run | Safety | Task outcome |
|---|---|---|
| 1 (kit) | PASS 3/3 | placed 3/3 |
| 2 (box) | PASS 3/3 | placed 3/3 |
| 3 (forced FAIL) | **FAIL 0/1**, rule `joint_margin`, within ~0.03 s | none |

How to read `report.html`:
- **Safety (the verdict)** at the top is the only pass/fail.
- **Task outcome** gives the counts of none / grasped / lifted / placed.
- **Reported metrics vs criteria** lists timing gates. "over" there is informational: tracking lag and gripper lag
  are expected with the scripted stand-in.
- The **per-episode table**:
  - `gt = agree` means the live judge and the offline re-check gave the same result;
  - `slip_mm` is the eraser's displacement, so roughly 100 mm here is the carry to the target, not slip;
  - the **Metrics** list has `ee_speed_max_m_s` and `joint_margin_min_deg`, the safety margins.

## Environment

`./setup_env.sh` builds `~/envs/env_sim` (Python 3.10, mujoco 3.3.0, robotmq, real-env, sensor_rig; no torch; about 0.8 GB).
`run_viewer.sh` runs it automatically the first time if the env is missing. It needs conda (Anaconda, Miniconda or
Miniforge; found via `$CONDA_EXE`, `PATH`, or `~/miniforge3`, `~/miniconda3`, `~/anaconda3`). The host config
(`../configs/station/host/pc_laptop.yaml`) uses `~` paths, so it works for any user. Always
run with `env -u PYTHONPATH -u MUJOCO_GL` from this directory (`franka_sim` imports from the cwd).

## Run it

```
./run_viewer.sh [POSES=3] [--box] [--set key=value ...]
```

Runs `python -m franka_sim eval --profile sim-rig --policy scripted_pick_place --viewer --poses POSES
--episodes-per-pose 1` with the MuJoCo viewer (`DISPLAY` defaults to `:1`), results in
`~/sim_runs/scripted_<time>/`. `--box` swaps the finger kit for box fingers (+10 cm); any further arguments go to
`eval` (e.g. `--set criteria.safety.joint_margin_deg=60`). Headless: call `python -m franka_sim eval ...` without
`--viewer`. Other subcommands: `python -m franka_sim --help` (`check-assets`, `model`, `eval`).

The scripted policy (`franka_sim/policies/scripted/`, config `../configs/run/policy/scripted_pick_place.yaml`) is a
stand-in for a trained policy: it reads the simulator's state stream (not the images) and walks ABOVE -> DESCEND ->
CLOSE -> LIFT -> CARRY -> LOWER -> OPEN -> RETREAT -> DONE with min-jerk chunks inside the speed limits. Its phase
changes are in each session's `logs/inference.log`.

## Swap in another policy

Write `../configs/run/policy/<id>.yaml` and a server per [../docs/adding_a_policy.md](../docs/adding_a_policy.md)
(five robotmq topics, K = 16 relative chunks; check it with `python -m franka_deploy policy-contract`), then run
`./run_viewer.sh` with `--policy <id>` after it (the last `--policy` wins), or call `eval --policy <id>` directly.

## Outputs

- `~/sim_runs/scripted_<time>/summary.html`: the run's summary; per stage `3_<policy>_.../report.html`,
  `report.json` (`rows`: per episode Safety, task outcome, judge vs offline ground truth, tracking p95, wall time) and
  `metrics.json`.
- `3_.../work/session_*/`: the launcher session (resolved configs in `config/`, process logs in `logs/`, recorded
  episodes with `controller_trace.npz`, `bridge.jsonl` and `sim_truth.json` in `task/episodes/`).

## Placeholders still open

- Rig cameras: `sensor/rig1_wifi.yaml` K is the virtual-iPhone K and the extrinsics are iPhone values; replace with the
  IMX519 / IMX219-160 calibration (the scripted policy ignores images, a trained one will not).
- `tooling/franka_hand_iphumi_rig1.yaml` `C_T_TCP` is a copy of the iPhone17,1 entry; measure the rig.
- Finger kit tag positions (the kit's tags sit ~23 mm too far inward per side vs. the camera check; measure the real
  gripper) and the adapter CAD (D9: 8 mm adapter, camera on -y, currently a parametric block).
