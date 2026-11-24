# sim_deploy/eval: run a policy for N rollouts and write sim_evals/

A **rollout** is one closed-loop trial of the policy from one eraser placement (policy_deploy's manager calls it an
"episode" in its keys and log lines; in sim_deploy "episode" means a training demonstration of a zarr dataset).

The eval driver runs the copied `policy_deploy run --mock` against the running sim (sim PC + sim Pi) exactly as an
operator would: it presses the laptop's keys in a pseudo-terminal and the PC's keys (`h`, `p`) through the sim PC's
scene control (P1's `key`), places the eraser between rollouts through the sim PC's
scene control, judges every rollout from the sim's ground truth, and writes the results (PLAN.md section 7).
Usually started by `../run.sh`, which brings the sim PC and sim Pi up first.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
sim_deploy/run.sh --policy ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --rollouts 20 --seed 0 --tag first
# the driver alone (the sim PC and sim Pi already running):
~/envs/sim_deploy/bin/python -m sim_deploy.eval run --policy hold_still --rollouts 3 --seed 0 --tag g5
~/envs/sim_deploy/bin/python -m sim_deploy.eval run --policy "replay:$HOME/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#2" --rollouts 1 --tag g6
# check everything that can be checked without the sim (no sim process needed):
~/envs/sim_deploy/bin/python -m sim_deploy.eval run --policy <...> --rollouts 3 --dry-run
```

| Option | Meaning |
|---|---|
| `--policy` | a training run folder (`~/Capstone/data/<task>/policy/<run>/`), `hold_still`, or `replay:<zarr>#<index>` (the `policy_deploy use` syntax) |
| `--rollouts N --seed S` | rollout n uses the eraser pose of `rng([S, n])` (`../scene/README.md`); the replay always uses its replay placement |
| `--starts random\|set\|set:<FILE>` | `random` (default): as above. `set`: rollout n puts the eraser at start n of the start set `start_sets/w3.yaml`, the same for every checkpoint (below); `set:<FILE or name>`: another set. Needs `--home start`; not with a replay; `--rollouts` at most the set's 18 |
| `--tag T` | the end of the eval folder's name |
| `--timeout 60` | seconds per rollout before it counts as a timeout |
| `--tap` | `manager.save_tap: true` (observation, chunk and waypoints per step; the four images every 10 steps) |
| `--precision`, `--diffusion-steps`, `--exec-h` | passed to `policy_deploy use` (default: what `stations/policy_sim.yaml` has) |
| `--behind-tip` | replay placement: the eraser centre this far behind the fingertip point (default half its resting height) |
| `--video-fps 10` | external-camera video rate (0 = none) |
| `--dry-run` | prepare the eval folder and check it, no sim needed (below) |
| `--home start\|deploy` | P2. `start` (default): each rollout `h`, then the PC's `p` to `stations/start_pose.yaml`'s q, the eraser relative to that pose (`scene/randomization_start.yaml`). `deploy`: `h` only, `scene/randomization.yaml`. `run.sh --home` starts the sim PC to match (`deploy`: `q_start_rad: null`). `trainstart` = `start` |
| `--home-key auto\|laptop\|pc` | how `h` is pressed: the laptop's key (C3 home) or the PC's (scene `key`); `auto` (default): the laptop's if the copy's `policy_deploy` still has one (the ported start-pose change removed it, so today `pc`) |
| `--only-rollouts I,J` | run only these rollout indices with `--seed`'s keys `[seed, n]`: the same placements as those rollouts of a full run (batch retries); folders keep the indices |
| `--resume <EVAL>` | continue an eval after the sim PC latched a fault (exit code 3; `run.sh` restarts the sim and does this itself) |

## What one eval does

1. Resolves the target: task and run folder (`sim_evals/<task>/<run_folder>/`; `hold_still` goes under `_baseline/`,
   a replay under `_replay/<dataset folder>/<dataset>_ep<index>/`), the checkpoint, the training run's git commit
   (`train_meta.json clone_git_commit`) and its policy type (plain diffusion or behavior prompting, from
   `run_config/job_info.json` and the station after `use`).
2. Checks `sim_deploy/build/sim_rig/` against its `model.json` sha.
3. Freezes the station: `stations/policy_sim.yaml` copied to `<eval>/station.yaml` with `session_root` = `<eval>/session`
   (and `manager.save_tap` with `--tap`), then the copy's `policy_deploy use <target> --station <eval>/station.yaml`
   (log: `use.log`).
4. Writes `eval.yaml` (frozen: the station file, the seeds, the scene settings, the judge, both git commits, the
   checkpoint, the model sha, the replay placement, and `home`: the variant, how `h` is pressed, `q_home_rad`, the
   tolerance, the rollout-start TCP, the sim PC station, and for `start` the start pose used: its q, the file's text,
   TCP, tilt, heading, height above the table, the checks and the distance from the demos).
5. Connects: scene REQ to :5651 (`hello`: the sim PC's model sha must equal model.json's), C2 SUB :5620 (read
   only, decoded with `deploy_contracts`), state SUB :5650 (the driver's own MuJoCo copy renders `cam_external`).
6. Starts `~/envs/sim_policy_deploy/bin/policy_deploy run --mock --station <eval>/station.yaml` in a pty (cwd
   `deployment/policy_deploy`, no PYTHONPATH; the launcher starts the torch env for diffusion) and waits for `[idle]`.
   Its whole terminal output goes to `policy_deploy.log`.
7. Per rollout: `h` (scene `key h`: wait for C2 `homing`, then `idle` with every joint within
   `homing.at_pose_tol_rad` 0.05 of `homing.q_home_rad`; with a laptop `h`: "home pose reached"); `--home start`:
   scene `key p` (a refusal, e.g. "not at home: press h first", ends the rollout as `start_pose_failed`), wait for C2
   `homing` within 3 s, then `idle` with every joint within the tolerance of `start_pose.yaml`'s q (else the eval
   stops: the sim PC runs another start pose, restart it); `--home deploy`: check the TCP is at the deploy home.
   Record where the arm is (`start_pose`: C2 q, TCP, tilt, height above the table, largest joint error, width);
   `set_object` the eraser (seeded `random_placement`, recorded as `placement` with `clipped`; `--starts set`: the
   file's start n, recorded as `placement` source `demo` / `real_trial` with `held_out`); wait `settle_s`; read
   its start height; `c`, wait for "episode N started" (`c` is pressed again while the manager refuses it, e.g.
   "waiting for the PC clock map"; a "not at the start pose" refusal ends it as `start_failed` with a hint); then
   poll ground truth at 20 Hz until one of:
   - **success** (`../evaltool/SCHEMA.md` scoring constants; the live judge only decides when to stop, the scores
     come from `evaltool score`): the eraser centre at least 5 cm above its settled height and its lowest corner at
     least 1 cm above the table, both in every poll for 1 s (grasp not required; the driver then presses `y`);
   - **interrupted** (E7): a SIGINT / Ctrl-C during the rollout; recorded, in no rate;
   - **timeout** (`--timeout`, default 60 s), the eraser off the table, C2 mode `fault` or `stopped`, the manager ending the
     rollout itself, or policy_deploy exiting (the driver presses `n` if the rollout is still running).
   Also recorded: grasped (both fingers' pads or holders touching the eraser) and when, lifted (grasped and
   2 cm up), the highest lift, anything touching the table (fingers, rig, arm), arm reflexes, C2 mode counts and
   errors, and the gate counters over the rollout.
   If policy_deploy exits (a fault concludes it), it is started again for the remaining rollouts.
8. After the last rollout: `s`, `q`, then each rollout's slice of the session is collected and the results written.

## Start set (`--starts set`, `start_set.py`, W3 2026-11-22)

One committed file, `start_sets/w3.yaml`, with the same 18 eraser starts for every checkpoint (it replaced the
per-dataset demo starts of the same day):

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.eval starts build      # the checkpoints of batches/w3.yaml (--plan, --checkpoints)
~/envs/sim_deploy/bin/python -m sim_deploy.eval starts show
~/envs/sim_deploy/bin/python -m sim_deploy.eval starts check      # the file recomputed in the current scene
```

- **Rollouts 0-9: the shared demos 0-9.** Every training dataset begins with the same 10 demos (pick1-ep10-v1;
  `build` checks names and data). Each placed by the replay placement (`../scene/README.md` "Replay placement"): the
  demo's start-to-grasp transform applied to the sim's start TCP, the eraser centre half its height behind the grasp
  fingertip point, its width along the closing axis, laid flat on the table (the vertical mismatch is reported as
  `height_error_m`).
- **Held-out demo.** Per checkpoint the split it trained with (`_extra_training_split_info` from the checkpoint, read
  on the CPU by `split_probe.py` in `~/envs/sim_policy_deploy_torch`, cross-checked against behavior_prompting's
  `get_val_mask` recomputed on its dataset). Every checkpoint held exactly one of the 10 out; that rollout is recorded
  `held_out: true` (rollout.json, `placement.held_out`, eval.yaml `scene.start_set.held_out_rollouts`).
- **Rollouts 10-17: the real-trial poses.** The eraser centre x, y and yaw fitted in W2 to the 8 real-arm trials of
  2026-11-20 (`sim_evals/_pipeline/W2/sets/robot_trials_fit.json`, copied into the file with its sha256); the sim's
  30 mm eraser flat there (the real one was 51 mm wide).
- **Checks** (all must pass; a failure is reported, nothing is substituted): the whole eraser on the table, inside the
  PC's workspace box, all corners in the IMX219's view at the start pose.
- The driver recomputes every start with its model and start pose and refuses the file if one moved (> 0.5 mm /
  0.1 deg) or fails a check, or if the checkpoint is not in the file's `checkpoints` (its held-out demo unknown).
  `--only-rollouts` and `--resume` use the same starts (resume: the ones frozen in eval.yaml).

## The rollout record (`sim_deploy.rollout/v2`, `../evaltool/SCHEMA.md`)

Every eval, whoever starts it, also writes what `evaltool score` reads (helpers in `gt_recorder.py`):

- `eval.yaml`: `record_schema`, `launcher` (env `SIM_DEPLOY_LAUNCHER`, default `cli`), `batch_id` (env
  `SIM_DEPLOY_BATCH_ID`, else null), `protocol` (seed rule and placement source, timeout, precision / diffusion steps /
  exec_h, look, model sha, `q_start_rad` (null for `--home deploy`), the scoring constants, the gate limits v_max /
  w_max / cone / box from the copy's `config.yaml` + `stations/control_sim.yaml`) and `protocol_sha` (sha256 of its
  canonical JSON, sorted keys, first 16 hex).
- `rollouts/NNN/gt.jsonl`: one row per judge poll (20 Hz), `t` on trace.json's zero. `eraser_pose`, `tcp_pose`,
  `gap_m`, `contacts_eraser` (every geom touching the eraser), `contacts_table` (every geom but the eraser touching the
  table) and `reflex` come from the sim PC's scene `ground_truth` (:5651; unnamed geoms carry their body's name);
  `reflex` is false, true, or C2's error text when it names the reflex. `min_z_fingers_rig_m` is computed on the
  laptop: the newest sim.state/v1 qpos (:5650, rows of another model sha dropped) put into the driver's own MuJoCo
  copy of the build (same sha, checked), the lowest z over the **vertices** of the collision geoms of the bodies
  `left_finger`, `right_finger` and `rig` (the rig body holds the mount). `c2_mode`: the newest C2 state's mode.
- `rollouts/NNN/rollout.json` (every rollout, also those that never started): `source` sim, `launcher`,
  `batch_id`, `seed_key`, `placement` (xy, yaw from the placed quaternion, the pose, random / replay / demo /
  real_trial, the sampled draw or the start), `held_out`, `eraser_settled_pose`, `start_pose_q`, `outcome`, `stop_reason` (success / timeout / fault: C2 fault or
  stopped, the manager ending the rollout, the eraser off the table / user: Ctrl-C / error: policy_deploy exited, the
  scene failed, or no rollout started), `clamps` (manager.log lines "clamped into the workspace box" between the
  manager's "episode N started" and "episode N: ..." lines, else the driver's wall-clock window), `c2` (gate counter
  deltas, loop_late, modes, errors), `step_summary`, `loadavg_start` / `loadavg_end` (1-min), `loop_late`, plus
  `wall_start` / `wall_end`. Written when the rollout ends and again after the session is collected.

## Results

```
sim_evals/index.csv                         one row per eval (date, task, run, policy, kind, dataset, rollouts, seed,
                                            tag, successes, rate, Wilson 95 % CI, grasp/lift rates, faults, stops,
                                            timeouts, folder)
sim_evals/<task>/<run_folder>/<YYYY-MM-DD_HHMM>_<tag>/
  eval.yaml  station.yaml  use.log  policy_deploy.log  start_pose.yaml (--home start: the file as used)
  rollouts.jsonl                            per rollout: seed key, outcome, grasp/lift/success and their times, faults,
                                            stops, table contacts, C2 modes, gate counters, step summary
  metrics.json                              success / grasp / lift rates with Wilson 95 % intervals, outcomes, times,
                                            steps sent / skipped (by reason), policy ms, cycle ms, margins
  summary.html                              the report (one table row per rollout, video links)
  session/<date>_mock/                      the whole policy_deploy session folder (all rollouts)
  rollouts/<n>/                             trace.json (lift, grasp, finger gap at 20 Hz), gt.jsonl, rollout.json,
                                            external.mp4,
                                            session_rollout.json + steps.jsonl (that rollout's lines of the session: its
                                            line of manager/episodes.jsonl and its steps),
                                            tap/ (with --tap)
```

Videos are MPEG-4 part 2 (`mp4v`): opencv-python-headless has no H.264 encoder. VLC and ffmpeg play them; some
browsers do not. They play in real time (the rate is the one actually rendered, at most `--video-fps`).

Dry runs go to `sim_evals/_dry_run/...` and add no `index.csv` row. They check: the model files against
model.json, the frozen station and `use`, the checkpoint and prompt dataset paths, the placements (or the replay
placement and its height error; with `--starts set` the file, this checkpoint's split and held-out rollout and one
row per start with its checks), the home (how `h` is pressed; `start`: the start pose's checks and that the copy's
config.yaml has `homing.q_start_rad`), `policy_deploy check --mock` (`check.log`; without the sim Pi and a terminal its
"Pi reachable" and "manager terminal" rows fail, the rest must pass), and which sim ports are in use.
