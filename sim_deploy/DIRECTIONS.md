# sim_deploy: directions

Run the laptop's real `policy_deploy` against a simulated Pi and PC (MuJoCo) and write the results to `sim_evals/`.
How it works: [PLAN.md](PLAN.md), [DESIGN.md](DESIGN.md). Placeholders: `<RUN_FOLDER>` = a training run folder with a
fetched checkpoint (e.g. `~/Capstone/data/<task>/policy/<run>`), `<ZARR>` = a packaged dataset
(`~/data/umi/datasets/<task>/<dataset>/<dataset>.zarr.zip`), `<i>` = an episode index in it, `<EVAL>` = an eval folder.

Always first, in every terminal:

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
```

## 1. Set up (once per machine)

```bash
sim_deploy/setup_envs.sh light torch sim            # ~/envs/sim_policy_deploy{,_torch}, ~/envs/sim_deploy
~/envs/sim_deploy/bin/pip install -e sim_deploy     # the sim_deploy package
```

Build the sim PC (once, and after a change in `deployment/control_deploy/franka_control` or `sim_deploy/sim_pc`):

```bash
cd deployment/control_deploy/franka_control
cmake -S . -B build-sim -DCMAKE_BUILD_TYPE=Release -DFRANKA_CONTROL_SIM=ON -DFRANKA_CONTROL_WITH_LIBFRANKA=OFF -DFRANKA_CONTROL_BUILD_TESTS=OFF
cmake --build build-sim -j8
cd ~/Capstone-main-sim
```

## 2. Build the model (after a change in `sim_deploy/stations/sim_rig.yaml`, the rig, the kit or the builder)

`run.sh` builds the look's model itself when it is missing or older than its station. By hand, and after a change
of `sim_rig.yaml` (the looks are generated from it):

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.look.make_stations                # the look stations from sim_rig.yaml
~/envs/sim_deploy/bin/python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml
~/envs/sim_deploy/bin/python -m sim_deploy.model build --station sim_deploy/stations/sim_rig_look_robot.yaml
~/envs/sim_deploy/bin/python -m sim_deploy.model build --station sim_deploy/stations/sim_rig_look_train.yaml
~/envs/sim_deploy/bin/python -m sim_deploy.scene check                     # eraser region at the start pose: table, box, in view
~/envs/sim_deploy/bin/python -m sim_deploy.scene --home deploy check       # ... at the deploy home
```

## 2b. The start pose (once; again when the real arm's value changes)

Every rollout starts like on the robot: the PC's `h` (home), then the PC's `p` (the start pose, `homing.q_start_rad`:
where the training demos started), then the laptop's `c`. The sim's value lives in one file,
`sim_deploy/stations/start_pose.yaml`; until the arm is measured it holds a stand-in. Set it from the PC's
measurement with one command (each one prints the pose in the sim and its checks, and refuses a pose that fails them):

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --from-git                  # upstream's station.yaml (below)
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --from-probe <PROBE_TXT>    # a saved `pc_probe --start-pose` printout
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --q "<q1>,<q2>,<q3>,<q4>,<q5>,<q6>,<q7>"
~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --show                      # the one in use
```

`<PROBE_TXT>` = the terminal output of `pc_probe --seconds 3 --start-pose` on the laptop, saved to a file (its
`q_start_rad: [...]` line is read). `--from-git` reads `deployment/control_deploy/station.yaml`
`franka_control: homing: q_start_rad:` from upstream (below) and fails with "not set yet" if it is null. A running sim
must be restarted to pick up a new value (`run.sh` merges the file at each sim PC start). Then `scene check`.

Where "upstream" is read (`--from-git`, run.sh's start-pose line, the web UI; `sim_deploy/upstream.py`):

- `SIM_DEPLOY_UPSTREAM_REPO`: the upstream git worktree (default `~/Capstone-final-deployment`; this checkout if missing).
- `SIM_DEPLOY_UPSTREAM_REF`: the branch or commit read there (default `rnoro/final-deployment`).

After a rebuild, restart any running sim (every process checks the model sha at start).

## 2c. Start-pose sweep (after the real arm's start pose, the table or the eraser region changes)

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep candidates          # grid x IK x checks -> sim_evals/_start_pose_sweep/candidates.md
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep handoff --q="<q1>,...,<q7>"   # one pose: sim + real-PC checks, p path clearance (JSON)
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep run --stage screen --poses feasible --policies <RUN_FOLDER> --rollouts 1 --timeout 25 --fast-homing --video-fps 0
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep run --stage full --poses <ID>,<ID> --policies <RUN_FOLDER>,<RUN_FOLDER> --rollouts 2 --timeout 60 --fast-homing
~/envs/sim_deploy/bin/python -m sim_deploy.scene.start_sweep summary             # -> sim_evals/_start_pose_sweep/summary.{csv,md}
```

`--fast-homing` merges the sim-only `stations/homing_fast_sweep.yaml` (h and p at 0.6 rad/s; the policy never sees
them). With `run.sh` the same goes through `SIM_DEPLOY_SIM_PC_OVERLAY=<station overlay>` (e.g. `stations/vmax05_g6.yaml`,
check G6 only). If another program on this machine briefly holds a sim port (a test suite's fake PC on :5620), run.sh
retries the sim start: `SIM_DEPLOY_START_TRIES=12 SIM_DEPLOY_START_WAIT_S=10 sim_deploy/run.sh ...`.

## 3. Evaluate

A **rollout** = one closed-loop trial of the policy from one eraser placement ("episode" = a training demonstration;
policy_deploy's own logs still say episode). Rules and record format: [evaltool/SCHEMA.md](evaltool/SCHEMA.md). The
sim decides when a rollout stops; the scores come from `evaltool score`. One sim chain at a time (UI, `run.sh` and
`eval batch` all bind the same ports).
Below, `$PY` = `~/envs/sim_deploy/bin/python`:

```bash
PY=~/envs/sim_deploy/bin/python
```

### 3a. Watch a policy yourself

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.ui          # then open http://127.0.0.1:8770
```

1. Pick a policy on the left (or a replay, or the hold_still baseline).
2. **Rollouts**: how many; **Eraser starts**: random from the seed (default 523: the eraser near where the teleop
   demos had the object), or the W3 start set (the batch's 18 starts; needs
   Arm start = start pose).
3. **Scene**: **Arm start** (start pose, like the real arm; or deploy home only) and **Scene look** (robot cell =
   default, training scene, current). All have the 30 mm eraser.
4. **Policy settings** (folded): bf16, 8 steps, exec_h 12 = the arm's; leave them ("Reset to the arm's").
5. Play (top left, next to the cameras; or Space). The page runs `run.sh` (section 4); top right shows the external
   camera, the two Pi cameras and the policy's view, the Progress panel under them shows where the run is, and the
   results table at the bottom gets the new row.
6. Stop: the Stop button (twice: force). Ctrl-C in the UI terminal stops a running eval first.

The same without the page:

```bash
sim_deploy/run.sh --policy <RUN_FOLDER> --rollouts 1 --seed 0 --tag <TAG> --look <robot|train|current>
sim_deploy/run.sh --policy <RUN_FOLDER> --rollouts 18 --starts set --home start --tag <TAG>   # the W3 start set
```

The video is `<EVAL>/rollouts/000/external.mp4` (section 5).

### 3b. Score and report eval folders

```bash
$PY -m sim_deploy.evaltool score <EVAL> [<EVAL> ...]                         # -> <EVAL>/score.json
$PY -m sim_deploy.evaltool report --select '<REGEX>' --out sim_evals/_reports/<NAME>
$PY -m sim_deploy.evaltool report --batch <BATCH_ID> --out sim_evals/_reports/<NAME>
$PY -m sim_deploy.evaltool frames <EVAL> [--rollout <n>]                     # -> <EVAL>/rollouts/NNN/frames.jpg
$PY -m sim_deploy.evaltool prune-index [--dry-run]                           # drop index.csv rows of deleted evals
```

- `--select` = a regex on the eval's tag; also `--protocol <SHA>` (only evals run under that protocol),
  `--group checkpoint,kind,look` (default), `--root <DIR>` (default `sim_evals`).
- `report` scores in memory; it writes only into `--out`: `report.md`, `report.html`, `rollouts.csv`,
  `checkpoints.csv`, `kind_x_look.csv`. Reports go under `sim_evals/_reports/<NAME>/`. Besides the rates it has
  success within 20 s / 30 s, the start split (demos trained / held-out demo / real-trial poses) and sim vs real
  chunk timing per checkpoint.
- After deleting eval folders, `prune-index` drops their `index.csv` rows (backup in `sim_evals/_logs/`).
- Older eval folders (no `gt.jsonl`) are scored from `trace.json`; what they cannot give is listed as `missing`.
- Metrics, validity and groups: [evaltool/README.md](evaltool/README.md).

### 3c. Unattended batch

Before running:

- the laptop on AC power;
- `powerprofilesctl set performance`;
- close heavy apps (browsers with video, IDE indexing, other sims, test suites);
- `uptime`: load average (1 min) < 3 (each run waits while it is >= 3).

```bash
$PY -m sim_deploy.eval batch --plan sim_deploy/eval/batches/<PLAN>.yaml --dry-run    # run list, wall time, checks
$PY -m sim_deploy.eval batch --plan sim_deploy/eval/batches/<PLAN>.yaml
$PY -m sim_deploy.eval batch --resume <BATCH_ID>                                     # after a stop or crash
```

- Plans: `batches/smoke.yaml` (2 checkpoints x 2 looks x 2 rollouts, random placements), `batches/w3.yaml` (all 14 x
  robot x 18, the start set). Keys: [eval/batch.py](eval/batch.py) docstring.
- `starts: set` (W3): every checkpoint gets the same 18 eraser starts of `eval/start_sets/w3.yaml`: rollouts 0-9 =
  the shared training demos 0-9 (one of them the checkpoint's held-out demo), rollouts 10-17 = the poses of the 8
  real-arm trials. The dry run recomputes the file and prints each checkpoint's held-out rollout. Build or rebuild it
  (after a new checkpoint, dataset, start pose or model) and look at it:

```bash
$PY -m sim_deploy.eval starts build      # the checkpoints of batches/w3.yaml (~30 s, reads each checkpoint's split on the CPU)
$PY -m sim_deploy.eval starts show       # the 18 starts, each checkpoint's held-out demo, the checks
$PY -m sim_deploy.eval starts check      # recompute the file in the current scene
```

- `<BATCH_ID>` = `<name>_<YYYY-mm-dd_HHMM>`, printed at the start. State: `sim_evals/_batches/<BATCH_ID>/`
  (`plan.yaml`, `manifest.jsonl`, `logs/`).
- Each run goes through `run.sh`; it waits for the GPU lock (`sim_evals/.gpu.lock`), a UI run, busy sim ports and the
  load. Rollouts `evaltool score` marks invalid are re-run (`invalid_retries`, the driver's `--only-rollouts LIST`
  with the same placements).
- GPU care: each run goes in run.sh calls of at most `piece_rollouts` (6) new rollouts in the same eval folder;
  before every call the batch waits until the GPU is below `gpu_cool_c` (65 C; at most `gpu_cool_max_s` 900 s, then a
  WARN) and `gpu_pause_s` (60 s) have passed since the previous call. Rollouts themselves are never slowed. Plans
  without these keys (also a frozen one on `--resume`) get these defaults; `piece_rollouts: 0` = one call per run.
- GPU fault (kernel `NVRM: Xid` or nvidia-smi not ready): the batch stops with exit 2. After an Xid it writes
  `<EVAL>/gpu_fault.json`; the rollouts running at or after the Xid become invalid. Reboot, check `nvidia-smi`,
  then `--resume <BATCH_ID>`: it scores that run and re-runs those rollouts.
- At the end the report is written to `sim_evals/_reports/<BATCH_ID>/`.

### 3d. Real arm trials

1. Copy [evaltool/labels_template.yaml](evaltool/labels_template.yaml) into the trial folder as `labels.yaml`
   right after the trial (or collect several trials in one file under `trials:`). Fill in outcome, stop_reason,
   grasped, lifted, note.
2. Confirm it: `draft: false`. With `draft: true` the trial is imported but unlabelled (counted, not rated).
3. Import, score, report:

```bash
$PY -m sim_deploy.evaltool import-real <TRIAL_DIR> [<TRIAL_DIR> ...] [--labels <LABELS_YAML>]
$PY -m sim_deploy.evaltool score sim_evals/_real/<task>/<run>/<session>
$PY -m sim_deploy.evaltool report --root sim_evals/_real --group checkpoint,origin --out sim_evals/_reports/<NAME>
```

- `<TRIAL_DIR>` = `~/Capstone/data/<task>/policy/<run>/deploy/<date>_<HH-MM>_trialN_probe_live/`.
- Output: `sim_evals/_real/<task>/<run>/<session>/` (eval-like folder, `source: real`).
- Future trials: run with `manager.save_tap: true` so the TCP path is imported too.

## 4. Run an eval

One command starts the sim PC and the sim Pi, runs the rollouts, and stops everything at the end:

```bash
sim_deploy/run.sh --policy <RUN_FOLDER> --rollouts 20 --seed 0 --tag <TAG>                 # look robot (default)
sim_deploy/run.sh --policy <RUN_FOLDER> --rollouts 20 --seed 0 --tag <TAG> --look train    # the training scene
sim_deploy/run.sh --policy hold_still --rollouts 3 --tag hold                  # baseline: the arm must not move
sim_deploy/run.sh --policy "replay:<ZARR>#<i>" --rollouts 1 --tag replay      # a recorded episode, open loop
```

Options (all go to the driver; `python -m sim_deploy.eval run --help` lists them):

| Option | Meaning |
|---|---|
| `--home start` (default) | each rollout: `h`, then the PC's `p` to the start pose (section 2b), the eraser placed ahead of the fingertips as in the demos (`scene/randomization_start.yaml`), then `c`. `--home trainstart` = this |
| `--home deploy` | `h` only: fingers straight down, 35.5 cm above the table, the eraser under them (`scene/randomization.yaml`); the sim PC runs with no start pose (the real PC refuses `c` away from a set one) |
| `--look robot` (default) | the scene both sims load: `robot` = the real robot cell (`stations/sim_rig_look_robot.yaml`), `train` = the training scene (`sim_rig_look_train.yaml`), `current` = `sim_rig.yaml`; all with the 30 mm eraser. Recorded in `eval.yaml` and `index.csv` |
| `--precision bf16 --diffusion-steps 8 --exec-h 12` | the defaults (the arm's settings); another value only to experiment |
| `--timeout 60` | seconds per rollout |
| `--tap` | save the policy's inputs and outputs (the four images every 10 steps) in the session |
| `--dry-run` | check the eval folder, station, checkpoint and placements without starting the sim |
| `--starts set` | rollout n: the eraser at start n of the start set `eval/start_sets/w3.yaml` (section 3c) instead of a random pose; `set:<FILE>` for another file |
| `--only-rollouts LIST` | run only these rollout indices (e.g. `3,7`) with the same placements as in a full run of that `--seed` (batch retries) |
| `--start-pose <FILE>` | another start pose file (format of `stations/start_pose.yaml`; run.sh merges it into the sim PC's station too) |

Use the same `--seed` for every policy you compare: rollout n always gets the same eraser pose.

At each start `run.sh` prints whether `stations/start_pose.yaml` equals the real arm's (upstream's `station.yaml`
q_start_rad; upstream = `SIM_DEPLOY_UPSTREAM_REPO` / `SIM_DEPLOY_UPSTREAM_REF`, section 2b). A framed `start pose WARNING` means they differ; it runs anyway with the sim's file.
Adopt the arm's with `~/envs/sim_deploy/bin/python -m sim_deploy.scene start-pose --from-git` (section 2b).

If the sim PC latches a fault (the gate refused a waypoint, a reflex), the real PC would need a restart, and so does
the sim: `run.sh` restarts the sim PC and sim Pi and continues the same eval folder (`--resume <EVAL>`), up to 5 times.

## 5. Results

```
sim_evals/index.csv                                    one row per eval
sim_evals/<task>/<run_folder>/<date>_<time>_<tag>/     a learned policy
sim_evals/_baseline/hold_still/...                     hold_still
sim_evals/_replay/<task>/<dataset>_ep<i>/...           a replay
sim_evals/_logs/<date>_<time>/                         run.sh: sim_pc.log, sim_pi.log, eval.log
```

- The written-up experiments (question, setup, commands, results, conclusions; tracked in git): [../sim-experiments/README.md](../sim-experiments/README.md).
- `summary.html` (open in a browser): the eval's settings, then one row per rollout: outcome (success, fail,
  timeout, fault, stopped, interrupted), grasped / lifted, highest lift, times, faults, and a link to the
  external-camera video (`rollouts/<n>/external.mp4`, mp4v: VLC or ffmpeg plays it).
- `index.csv`: `successes` / `rollouts`, `success_rate` with its Wilson 95 % interval (`ci95_lo`, `ci95_hi`),
  `grasp_rate`, `lift_rate`, `faults`, `stops`, `timeouts`, and `eval_dir`. Interrupted rollouts are listed in
  `rollouts.jsonl` but are in no rate.
- `metrics.json`: the same plus inference (`policy_ms_p50`, `policy_ms_p95_max`), step margins (`margin_ms_min`) and
  skipped steps by reason. `eval.yaml` freezes everything the eval used (station, seeds, home, checkpoint, commits).

## 6. Stop

- Ctrl-C in the `run.sh` terminal: the rollout in progress is recorded as `interrupted`, results are written, the
  sim PC and sim Pi stop. On exit run.sh also ends anything else it started that is still running (policy_deploy's
  frames / policy / launcher; env `SIM_DEPLOY_RUN_TAG`, `eval/procs.py`) and prints a line for each.
- Check that nothing is left: `ss -ltn | grep -E ':(5555|56[0-9]{2}) '` prints nothing.

## 7. Common failures

| You see | Do |
|---|---|
| `port 5555 is already in use` | another sim or policy_deploy runs: stop it (`pkill -f franka_control`, `pkill -f sim_deploy.sim_pi`, `pkill -f policy_deploy`) |
| `the sim PC runs model ..., model.json says ...` | the model was rebuilt while a sim ran: stop it and start again |
| `no <RUN_FOLDER>/... checkpoint` / `use failed` | the run folder has no fetched checkpoint (`train/fetch.sh` on the training branch) |
| `--home deploy: after homing the TCP is ...` | the sim PC runs another home: start it with `run.sh` (not by hand) |
| `start_pose.yaml: q_start_rad is not set` | set it (section 2b), or run `--home deploy` |
| `key p: ... the sim PC runs another start pose` | `start_pose.yaml` changed while the sim ran: stop it and start `run.sh` again |
| `key p: scene control key: key p refused: not at home: press h first` | the arm did not reach home first: read `sim_pc.log` in the run's `_logs` folder |
| `c: the PC refused start: ... not at the start pose` | the sim PC has a start pose but the rollout began elsewhere: use `run.sh` (it matches the sim PC to `--home`) |
| `safety gate rejected waypoint ...: outside workspace box` | the policy left the gate's box (sim stations: x >= 0.20, z >= -0.005: the table is at z 0); the rollout is a fault, run.sh restarts the sim |
| skipped steps / `margin_ms` < 0 | inference too slow: `--precision bf16 --diffusion-steps 8 --exec-h 12`, laptop on AC power |
| `policy_deploy did not reach idle` | read `<EVAL>/policy_deploy.log` (its last FAILED line names the process) |
