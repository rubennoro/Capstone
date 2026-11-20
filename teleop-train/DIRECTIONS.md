# teleop-train: directions

From teleop episodes to a trained policy on the arm. Every command runs on the laptop.

```
teleop_collect run ──▶ ~/CapstoneTeleop/<task>/episode_NNN/           (record; read only from here on)
teleop_train convert ─▶ ~/Capstone/data/<task>/episode_NNN/processed/  (one training zarr per episode)
teleop_train package ─▶ ~/Capstone/data/<task>/package/<dataset>/      (the .zarr.zip the cluster trains on)
train/all.sh ─────────▶ ~/Capstone/data/<task>/policy/<run>/           (upload, train, fetch, eval)
policy_deploy use ────▶ the policy on the arm
```

Two names appear in every command. Set them once per terminal:

```bash
TASK=pick_block          # the teleop task: the folder under ~/CapstoneTeleop/ (lowercase letters, digits, _)
DATASET=pick_block-v1    # a new name for each package: letters, digits, - and _; never reused
```

## 1. Set up (once)

```bash
cd ~/Capstone-teleop-collect-deploy && unset PYTHONPATH
python3.10 -m venv ~/envs/teleop_train
~/envs/teleop_train/bin/pip install --upgrade pip
~/envs/teleop_train/bin/pip install -e "teleop-train[test]"
~/envs/teleop_train/bin/teleop_train --help
```

**Good:** the help lists `convert` and `package`. Training also needs `rnoro/training` in `~/Capstone`, set up as its
`train/SETUP.md` says (already done on this laptop).

## 2. Record episodes

Follow [../deployment/teleop_collect/DIRECTIONS.md](../deployment/teleop_collect/DIRECTIONS.md): its steps 9-11
cover the session routine, collecting and the keys.

```bash
~/envs/teleop_collect/bin/teleop_collect check
~/envs/teleop_collect/bin/teleop_collect run       # then the page at http://127.0.0.1:8765
```

- In the page, create the task `$TASK`, then for each demo: Home, Start, drive, Stop, then save as **success** or
  **fail**.
- Keep each demo under about 60 s. Behavior prompting reads at most 60 s per demo (§7 of PLAN.md).
- Episodes land in `~/CapstoneTeleop/$TASK/episode_NNN/`.
- Before recording, the PC and this branch must have the same `contracts/frames.yaml` (the TCP); `check` says so. To
  see where the TCP axes point, use `pc_probe --axes` (teleop_collect DIRECTIONS, "pc_probe").

## 3. Convert

```bash
~/envs/teleop_train/bin/teleop_train convert $TASK
```

Reads `~/CapstoneTeleop/$TASK/` (never writes there) and writes one training zarr per episode into
`~/Capstone/data/$TASK/episode_NNN/processed/`. It prints one row per episode:

```
episode  outcome    raw s   kept s   steps  state       reason
      0  success      8.9      5.7     342  converted
      3  success      5.7      5.6       -  excluded    gap of 333.3 ms between consecutive main frames inside the kept span (limit 100 ms)
3 converted, 0 up to date, 1 excluded; 3 of 4 usable
```

- `kept s` is the episode with the still time at both ends trimmed: 0.5 s is kept before the first motion and after
  the last.
- Run it again any time. Episodes already converted show `up to date` and are skipped.
- An `excluded` episode is never packaged. Its reason is in its `status.json`, and section 7 below says what each
  reason means.
- `fail` episodes are converted too; `package` leaves them out unless told otherwise.

| Option | Use |
|---|---|
| `--episodes 3,7-9` | only these episode numbers |
| `--force` | convert again even when up to date |
| `--pose-source measured` | record the measured TCP pose instead of the commanded one (default `target`; the deployment must then use the same) |
| `--raw-root DIR` / `--out-root DIR` | other folders (defaults `~/CapstoneTeleop`, `~/Capstone/data`) |

**Exit code:** 0 if at least one episode is usable, 1 otherwise or on a refusal (one line saying why).

## 4. Package

```bash
~/envs/teleop_train/bin/teleop_train package $TASK $DATASET
```

It converts anything not yet converted, then joins the usable `success` episodes, in episode order, into
`~/Capstone/data/$TASK/package/$DATASET/` (`$DATASET.zarr.zip` + `package.json`). It prints what went in, what was
left out and why, and the next command.

| Option | Use |
|---|---|
| `--episodes N` | only the first N usable episodes (e.g. 10, 20, 30 for a data-size comparison; each a new `$DATASET`) |
| `--include-fail` | also package episodes saved as `fail` |
| `--overwrite` | replace a dataset of that name (only if it was never uploaded: the cluster refuses a changed zip under an old name) |
| `--pose-source`, `--raw-root`, `--out-root` | as for `convert` |

**Refused (nothing written):**
- the episodes don't share one rig and one TCP;
- their TCP differs from this branch's `deployment/contracts/frames.yaml` (the TCP the deployment runs);
- their rig differs from `deployment/contracts/rigs/rig_pi_v3_640.yaml` (the rig the deployment preps images with);
- the name already exists;
- no episode is usable.

## 5. Train

**Choose the policy** with `--policy` on each run:

```bash
cd ~/Capstone
train/all.sh $TASK $DATASET --policy plain        # plain diffusion (umi_policy_dunet)
train/all.sh $TASK $DATASET --policy prompting    # behavior prompting (umi_policy_dunetp)
```

- **Fixed per run:** `--policy` goes to the cluster with the job and is recorded with the run, so the next run can
  use the other one. Nothing to switch back.
- **Without `--policy`:** the scripts train behavior prompting.
- **Plain runs:** no rollout or attention videos (`eval.sh --attention` refuses them); the offline eval works for
  both.

`all.sh` takes the package as it is in `data/$TASK/package/$DATASET/`, because it runs without `--pull` / `--package`.
It then uploads, runs a smoke run and the full run on an H200, waits, fetches, writes the reports, and runs the
offline eval. It takes hours, so start it in `tmux`. If it stops after submitting, it prints the line to pick up again:
`train/all.sh $TASK --run <run>`. Its dataset-report step expects handheld files and may fail; that step is not fatal,
and training goes on.

The same, one step at a time:

| Step | Command |
|---|---|
| upload | `train/upload.sh $TASK $DATASET` |
| submit (3-epoch smoke run first) | `train/submit.sh $TASK $DATASET --policy plain --smoke-first --note "teleop v1"` |
| watch | `train/status.sh $TASK <run> --wait` |
| fetch | `train/fetch.sh $TASK <run>` |
| offline eval | `train/analysis/eval.sh $TASK <run> --ckpt top1 --per-demo 8` |

Hydra overrides go after `--`. For example, behavior prompting with demos longer than 60 s:

```bash
train/submit.sh $TASK $DATASET --policy prompting -- task.shape_meta.max_sequence_length=1800
``` Details: `~/Capstone/train/README.md` and `train/DIRECTIONS.md`.

## 6. Run it on the arm

In `~/Capstone-final-deployment/deployment/policy_deploy` (its DIRECTIONS.md, "A trained policy"):

```bash
~/envs/policy_deploy/bin/policy_deploy use ~/Capstone/data/$TASK/policy/<run> --precision bf16 --diffusion-steps 8 --exec-h 12
~/envs/policy_deploy/bin/policy_deploy check
~/envs/policy_deploy/bin/policy_deploy run
```

- On the real arm, `check` refuses any trained policy until `contracts/frames.yaml` says `settled: true`. Its step 4
  covers settling: `pc_probe --axes` and the camera check.
- The policy must run with the TCP it was trained with, which is the `F_T_TCP` in `package.json`.
- With `--pose-source measured`, the deployment's `pose_source` in `config/manager.yaml` must be `measured` too.

## 7. When an episode is excluded or a command refuses

| Message (start) | Meaning, what to do |
|---|---|
| `COMPLETE: missing` | the recording did not finish (laptop crash); teleop_collect deletes it at its next start |
| `gap of ... ms between consecutive arm states` | arm states missing inside the demo (limit 50 ms): the PC link dropped; record it again |
| `gap of ... ms between consecutive main frames` / `ultrawide frames` | camera frames missing inside the demo (limits 100 / 200 ms): Wi-Fi; record it again |
| `no motion found` | nothing moved above 5 mm/s, 1 °/s or 5 mm/s of gripper |
| `kept span ... is shorter than 2 s` | too short after trimming |
| `...: ... the rig expects 640x480` | the stream size differs from the recorded rig |
| `not a rigid transform` | a bad pose in `robot_states.csv`; report it |
| `... exists without teleop_source.yaml` | `~/Capstone/data/$TASK/` is a handheld folder; use another task name or `--out-root` |
| `dataset ... already exists` | choose a new `$DATASET` (or `--overwrite` if never uploaded) |
| `the episodes' F_T_TCP differs from the deployment's ...frames.yaml` | recorded with another TCP than the deployment runs: record again with the current TCP |
| `the episodes were recorded with rig ..., not the deployment's rig_pi_v3_640.yaml` | recorded with another rig file (its ultrawide images would differ from live): record again with teleop_collect's current `rig_file` |
| `converted with other settings or another teleop_train version` | run `teleop_train convert $TASK` again |
| `... has another rig / F_T_TCP than ...` | episodes from different set-ups: package them separately (`--episodes`, or another task) |
| `--episodes N: only M usable episodes` | ask for at most M |
| `nothing to package` | no usable `success` episode: see `teleop_train convert $TASK` |

## 8. Trying it without the arm

`teleop_collect run --mock` records stand-in episodes into `~/CapstoneTeleop_mock/`: real Pi frames and fake-PC
poses. Convert them into a scratch folder, so they never mix with real data:

```bash
~/envs/teleop_train/bin/teleop_train package $TASK mock-v1 --raw-root ~/CapstoneTeleop_mock --out-root /tmp/teleop_train_mock
```

## 9. Tests (optional)

```bash
cd ~/Capstone-teleop-collect-deploy/teleop-train && unset PYTHONPATH
~/envs/teleop_train/bin/python -m pytest tests -q        # about 2.5 min
```

## Commands

| Command | What it does |
|---|---|
| `teleop_train convert TASK [--episodes LIST] [--force]` | episodes → per-episode training zarrs, with the checks |
| `teleop_train package TASK DATASET [--episodes N] [--include-fail] [--overwrite]` | converted episodes → `DATASET.zarr.zip` + `package.json` |
| both: `[--pose-source target\|measured] [--raw-root DIR] [--out-root DIR]` | |
| `train/all.sh TASK DATASET --policy plain\|prompting` | (in `~/Capstone`) train end to end: plain diffusion or behavior prompting |
| `train/submit.sh TASK DATASET --policy plain\|prompting [--smoke-first]` | (in `~/Capstone`) submit only |
| `policy_deploy use <run folder>` | (in final-deployment) choose the trained policy |
