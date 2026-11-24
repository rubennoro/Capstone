# 03 W4: easy to run any policy (one command, one click)

**Status:** in progress (2026-11-22). Built: look selection, the arm's settings as defaults, `eval batch`, the eval
tool (`evaltool score / report / frames / import-real`). Open: a smoke batch, the UI end to end on a real run · **Date:** 2026-11-20 (planned), 2026-11-21, 2026-11-22 · **Who:** W4 executor + reviewer; this skeleton by the docs lead · **Plan ref:** sim_deploy/PLAN.md §11 (W4 row);
`sim_deploy/ui/DESIGN.md` (U1-U5); `sim_deploy/DIRECTIONS.md`

## Question

Does one command (`run.sh` / `eval batch`) or one click in the UI run any of the 14 checkpoints with its correct
settings (kind, prompt, precision, diffusion steps, `exec_h`) and no hand-typed flags? Yes / no per checkpoint.

## Why it matters for the real robot

Today the settings are flags someone has to remember (`--precision bf16 --diffusion-steps 8 --exec-h 12`, named for BP
only in sim_deploy/DIRECTIONS.md); on the arm upstream uses one `F=...` string for every policy (upstream
policy_deploy/DIRECTIONS.md at `f632576`, "Running trials"). A per-checkpoint settings file read by the catalog, the
driver and the UI makes the sim run each checkpoint as it should run on the arm, and the same file can drive
`policy_deploy use` on the real laptop.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted), after W0 |
| Catalog | `python -m sim_deploy.ui.catalog [--json] [--no-replays]` (`sim_deploy/ui/catalog.py`, U5): lists `<data_root>/<TASK>/policy/run_*/` with the driver's checkpoint rule (`sim_deploy.eval.target.resolve`: the last of `sorted(checkpoints/*.ckpt)`), the kind from `run_config/job_info.json` / `submit_config.env` (no `TRAIN_CONFIG` line = BP) / README / job.json, labels from RUNS.md, replays from `~/data/umi/datasets`, past evals from `sim_evals/index.csv`; imports no torch; writes only `sim_deploy/ui/.cache/replays.json` (skipped with `--no-replays`) |
| Driver | `python -m sim_deploy.eval run` and `batch` (2026-11-22); `run` passes `--precision`, `--diffusion-steps`, `--exec-h` to `policy_deploy use`, defaults bf16 / 8 / 12 since 2026-11-21 (`eval/driver.py`); `--look` (default robot) |
| UI | `python -m sim_deploy.ui [--port 8770]` (127.0.0.1 only); built (U1-U5), not yet used for a real run |
| Settings source | one setting for all: bf16 / 8 / exec_h 12 (SCHEMA.md, until W1's timing bench is redone); a plan can override per checkpoint (`{id, settings}`) |
| Checkpoints | all 14; the end-to-end check on at least one BP and one plain |

## Method

1. **Settings applied automatically**: the catalog shows each checkpoint's settings; the driver reads them when no
   flag is given (an explicit flag still wins, and `eval.yaml` records which was used); the UI's Play uses the same.
2. **`eval batch`**: a list of run folders or "all", same seed, one after another on one chain, a resume after an
   interruption, one summary.
3. **UI end to end**: a real run from the page (pick, Play, live views, results) for one BP and one plain checkpoint,
   results identical in form to a `run.sh` eval folder.
4. **DIRECTIONS**: `sim_deploy/DIRECTIONS.md` command by command for "run any checkpoint", "run all", "from the UI".

Pass: for each checkpoint, one command and one click give an eval whose `eval.yaml` / frozen `station.yaml` show the
settings file's values.

## Commands to reproduce

To be confirmed by the W4 coordinator. Templates:

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
~/envs/sim_deploy/bin/python -m sim_deploy.ui.catalog --no-replays             # the 14 runnable checkpoints
sim_deploy/run.sh --policy ~/Capstone/data/PICK_DOUBLE/policy/run_002_2026-11-19 --rollouts 3 --seed 0 --tag w4_e2e
~/envs/sim_deploy/bin/python -m sim_deploy.eval batch --plan sim_deploy/eval/batches/smoke.yaml [--dry-run]
~/envs/sim_deploy/bin/python -m sim_deploy.evaltool report --batch <BATCH_ID> --out sim_evals/_reports/<BATCH_ID>
~/envs/sim_deploy/bin/python -m sim_deploy.ui                                  # http://127.0.0.1:8770
```

## Results

### 1. Look + the arm's settings (2026-11-21, the user's decisions)

What was built (details: `sim_deploy/ui/DESIGN.md` "W4: look and the arm's policy settings"):

- `run.sh --look robot|train|current` (default `robot`; `sim_deploy/looks.py`): the sim PC and the sim Pi both load
  `build/<look station id>/` (robot = `sim_rig_look_robot.yaml`, train = `sim_rig_look_train.yaml`, current =
  `sim_rig.yaml`); `eval.yaml` `look:`, `index.csv` `look`, a look row in `summary.html`. UI: a look select (default
  robot cell) and a look column in the results.
- Eraser 30 mm wide in all three run looks (`look/make_stations.py` `RUN_LOOKS`): robot 127 x 30 x 12 (physics
  bit-identical to sim_rig, `check_look physics`), train 111 x 30 x 12 (differs from sim_rig only in the eraser
  length). The image-only looks keep the real width; `robot_nocam` follows `robot`.
- Start pose: `run.sh` and the UI compare `stations/start_pose.yaml` with `rnoro/final-deployment`'s `station.yaml`
  q_start_rad at each start (equal today) and warn if they differ; nothing is overwritten.
- Policy settings bf16 / 8 DDIM steps / exec_h 12 are the defaults of the driver, the UI server/runner and the page
  (also for `run.sh` without flags); `eval.yaml` station_policy / station_manager show them.

Smoke (one 20 s rollout of PICK_TEST1 run_006 per look, seed 0, `--home start`, real limits, under the GPU lock):

| look | model sha | outcome | steps sent / skipped | margin min ms | eval folder |
|---|---|---|---|---|---|
| robot | 59dd4765 | timeout (no grasp) | 101 / 0 | +3.7 | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-21_1110_look_smoke_robot` |
| train | 19c22fb7 | timeout (no grasp) | 99 / 0 | -4.1 | `..._1112_look_smoke_train` |
| current | 96822113 | fault: gate, tilt 62.7 deg > 60 + 2 | 76 / 0 | -47.5 | `..._1113_look_smoke_current` |

One sim Pi frame per look (main + ultrawide, mid-rollout) and a mosaic: `sim_evals/_pipeline/W4/look_smoke/`; each
shows its look (robot: white table + the cell's backdrop; train: wood, lime eraser, blue pads; current: E7's wood).
A smoke checks the chain, not the policy (one rollout each).

### 2. Batch and eval tool (2026-11-22)

| Part | State | Where |
|---|---|---|
| Rollout record `sim_deploy.rollout/v2` (was `sim_deploy.episode/v1`; `gt.jsonl`, `rollout.json`, `protocol_sha` in `eval.yaml`) | built; every driver run writes it | `sim_deploy/eval/gt_recorder.py`, `evaltool/SCHEMA.md` |
| `evaltool score / report / frames` | built; reports from old folders (`trace.json`) tried: `sim_evals/_reports/_dev_old/` | `sim_deploy/evaltool/` |
| `evaltool import-real` + `labels.yaml` | built; the 9 trials of 2026-11-20 imported with draft labels (`sim_evals/_real/labels_2026-11-20.yaml`, `draft: true`): the user confirms them | `sim_evals/_real/`, `_reports/_dev_real/` |
| `eval batch` (plan, `--resume`, `--dry-run`, invalid-rollout retries, GPU lock, load gate, report at the end) | built; `w3.yaml` dry run OK (28 runs, 3 h 48 min to 7 h 18 min); `smoke.yaml` not run | `sim_deploy/eval/batch.py`, `eval/batches/` |
| Driver `--only-rollouts LIST` (retries with the same placements) | being added | `sim_deploy/eval/driver.py` |
| UI end to end on a real run | open: the last UI runs (2026-11-21 11:53-12:08) ended before the sim started ("no pi_stream packet") | `sim_evals/_logs/ui/` |
| UI runs as `launcher: ui` | open: the UI does not set `SIM_DEPLOY_LAUNCHER=ui`, so its evals record `cli` | `sim_deploy/ui/runner.py` |

How to run: sim_deploy/DIRECTIONS.md §3 (watch, score / report, batch, real trials).

## Analysis

(to be written by the W4 coordinator / integrated by the docs lead)

## Conclusion

(to be written by the W4 coordinator / integrated by the docs lead)

## Limitations / sim-vs-real caveats

- The settings come from W1's bench on this laptop; another GPU or a loaded laptop needs its own.
- The UI binds :8770 and drives the chain: it counts as the one chain while it runs.

## Open questions

1. Per-checkpoint settings after W1's timing bench: where the file lives so the sim and the real laptop read the
   same one. Until then bf16 / 8 / 12 for all.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| end-to-end evals | `sim_evals/<task>/<run_folder>/<date>_<time>_<tag>/` |
| UI logs | `sim_evals/_logs/ui/` |
| batches | `sim_evals/_batches/<BATCH_ID>/` |
| reports | `sim_evals/_reports/<NAME>/` |
| imported real trials | `sim_evals/_real/<task>/<run>/<session>/` |
