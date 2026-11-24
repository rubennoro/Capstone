# sim_deploy/evaltool: score recorded rollouts, separately from running them

The eval driver (`sim_deploy/eval/`) runs rollouts and records them; its live judge only decides when to stop. The
scores come from here. Record format and the agreed rules: [SCHEMA.md](SCHEMA.md); the rule values are defined once in
[constants.py](constants.py) and copied into every `score.json` (`constants`).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
PY=~/envs/sim_deploy/bin/python
$PY -m sim_deploy.evaltool score <eval_dir> [<eval_dir> ...]          # -> <eval>/score.json
$PY -m sim_deploy.evaltool report --select '<regex on tag>' [--batch ID] [--protocol SHA] \
    [--group checkpoint,kind,look] --out sim_evals/_reports/<name>      # [--root sim_evals]
$PY -m sim_deploy.evaltool frames <eval_dir> [--rollout N]           # -> <eval>/rollouts/NNN/frames.jpg
$PY -m sim_deploy.evaltool prune-index [--dry-run]                   # drop index.csv rows of deleted eval folders
$PY -m sim_deploy.evaltool import-real ...                           # import_real.py (its own --help)
$PY -m sim_deploy.evaltool migrate-rollouts [--dry-run]              # one-off, done 2026-11-22 (SCHEMA.md "Versions")
```

## Outputs

| command | writes | nothing else |
|---|---|---|
| `score` | `<eval>/score.json`: `constants`, `eval` (tag, task, run, look, batch_id, protocol_sha, ...), `summary`, `invalid_rollouts`, `rollouts` (one object per rollout, metrics below, `missing` = metrics the record could not give) | no other file in the eval folder is touched |
| `report` | `<out>/report.md`, `report.html` (one file, inline CSS, no external assets), `rollouts.csv` (one row per rollout), `checkpoints.csv` (one row per `--group` key), `kind_x_look.csv`; a "Lift without grasp (check by hand)" section (count per eval + eval / rollout table) and a `lift_without_grasp` column in the rollout and group tables | scores in memory; eval folders are not written |
| `frames` | `<eval>/rollouts/NNN/frames.jpg`: start / closest approach / gripper close / end from `external.mp4`; a second row with the tap's main-camera image nearest each time when the rollout has tap images | |
| `prune-index` | `sim_evals/index.csv` without the rows whose `eval_dir` is gone; the old file first to `sim_evals/_logs/index_before_prune_<time>.csv` (2026-11-22: 5 rows of the stopped W3 batch dropped) | |
| `migrate-rollouts` | renames `episodes` -> `rollouts` in every eval folder under `sim_evals/`, `index.csv` (backup in `_logs/`), `_batches/`; log `sim_evals/_logs/migrate_rollouts_<time>.txt` | `session/`, `_reports/`, run logs |

`report` selection: every folder under `--root` with an `eval.yaml` (not `_dry_run/`, `_reports/`, not dry runs);
`--select` = `re.search` on `eval.yaml` `tag`; `--batch` = `eval.yaml` `batch_id`; `--protocol` = prefix of
`protocol_sha`. Group keys: `checkpoint` (`<task>/<run_folder>`), `kind`, `look` (`none` = folder older than the
looks), `origin` (`sim` / `real`), `tag`, `batch`, `protocol`. In `checkpoints.csv`, `valid` = valid rollouts (labelled or not),
`rated` = valid and labelled (the rate's denominator), `unlabelled` = real trials without a confirmed label. The report also has BP vs plain pooled, kind x look,
robot vs train paired on `seed_key`, "Success within time and by start" (success with `t_success` <= 20 s / 30 s next
to the 60 s rate; for start sets: the demo starts the checkpoint trained on, its held-out demo, the real-trial poses;
`checkpoints.csv` `success_20s*`, `success_30s*`, `demo_trained_*`, `demo_held_out_*`, `real_trial_*`) and "Timing vs
the real arm" (sent chunks pooled per checkpoint: sim policy / cycle ms p50 / p90, `sim_*` in `checkpoints.csv`; real
from the steps.jsonl of the trials under `sim_evals/_real`, else the documented p50s in `report.py`
`REAL_TIMING_FALLBACK`: run_006 119 / 132 ms, the plain runs ~97 / ~110 ms).

## Metrics (per rollout, `score.json` `rollouts[]`)

Source: `rollouts/NNN/gt.jsonl` + `rollout.json` (sim_deploy.rollout/v2) when present; else `trace.json` (20 Hz: t,
lift, grasped, finger gap, and TCP pose since P3) + the driver's `rollouts.jsonl` line; real trials (`source: real`):
`rollout.json` only. `source` says which.

| metric | definition |
|---|---|
| `grasp`, `t_grasp` | a gt row where both sides (`pad_l`/`finger_holder_l` and `pad_r`/`finger_holder_r`) touch the eraser; first such time. Real: the label's `grasped` |
| `success`, `t_success` | in every row of a run held continuously >= 1.0 s (rows 50 ms apart, 1 ms tolerance): eraser centre z >= settled z + 0.05 m and the eraser box's lowest corner (row pose + the eval's eraser size, `scene.station_scene`) >= table top + 0.01 m (score v4); grasp not required (user decisions 2026-11-22); `t_success` = the row where the 1 s is complete (the driver stops on that row). trace.json-only folders: the lift alone (`clear_of_table` in `missing`). Real: label `outcome == success` (the eraser lifted clear of the table); `outcome: null` = unlabelled |
| `lift_without_grasp` | success, and no row of the success hold window (from the first row of the lifted run that completed the 1 s to `t_success`) is a grasp row. Still a success; flagged for a human check (one-finger carry or physics artifact). null when not a success, real, or no grasp column |
| `lifted` | a grasp row with lift >= 2 cm (reported only) |
| `max_lift_m` | max of eraser z - settled z |
| `min_z.tcp_m`, `min_z.fingers_rig_m` | lowest TCP z; lowest finger / rig / mount geom z (gt.jsonl only) |
| `table_contacts` | gt: rows with anything but the eraser on the table + geom names; old: the driver's poll count / geoms |
| `approach_error_m`, `t_approach` | min TCP-to-eraser-centre distance before the gap first closes (all rows if it never closes); eraser = its pose in that gt row (old: the settled pose) |
| `close_error_m`, `t_close` | that distance at the first row with gap <= open gap - 5 mm (open gap = the first non-null row's) |
| `first_second_progress_m` | TCP displacement over the first 1 s projected on the unit vector TCP(0) -> eraser(0); `first_second_dist_drop_m` = the drop in that distance |
| `faults` | per gate rule from the fault / C2 error texts: `box`, `cone`, `lead`, `speed`, `width`, `nonfinite`, `reflex`, `other` (real imports: `c2.fault_rule`) |
| `gate` | accepted / rejected / shortened (speed rule) / pulled_back / width_clips over the rollout |
| `clamps` | `rollout.json` `clamps`; old: manager.log lines "clamped into the workspace box" between the manager's "episode N started" and "episode N:" |
| `timing` | from `steps.jsonl`: chunks, sent, skipped by reason, policy_ms p50/p90/p95, cycle_ms p50/p90/p95, image age p50, margin min/p50 |
| `start_source`, `held_out`, `start_id`, `start_group` | the placement's source (random / replay / demo / real_trial) and, for a start set, whether it is the checkpoint's held-out demo; `start_group` = demo_trained / demo_held_out / real_trial |
| `validity` | `valid`, `reasons`, `loop_late`, `loop_late_per_s` (below) |

`summary`: success / grasp / lifted with Wilson 95 % CI, `lift_without_grasp` {k, rollouts} over rated rollouts (`sim_deploy.eval.report.wilson`) over rated rollouts (valid
and labelled), counts of invalid (by reason) and unlabelled, stop reasons, faults by rule, medians of the metrics over
valid rollouts, gate / clamp totals, timing; `success_within` {20s, 30s}, `by_start` {demo_trained, demo_held_out,
real_trial} (start sets only), `timing_pooled` (policy / cycle ms p50 / p90 over every sent chunk of the valid rollouts).

`checkpoints.csv` also has `w1_first_chunk_toward_grasp_cm` / `w1_demo_first_chunk_toward_grasp_cm`: the medians of
W1 step 3's first-chunk measure (`sim_evals/_pipeline/W1/openloop/summary.json` `runs[<task>/<run>].approach_row6`:
k = 15 TCP displacement toward the demo's grasp point, row 6 of each sampled real training episode, bf16/8 policy and the
recorded demo action), to set next to `first_second_progress` (sim, toward the eraser, 1 s; note cm vs m).

## Validity

An invalid rollout is excluded from every rate and counted (`invalid`, `invalid_reasons`); a batch re-runs it.

- sim: `stop_reason` `user` or `error` (old outcomes mapped as the driver's `gt_recorder.STOP_REASON`: `interrupted`
  -> user; `stopped` (C2), `fail` -> fault; `crashed`, `sim_error` and any other (start / home refused, not run) ->
  error); loop_late > 100 per second of rollout; a fault
  whose text says the sim fell behind wall time (`communication_constraints_violation`); no ground-truth rows; a
  timeout in which chunks were sent but 0 waypoints accepted (`no_accepted_waypoints`); a rollout that started at or
  after the kernel Xid recorded in `<eval>/gpu_fault.json` by the batch (`gpu_fault`).
- real: only `stop_reason` `error`. The operator's stop is how a failing real trial ends, and loop_late (up to ~289 per
  real rollout) is a timing metric there. `outcome: null` (labels not confirmed) = unlabelled: counted, not rated,
  not a failure; its `outcome_draft` is shown.

## Batches (`--batch ID`)

With `sim_evals/_batches/<ID>/manifest.jsonl` (sim_deploy/eval/batch.py) the report counts, per run key chain
(`retry_of`), each rollout index once: from the highest retry level whose `use_rollouts` holds it and whose eval
recorded it, else from the original run. Other rollouts of a retry eval (extras) and the replaced originals are left
out; the selection line says how many. Without a manifest every rollout of the `batch_id` folders counts.

## Decisions (where SCHEMA.md leaves room)

1. loop_late threshold: 100 late 1 kHz cycles per second (10 %); sim runs show 3-4 % as normal, 12 % in overloaded runs.
2. A sim-lag fault (`communication_constraints_violation`) is invalid like loop_late; the stop reasons `user` and
   `error` make a sim rollout invalid; for real trials only `error` does.
3. Success is the lift with the eraser clear of the table (user decisions 2026-11-22, score v4; v3: the lift alone;
   v2 also needed a grasp row in the lifted run); hold tolerance 1 ms; clearance from the box corners at each row's
   pose (`constants.clear_of_table`, shared with the driver). The driver tests the same rule (constants imported from here) on each gt row as written, so
   its early stop and `t_success` agree with the score. `lift_without_grasp` window = the hold run up to `t_success`.
4. "Grasp point" = the eraser centre; "open" gap = the first non-null gap.
5. Old outcomes map to `stop_reason` exactly as the driver's `gt_recorder.STOP_REASON`, so both formats agree.
6. Robot vs train pairing: valid labelled rollouts with equal `seed_key`; if a (checkpoint, look, seed_key) repeats,
   the newest eval's rollout; exact McNemar p on the discordant pairs.
7. Kind: `sim_deploy.ui.catalog` on the run folder (`policy.raw`, or `~/Capstone/data/<task>/policy/<run>` for real
   imports); else `eval.yaml` `policy.kind`. `hold_still` and `replay` keep their own kind.
8. Tap image times in `frames.jpg`: chunk anchor time minus the rollout's first chunk anchor (approximate).
