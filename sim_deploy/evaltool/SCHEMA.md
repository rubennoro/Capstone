# Rollout record `sim_deploy.rollout/v2` and scoring rules (agreed 2026-11-21)

One format whatever launched the run (UI, batch, CLI). The eval driver (`sim_deploy/eval/driver.py`) writes it; the
eval tool (`sim_deploy/evaltool/`) only reads it. The driver's live judge decides only when to stop a rollout; the
scores come from `evaltool score`.

A **rollout** is one closed-loop trial of a policy from one placement (sim), or one attempt of a real-arm trial
(`import-real`). "Episode" is kept for the training demonstrations (zarr datasets) and for policy_deploy's own names
(its `manager/episodes.jsonl`, `episode_id`, log lines "episode N started"), which sim_deploy reads unchanged.

## Versions

| version | what changed |
|---|---|
| `sim_deploy.episode/v1` (2026-11-21) | first record: `episodes.jsonl`, `episodes/NNN/episode.json`, `session_episode.json` |
| `sim_deploy.rollout/v2` (2026-11-22) | the same content, eval trials renamed to rollouts (user decision): folders `rollouts/NNN/`, files `rollouts.jsonl`, `rollout.json`, `session_rollout.json`; keys below. eval.yaml `schema` `sim_deploy.eval/v1` -> `/v2`, score.json `sim_deploy.score/v1` -> `/v2` |
| demo starts (2026-11-22, additive) | the driver's `--starts demos[:FILE]` (batch plan `starts: demos`, user decision for W3): rollout n's eraser at demo start n of `sim_deploy/eval/demo_starts/<dataset>.yaml` (a training demo's grasp carried to the start pose; `sim_deploy/eval/demo_starts.py`). Record schema unchanged (`sim_deploy.rollout/v2`); new: eval.yaml `scene.placements` (`random` / `replay` / `demos`) and `scene.demo_starts` (the file as frozen), rollout.json `placement.source` `demo` with `placement.demo`, `protocol.seed_rule.placement_source` `demos:<dataset>` with every rollout's demo and pose in `placement_rules` and another `seed_rule.key` text, so these evals never share a `protocol_sha` with random-placement ones. Random-placement evals hash exactly as before |
| start sets (2026-11-22, replaces the demo starts; additive) | user decision for W3: one fixed list of eraser starts for every checkpoint, `sim_deploy/eval/start_sets/w3.yaml` (`sim_deploy/eval/start_set.py`): rollouts 0-9 = the shared training demos 0-9 (the first 10 demos of all 9 datasets) at their replay placement from the start pose, rollouts 10-17 = the eraser poses fitted to the 8 real-arm trials of 2026-11-20 (the sim's 30 mm eraser, flat). Driver `--starts set[:FILE]` (plan `starts: set`); `--starts demos` and the per-dataset files are gone (only dry runs used them). Record schema unchanged (`sim_deploy.rollout/v2`); new: eval.yaml `scene.placements` `set` and `scene.start_set` (the file as frozen, this checkpoint's split and `held_out_rollouts`), rollout.json `placement.source` `demo` / `real_trial` with `placement.start` and `placement.held_out`, top-level `held_out` (true = the checkpoint's val demo); `protocol.seed_rule.placement_source` `set:<name>` with every rollout's start and pose (not the held-out demo: every checkpoint runs the same protocol) |
| score.json `sim_deploy.score/v3` (2026-11-22) | success = lift only, grasp no longer required (user decision 2026-11-22; §Scoring constants); new per-rollout flag `lift_without_grasp` and `summary.lift_without_grasp` {k, rollouts}; `constants` gains `success_needs_grasp`, `success_rule`, `lift_without_grasp_rule`; scorer_version 2. The record (`sim_deploy.rollout/v2`) is unchanged. The driver's live judge and eval.yaml `protocol.judge` use the same rule, so new evals get a new `protocol_sha`. Every folder under `sim_evals/` with a score.json was rescored the same day: no rollout changed outcome |
| gpu_fault window (2026-11-22, scorer_version 4, same schema) | `gpu_fault` = the rollout's `[wall_start, wall_end]` overlaps `[xid_time, cleared)` (was: started at/after `xid_time`, with no end): a rollout running at the Xid is now invalid too, and rollouts after the reboot + `--resume` stay valid. `gpu_fault.json` gains `cleared`, `cleared_by`, `earlier`; score.json `gpu_fault` copies them |
| score.json `sim_deploy.score/v4` (2026-11-22) | success also needs the eraser **clear of the table** in every row of the hold: its box's lowest corner (from `eraser_pose` and the eval's eraser size) >= table top + 0.01 m (user decision 2026-11-22, "lifting off the ground"); folders with only `trace.json` (no orientation) keep the lift alone and list `clear_of_table` in `missing`. New validity reason `gpu_fault` (a rollout that started at/after the first kernel Xid in `<eval>/gpu_fault.json`, written by the batch). New per rollout `start_source`, `held_out`, `start_id`, `start_group`, `timing.policy_ms_p90` / `cycle_ms_p90`; summary `success_within` {20s, 30s}, `by_start` {demo_trained, demo_held_out, real_trial}, `timing_pooled`; top-level `gpu_fault`; `constants` gains `clear_m`, `success_within_s`, `gpu_fault_file`; scorer_version 3. The driver's live judge and `protocol.judge` (`success_clear_m`) use the same rule, so new evals get a new `protocol_sha`. Every score.json under `sim_evals/` (94 folders, 103 rollouts) was rescored the same day: no rollout changed outcome or validity (88 of the rollouts are trace.json-only: no clearance check) |

Keys renamed in v2 (old -> new): eval.yaml `episodes` -> `rollouts`, `episode_seeds` -> `rollout_seeds`,
`only_episodes` -> `only_rollouts`, `home.episode_start_O_T_tcp` -> `home.rollout_start_O_T_tcp`, `protocol.seed_rule.key`
text "per episode n" -> "per rollout n" (protocol_sha recomputed); `rollouts.jsonl` lines `pd_episode_id` ->
`pd_rollout_id` (policy_deploy's episode_id of the rollout), `session_episode` -> `session_rollout`, paths
`episodes/NNN/...` -> `rollouts/NNN/...`; real `rollout.json` `session.episode_index` -> `session.rollout_index`
(`session.episode_id` stays: policy_deploy's id); score.json `invalid_episodes` -> `invalid_rollouts`, `episodes` ->
`rollouts`, `summary.episodes_recorded` / `episodes_rated` / `table_contact_episodes` -> `rollouts_*` /
`table_contact_rollouts`; `index.csv` column `episodes` -> `rollouts`; batch plan key `episodes` -> `rollouts`, manifest
`episodes` / `use_episodes` / `only_episodes` / `done_episodes` -> `rollouts` / `use_rollouts` / `only_rollouts` /
`done_rollouts`; CLI `--episodes` / `--only-episodes` -> `--rollouts` / `--only-rollouts`, evaltool frames
`--episode` -> `--rollout`. Folders written before were converted once by `python -m sim_deploy.evaltool
migrate-rollouts` (log in `sim_evals/_logs/migrate_rollouts_<time>.txt`); the readers accept only v2 names.

## Eval folder (existing layout, kept)

`<eval>/eval.yaml`, `rollouts.jsonl`, `rollouts/NNN/{trace.json, steps.jsonl, session_rollout.json, external.mp4, tap/}`,
`session/` (incl. `logs/manager.log`), `metrics.json`, `summary.html`, `index.csv`.

### Added to `eval.yaml`

| key | meaning |
|---|---|
| `record_schema` | `sim_deploy.rollout/v2` |
| `launcher` | `ui` / `batch` / `cli` (from env `SIM_DEPLOY_LAUNCHER`, default `cli`); `real` for imported arm trials |
| `batch_id` | from env `SIM_DEPLOY_BATCH_ID`, else null |
| `protocol` | the dict hashed below |
| `scene.placements` | `random` (rng([seed, n]) in `scene.randomization`), `replay` (`scene.replay_placement`) or `set` (`scene.start_set`); missing in evals before 2026-11-22 = random or replay (`demos` only in dry runs of that day) |
| `scene.start_set` | `--starts set` only (else null): `file` (repo-relative), `file_sha256`, `name`, `built` (model sha, look, start pose q and TCP, eraser size), `rule`, `demo_source` (dataset, path, size, sha256, demo names), `real_trial_source` (fit file + sha256, the real object's size), `checkpoint_split` (this checkpoint's split as the file records it: dataset, sample_type, val demos from the checkpoint and recomputed, `agree`, `held_out`), `held_out_rollouts`, `protocol_rules` (hashed), `starts` (all, as in the file, each with `held_out` for this checkpoint) |
| `protocol_sha` | sha256 (first 16 hex) of canonical JSON of `protocol`: seed rule (seed, placement source; start set: its name, the demo dataset's and the real-trial fit's sha256, q_start_rad and every rollout's source, id and eraser pose to 0.1 mm), timeout_s, precision, diffusion_steps, exec_h, look, model_sha256, start pose q_start_rad, judge rules (this file's §Scoring constants; `sim_deploy/eval/gt_recorder.py` SCORING, values from `evaltool/constants.py`), real limits (v_max, w_max, cone, box). Evals written before the lift-only rule (2026-11-22) keep their recorded hash, which differs from any new eval's |

### Added per rollout: `rollouts/NNN/gt.jsonl` (sim ground truth, 20 Hz, one JSON object per line)

| key | unit / type |
|---|---|
| `t` | s since the rollout started (the manager's "episode N started"; same zero as trace.json) |
| `eraser_pose` | [x, y, z, qw, qx, qy, qz] world (base frame), m |
| `tcp_pose` | [x, y, z, qw, qx, qy, qz] world, m |
| `gap_m` | finger gap, m |
| `contacts_eraser` | list of geom names (pads / holders, left and right) touching the eraser |
| `contacts_table` | list of geom names other than the eraser touching the table |
| `min_z_fingers_rig_m` | lowest z of any finger / rig / mount geom (AABB or vertex, say which) |
| `c2_mode` | the PC's C2 mode string |
| `reflex` | bool / reason string or null |

### Added per rollout: `rollouts/NNN/rollout.json`

`source` (`sim`/`real`), `launcher`, `batch_id`, `seed_key`, `placement` (xy, yaw, pose, `source` `random` / `replay` /
`demo` / `real_trial`; random: `sampled` = the draw; start set: `held_out` and `start` = `start_set`, `file`,
`rollout_slot`, `id`, `demo` (index, name) + `height_error_m`, `approach_tilt_deg` or `real_trial` (trial id, folder,
fitted x, y, yaw, fit residual), `eraser`, `start_frame` (ahead / left / yaw_rel at the start pose)), `held_out` (start
set: true = the checkpoint's own val demo, false = a demo it trained on or a real-trial pose; else null), `eraser_settled_pose`
(after settling, before the policy starts), `start_pose_q`, `outcome` (driver's live verdict), `stop_reason`
(`success` / `timeout` / `fault` / `user` / `error`), `clamps` (count of manager.log lines "clamped into the workspace
box" in this rollout's time window), `c2` (accepted / shortened / pulled_back / rejected / loop_late / modes / errors
for the window), `step_summary` (policy_ms p50/p95, margin min/p50, skipped steps by reason), `loadavg_start`,
`loadavg_end`, `loop_late`.

## Scoring constants (user decisions 2026-11-21: all defaults; success = lift only + clear of the table 2026-11-22)

Defined once in [constants.py](constants.py); the driver's live judge (`sim_deploy/eval/driver.py` JUDGE) and the
hashed `protocol.judge` (`sim_deploy/eval/gt_recorder.py` SCORING) import them.

| rule | value |
|---|---|
| grasp | both sides (left and right pad or holder) touch the eraser in the same gt row; a reported metric, not part of success (since 2026-11-22) |
| success | in every gt row of a run held continuously ≥ 1.0 s (rows 50 ms apart, 1 ms tolerance): eraser centre z ≥ settled z + 0.05 m AND the eraser box's lowest corner (pose + size) ≥ table top + 0.01 m (`CLEAR_M`, score v4); grasp not required (user decisions 2026-11-22; until then: grasp AND the lift, score v2). trace.json-only folders: the lift alone |
| success within | also reported: success with `t_success` ≤ 20 s and ≤ 30 s (`SUCCESS_WITHIN_S`) |
| lift_without_grasp | flag on a successful sim rollout: no gt row of the success hold window (the run of rows with lift ≥ 0.05 m that completed the 1 s, from its first row to `t_success` inclusive) has both sides touching the eraser. Listed in the report for a human check (one-finger carry or a physics artifact); still a success (user decision 2026-11-22) |
| timeout | 60 s per rollout |
| early stop | the rollout stops once success is reached: the driver tests the same rule on each gt row as written, so it stops on the row evaltool counts as `t_success` |
| invalid rollout | sim only: `loop_late` over the threshold (evaltool constant, documented), a timeout in which chunks were sent but the PC accepted 0 waypoints (`no_accepted_waypoints`: the arm was never commanded, e.g. after a GPU fault; slow inference by itself is not invalid, user decision 2026-11-22), or a rollout whose `[wall_start, wall_end]` overlaps a GPU fault window `[xid_time, cleared)` of `<eval>/gpu_fault.json` (`gpu_fault`; the batch writes `xid_time` = the first kernel `NVRM: Xid` it found after a run, and `cleared` when a later batch start / `--resume` passes the GPU check: the start of the boot after the Xid, else that time; both naive local time like `wall_start`; open-ended until `cleared` is written; a rollout without `wall_end` overlaps when it started before the window end; older closed windows sit in `earlier`) → excluded from rates and re-run by batch. Real trials (`source: real`) are never marked invalid by `loop_late` (the real PC reports up to ~289 per rollout); it is reported as a timing metric |
| N | W3: 18 per checkpoint, look `robot` only: the start set `w3` (10 shared demo starts, one of them the checkpoint's held-out demo, + 8 real-trial poses) |
| UI runs in W3 tables | only when `protocol_sha` matches the batch's |
| reports | `sim_evals/_reports/<name>/` (report.md, report.html, CSVs) |
| policy settings | bf16 / 8 diffusion steps / exec_h 12 for every checkpoint (until W1's timing bench is redone) |
| real arm | future trials: `manager.save_tap` on + `labels.yaml` per trial; the 9 existing trials imported with outcome (labels), clamps, timing only. Label rule for `outcome: success`: the eraser lifted clear of the table (grasp not required, matching the sim rule; user decision 2026-11-22); existing labels kept |
| sim eraser | 30 mm wide in every look |

## Metrics (`evaltool score` → `<eval>/score.json`)

grasp, success (and within 20 / 30 s), lift_without_grasp, t_grasp, t_success, max_lift, start split (demo trained / held out / real-trial pose), min_z (fingers/rig; TCP), table_contacts (rows + geoms), approach_error
(min TCP–grasp-point distance before the gap first closes), close_error (that distance when the gap first drops 5 mm
below open), first_second_progress (TCP displacement toward the eraser in the first 1 s), faults by gate rule
(box / cone / speed / lead), shortened / pulled_back / width clips / clamps, chunk timing, validity. Old folders without
`gt.jsonl` are scored from `trace.json` with the metrics it allows (missing ones null, listed in `score.json`).
