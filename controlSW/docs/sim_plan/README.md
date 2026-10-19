# Sim evaluator build: process and status

This folder is the working area for the Franka simulation evaluator: MuJoCo, iPhone and sensor rig, a configurable
station. The approved plan is copied in [PLAN.md](PLAN.md). Design detail per track is in [design_track_a.md](design_track_a.md)
(sim core) and [design_track_b.md](design_track_b.md) (config, wireless sensor path, policy slot, eval runner).
Cross-track contracts are in [INTERFACES.md](INTERFACES.md). Each phase has a spec in `phases/<id>.md`, an append-only
`phases/<id>.WORKLOG` and a `phases/<id>.REVIEW.md`.

If PLAN.md, a design doc and a spec disagree, the more specific one wins: **spec > INTERFACES > PLAN > design docs**.

## Roles

- **Executor** (one per track): implements exactly one phase spec at a time.
  - Edits only files in its track's ownership list (INTERFACES.md §1).
  - Runs only the phase's tests. Full suites go through the suite lock.
  - Appends one line per step to `<id>.WORKLOG`.
  - Never commits.
  - Final message ≤ 40 lines: files changed, test commands with pass/fail counts, deviations from the spec with reasons, open questions.
- **Reviewer** (one per track): never edits code. Steps:
  1. Read `git diff <wave_base> --stat`, the changed files, and the direct callers of changed symbols (grep).
  2. Re-run the spec's test commands and the regression guard.
  3. Tick the acceptance checklist with evidence (command output or file:line).
  4. Check file ownership and run the station YAML validator once it exists.
  5. Write `<id>.REVIEW.md` (≤ 60 lines) with verdict PASS / FIX / BLOCK and numbered findings with file:line.
- **Orchestrator** (main session):
  - Writes the specs.
  - Runs the wave gate.
  - Commits locally on branch `sim-eval` as `sim-eval: wave N`. Never pushes.

## Environments (scratch, not in the repo)

```
S=/tmp/claude-1000/-home-ruben-Capstone/5ee0411f-3d2e-4964-b2ee-0d8c8e48aa78/scratchpad
RE=$S/env_realenv/bin/python   # real-env stack + mujoco 3.3.0 + flask-socketio + simple-websocket + websocket-client
BP=$S/env_bp/bin/python        # torch/behavior_prompting: inference, eval, zarr reading with imagecodecs
IP=$S/env_iphumi/bin/python    # sensor_rig tests, iPhUMI processing (regression pipeline check)
LOCK=$S/suite.lock             # flock target for full suites and e2e/process tests
ZARR=$S/deploy_work/prompt_zarr/firstreal-single.zarr.zip   # firstreal-single dataset (45 demos)
```

- franka_deploy tests: `cd controlSW/franka_deploy && env -u PYTHONPATH PYTHONPATH=. $RE -m pytest -q tests/...`
- Full suite: wrap it as `flock $LOCK <command>`.
- Headless rendering: `MUJOCO_GL=egl`. Call `renderer.close()` explicitly, to avoid a harmless EGL error at interpreter exit.
- New process tests take their ports from `FD_TEST_PORT_OFFSET` (Track A: 1000, Track B: 2000).
- Write outputs outside the repo (scratch or work_root). The disk is about 99% full.

## Wave gate (orchestrator, before each local commit)

1. Both REVIEW.md files say PASS.
2. The phase tests pass.
3. The full franka_deploy suite passes in RE, plus the BP subset if touched.
   - Baseline 2026-10-18: RE 145 passed, 8 skipped.
   - BP subset: `tests/test_light_workspace.py tests/test_eval_demos.py tests/test_eval_metrics.py tests/test_server_ckpt.py`.
   - BP cannot run the files that import real_env (that is pre-existing, not a regression).
4. The sensor_rig suite passes if touched.
5. `scripts/regression/check_protected.sh` and `env -u PYTHONPATH $RE -m pytest scripts/regression/test_iphone_deploy_unchanged.py` pass.
6. `iphone_pipeline_check.py` passes, run at W6 and W10.
7. At least 1.5 GB of disk is free.

## Rate-limit rules

- At most 4 agents at once, and at most 2 on Opus at once.
- Stagger launches by 60–90 s.
- On 429/overload: back off 60 → 120 → 240 → 480 s (cap 10 min), then resume the **same** agent with SendMessage: "continue from your WORKLOG".
- Never put two agents on one phase. After 2 rate-limit events in a wave, drop to 2 concurrent agents for the rest of that wave.

## Commit rule

Commit each phase as soon as its review PASSes, by explicit file list, before the next phase of the same track edits
the same files.

## Carry-over notes

- A7: rendered images are offset about 0.5 px (MuJoCo principal point is measured from W/2; OpenCV pixel centres are
  integers). Use cx+0.5, cy+0.5 in the renderer (A1 review finding 1).
- B6: align the `station-mock` FakeBackend home and F_T_TCP with the station (B2 open question); wire `robot.home_tcp`. Set `run/policy/demo_replay.yaml` python to `${host.python.realenv}` (replay_server's historical env).
- B6/B8: `iphone_replay_sender` ready-file `start_now_ns` is the slot-0 time but the first tick is slot 2 (B4 review); account for it in Stage 2b replay_gt alignment. App default fps is 60; our sensor yaml requires 30.
- B8/B9: treat launcher `sender_died` as an episode abort; sim profiles hardcode scratch paths (move to host group).
- A6: launcher sim_reset request timeout is 0.5 s; make the reset return fast or raise the timeout (B-owned launcher).

## Status

| Wave | Track A | Track B | Gate |
|---|---|---|---|
| W0 | setup | setup | done (branch, env, docs) |
| W1 | A0 assets: PASS | B1 station: PASS | committed d888546 (RE suite 190 passed / 8 skipped; regression guard PASS) |
| W2 | A1 model: PASS | B2 constants + launcher: PASS | committed 0da4d34 (B2), 2efc96f (A1); RE suite 215 passed |
| W3 | A2 kinematics: PASS (+link6 fix) | B3 policy slot: PASS after FIX | committed 8522223 (A2), see log (B3) |
| W4 | A3 Stage 1: built, NOT reviewed (uncommitted) | B4: PASS, 374c1ec | |
| W5 | A4 (waits on user tooling decision) | B5: PASS, a3d0d95 | |
| W6 | — | B6: PASS, d3b82cc | |
| W7 | — | B7: built, review interrupted (uncommitted) | |

**PAUSED 2026-10-18 22:40 EDT: see [STATUS.md](STATUS.md) for the resume point.**
