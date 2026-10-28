# Sim evaluator build: process and status

This folder is the working area for the Franka simulation evaluator: MuJoCo, iPhone and sensor rig, a configurable
station. The approved plan is copied in [PLAN.md](PLAN.md). Design detail per track is in [design_track_a.md](design_track_a.md)
(sim core) and [design_track_b.md](design_track_b.md) (config, wireless sensor path, policy slot, eval runner).
Cross-track contracts are in [INTERFACES.md](INTERFACES.md). Each phase has a spec in `phases/<id>.md`, an append-only
`phases/<id>.WORKLOG` and a `phases/<id>.REVIEW.md`. Current status: [STATUS.md](STATUS.md); open work: [TODO.md](TODO.md).

If PLAN.md, a design doc and a spec disagree, the more specific one wins: **spec > INTERFACES > PLAN > design docs**.

## Roles

Since A9 (2026-10-27) the user's rule is one executor per task, then **one** reviewer over all the executors' work and how
it fits together (no per-piece reviewer pairs). The roles below are otherwise unchanged.

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
S=/tmp/claude-1000/-home-ruben-Capstone/299e5dab-9afb-41ab-aa6c-ac52f8fb5552/scratchpad
RE=$S/env_realenv/bin/python   # real-env stack + mujoco 3.3.0 + flask-socketio + simple-websocket + websocket-client
BP=$S/env_bp/bin/python        # torch/behavior_prompting: inference, eval, zarr reading with imagecodecs
IP=$S/env_iphumi/bin/python    # sensor_rig tests, iPhUMI processing (regression pipeline check)
                               # needs conda-forge opencv 4.12.0 (pip opencv-headless changes solvePnP bits -> golden mismatch):
                               # conda create -p $S/env_iphumi -c conda-forge --override-channels python=3.10 numpy=2.2.6 opencv=4.12.0
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
   - Baseline: see [STATUS.md](STATUS.md).
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
| W4 | A3 Stage 1: built (committed in W5b) | B4: PASS, 374c1ec | |
| W5 | A4: see W5b | B5: PASS, a3d0d95 | |
| W6 | — | B6: PASS, d3b82cc | |
| W7 | — | B7 rig sender + `sim-rig-fake`: PASS, 806c9b6 | committed |
| W5b | A3 Stage 1 review fixes (860171f), A4 SimCore + MuJoCoBackend (067fab9) | B8 eval runner, Stage 1/2a/2b (f657715) | committed |
| W6c | A5 Franka Hand (d9aae67) | B9 Part 1: judge, Stage 3 on StubSim, sweeps (6eb7633) | committed |
| W7a | A6 scene runtime, ground truth (25cf996) | — | committed |
| W9a/9b | A8 Stage 1b campaign (0431ed4); A7 renderer, camera-check/gap (51371a0) | — | committed |
| W9c | — | B9 Part 2: Stage 3 on MuJoCo via BridgeSim (94f3575) | committed |
| W10a | — | B10 hardware-day tools (b2ff667) | committed |
| W10A | derived TCP, measured F_T_C_main honoured, loop stall stats (d960efe) | — | committed |
| W10B | sim TCP derivation in eval/deploy, judge offline fixes (9682000) | docs review fixes (45c190c) | on `sim-eval` via 3bfadbf + ba15402 (test fixes) |
| W10 verify | PLAN §11 steps 1-6 (iPhone + rig, demo_replay, bp_run001 Stage 3, smoke sweep, bp contract); step 7 (viewer) open | — | see STATUS.md "W10 verification" |
| W10 fixes | 1b on a Stage 1 subset + campaign port shift kept (cca6efb) | 2b/3 station ids + replay_run overrides (cd2dac6); `fitted` status (f112457) | cherry-picked onto `sim-eval` |
| W11 (A9) | finger kit: loader, model modes stock/box/kit, variants, Stage 1 length sweep, camera-check guard (f230a96, dad0ca3, 73c57cb) | finger_adapter keys + mode/variant validator (538b074, 54be744) | committed; review 1 FIX (A9.REVIEW.md) |
| W11f (A9) | review 1 fixes: kit masses in the finger inertials, tag ids, camera-check by id; real kit `iphumi_umi_soft_v1` (01adc08) | — | committed; review 2 FIX (A9.REVIEW2.md: real-kit P1 tags and P3 1b fail), findings in TODO.md, box stays default |

Test baselines and the resume point: see [STATUS.md](STATUS.md). Full history: `git log --oneline 4be9982..HEAD`.
