# Sim evaluator build: paused 2026-10-18 22:40 EDT. Resume here.

This is the one page to read before continuing. The approved plan is [PLAN.md](PLAN.md). Process rules, environments and carry-over notes are in [README.md](README.md). Cross-track contracts are in [INTERFACES.md](INTERFACES.md). Per-phase specs, worklogs and reviews are in `phases/`.

Branch: **`sim-eval`**, 9 local commits on top of `4be9982`, **nothing pushed**.

## How to resume (copy into a new session)

> Resume the Franka sim evaluator build. Read `controlSW/docs/sim_plan/STATUS.md` first, then README.md and INTERFACES.md
> in that folder. Continue from "Next actions" below, using the same executor + reviewer agent process (up to 4 agents,
> max 2 on Opus, stagger launches, resume agents via SendMessage on rate limits, commit each phase locally after its
> review PASSes, never push).

## Progress: 10 of 19 phases built, 9 committed

| Phase | What | State | Commit |
|---|---|---|---|
| A0 | Vendored mujoco_menagerie Panda + FR3 + Franka Hand (128 files, 95.8 MB, pinned 62ab00f2) | ✅ committed | d888546 |
| B1 | Station config system: `controlSW/configs/**` + `station.py` loader/validator | ✅ committed | d888546 |
| A1 | Model builder: arm + Hand + printed fingers + iPhone/rig housing + cradle + cameras + table/eraser | ✅ committed | 2efc96f |
| B2 | Station-aware launcher, constants lifted, bridge port 5555 → 5565 | ✅ committed | 0da4d34 |
| A2 | Kinematics (FK = Franka DH to 1e-15, IK ≥ 499/500) + frames | ✅ committed | 8522223 |
| B3 | Policy slot (bp_server/replay/fake/command) + `policy-contract` checker + `docs/adding_a_policy.md` | ✅ committed | bd0f05b |
| B4 | Wire protocol (v1 = iPhone app, v2 = rig/sim), Wi-Fi LinkEmulator, iPhone replay sender | ✅ committed | 374c1ec |
| B5 | WirelessCamera receiver + ClockMap (iPhone stamps ~3 ms p95; rig ping sync < 1 ms) | ✅ committed | a3d0d95 |
| B6 | Launcher wireless integration + profiles; **`sim-iphone-fake` runs end to end** | ✅ committed | d3b82cc |
| **A3** | Stage 1 offline demo-replay feasibility + design sweep | 🟡 **built and run, NOT reviewed, uncommitted** | — |
| **B7** | Rig stream sender + rig replay + `sim-rig-fake` (works end to end, stamp err p95 0.15 ms) | 🟡 **built, review interrupted, uncommitted** | — |
| A4 | SimCore + MuJoCoBackend (arm dynamics), bridge routing, PandaPyBackend reflex fix | spec ready (`phases/A4.md`) | — |
| B8 | Eval runner, criteria, Stage 1 wrapper, Stage 2a, Stage 2b | spec ready (`phases/B8.md`) | — |
| A5 | Franka Hand: hand_logic, non-blocking driver, hand_sim, hand_server | design only | — |
| A6 | Scene, sim_reset/sim_state, state PUB, ground truth | design only | — |
| A7 | Renderer (+0.5 px fix), camera-view check, render sender | design only | — |
| A8 | Stage 1b campaign (full-stack demo replay with physics) | design only | — |
| B9 | Stage 3 closed loop, judge, sweeps, summary | design only | — |
| B10 | Hardware-day tools (wireless latency QR, stream intrinsics) + docs (hardware_day, PURCHASING, ROADMAP, TODO) | design only | — |

Design detail for the "design only" phases is in [design_track_a.md](design_track_a.md) and [design_track_b.md](design_track_b.md). Write their specs from those, following the existing `phases/*.md` format.

## Uncommitted work in the tree (do not lose; review then commit)

**A3 (Track A):**
- `controlSW/franka_sim/franka_sim/stage1_replay.py`, `controlSW/franka_sim/tests/test_stage1.py`;
- `controlSW/franka_sim/franka_sim/__main__.py` (registers `stage1`);
- `controlSW/configs/run/sim/stage1.yaml`;
- `phases/A3.WORKLOG`.

**B7 (Track B):**
- `sensor_rig/sensor_rig/stream/{sender,cli}.py`, `sensor_rig/sensor_rig/cli.py` (one line), `sensor_rig/tests/test_stream_sender.py`;
- `controlSW/franka_deploy/franka_deploy/stream/rig_replay_sender.py`, `controlSW/franka_deploy/tests/test_rig_stream.py`;
- edits in `launch/launcher.py`, `configs/deploy.yaml`, `controlSW/configs/run/sim/rig_raw_replay.yaml`;
- `phases/B7.WORKLOG`.

Check with `git status`. Commit each phase **by explicit file list**, only after its review PASSes.

## Stage 1 result: hardware design answer (`results/stage1/`)

| Arm | Finger extension | Phone side | Demos feasible /45 | Worst phone↔arm clearance |
|---|---|---|---|---|
| Panda | stock | +y / −y | 20 / 4 | −22 / −59 mm (hits) |
| Panda | +5 cm | +y / −y | 24 / 2 | −10 / −16 mm |
| Panda | **+10 cm** | **−y** | **42** | **+37 mm** |
| FR3 | stock | +y / −y | 33 / 19 | −35 / −62 mm |
| FR3 | +5 cm | +y / −y | 33 / 16 | −12 mm |
| FR3 | **+10 cm** | **−y** | **43** | **+10 mm** |

- **Recommendation (sweep):** printed fingers about **10 cm longer** than stock, **phone on the −y side**. This holds for both arms.
  - Grasp-height sensitivity (h = 0.00): 42/45 for both arms, so the result is robust.
- **Open items from A3, to finish in its review:**
  - the demo behind the recurring 1.1° worst joint margin;
  - the clamp fraction of about 9 % at v_max 0.25 (the dataset has about 7 % of time above 0.25 m/s);
  - the 2–3 infeasible demos;
  - the +0.02 grasp-height sensitivity run was not finished.
- **Bug found by A3:** in `kinematics.py` (A2), the FR3 collision link groups are shifted by one link. The base is counted as the arm (always "hits" the table), and link 7 is never checked. Panda is unaffected.
  - A3 worked around it inside `stage1_replay.py` by building the groups from geom names.
  - **`kinematics.py` itself still has the bug.** Fix it with an FR3 test before A4 (see Next actions 2).
  - The earlier FR3 "0/45" numbers in `sweep_panda_and_buggy_fr3.log` are invalid.

## Decisions waiting on the user

1. **Confirm the tooling design:** finger extension (+10 cm recommended) and phone side (−y). This sets `tooling.finger_adapter.extension_m` and `tooling.tcp_x_along_hand` in both tooling YAMLs (still placeholders). A1/A3 showed that stock-length fingers put the phone into links 5/6.
2. **Panda or FR3:** both are about equally feasible with the recommended design. Confirm with Maral/Kian which arm the lab has.
3. **(Already answered 2026-10-18)** One executor + one reviewer per track; up to 4 agents; local commits; no push.

## Next actions (in order)

1. **Review A3** (Track A reviewer, Sonnet). Include the open items above. Then commit A3.
2. **Fix the FR3 link-group bug in `kinematics.py`** (resume or spawn a Track A executor; add an FR3 test that the tooling↔link6 and base↔table groups use the right geoms). Spot-review it, then commit.
3. **Review B7.** Restart the reviewer; the old one was stopped mid-run. **Must-check: camera intrinsics.**
   - The rig streams *virtual-iPhone* images, so `rig1_wifi.yaml` `streams.*.K` and the sender `hello` K must equal the iPhone deploy K at 960×720: main f 703.1714, c (476.3752, 361.3600); UW f 388.5231, c (480.6903, 359.7618).
   - Today the sender hello says 548.7/405 and the yaml says 703.2/421.5. Both look wrong for UW: FIX before commit.
   - Also re-run the flaky B4 test `test_stream_client.py::test_replay_sender_v1_pacing_size_and_K` (main 88 vs UW 89 frames under load) and harden it to ±1 frame.
4. **User decision on the tooling** (above), then set the tooling YAML values (with provenance).
5. **Launch A4** (Opus executor; spec `phases/A4.md`) and **B8** (Sonnet; spec `phases/B8.md`) in parallel.
6. Then write specs and run, in waves:
   - A5 + B9 (B9 needs A4–A7 for the closed loop with mujoco; it can start the judge/sweeps skeleton earlier);
   - A6 + B10;
   - A7;
   - A8.
   - Final W10: whole-system review, end-to-end verification (PLAN.md §11), `iphone_pipeline_check.py`, AGENTS.md update.

## Gotchas learned

- **Rate limits:** a session limit hit at about 21:45 EDT (resets at the hour). On 429, wait, then SendMessage the same agent to continue from its WORKLOG. A user-stopped agent cannot be resumed; spawn a fresh one pointing at the WORKLOG.
- **The suite lock (`$LOCK`) is often busy:** run lock-wrapped tests with `run_in_background` and wait for the notification. A foreground run hits the 120 s tool timeout.
- **Reviewers miss cross-arm issues:** both A2 bugs (tooling↔link6, FR3 link offset) passed review. Ask reviewers to test **both** arms and **both** camera sides explicitly.
- **Specs had wrong assumptions that executors caught** (the arkit_to_zup direction, the app streaming UW at 30 Hz not 10 Hz, the app's polling → websocket upgrade). Keep the rule: verify against code, then report deviations.
- **Disk:** about 3 GB free. Outputs go to the scratchpad. Copy anything worth keeping into `results/` (small files only).
- **Scratch env paths** (`/tmp/claude-1000/.../5ee0411f.../scratchpad/env_*`) are hardcoded in README.md, `station/host/pc_laptop.yaml` and the sim profiles in `deploy.yaml`. If the scratchpad is purged, recreate the envs (see AGENTS.md) and update those paths. B6 noted the launcher should take paths from the host group (to do in B8/B10).
