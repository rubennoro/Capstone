# Sim evaluator build: all phases built; W10 verification run except the viewer (2026-10-27). Resume here.

This is the one page to read before continuing. The approved plan is [PLAN.md](PLAN.md). Process rules, environments and carry-over notes are in [README.md](README.md) (its wave table has every commit). Cross-track contracts are in [INTERFACES.md](INTERFACES.md). Per-phase specs, worklogs and reviews are in `phases/`.

Branches, all local and **nothing pushed**:
- **`sim-eval`** has everything: all phases, W10B, the W10 verification and fixes, and the A9 finger kit (2026-10-27).
- `sim-eval-w10b` (worktree `.claude/worktrees/w10b`) was brought in commit by commit on 2026-10-27:
  - W10B and the docs wave were already in `3bfadbf`, except two test fixes, now taken.
  - Taken as-is: the verification results, `cd2dac6`, `cca6efb`, `f112457`, and these docs.
  - The worktree can now be removed.

## A9: plug-and-play finger kit (2026-10-27, on `sim-eval`)

- `tooling.finger_adapter.mode: stock | box | kit` (+ `kit_variant: full | finger_only`). Box stays the default. Spec: [phases/A9.md](phases/A9.md).
- Model and justification: [../../franka_sim/docs/finger_model.md](../../franka_sim/docs/finger_model.md).
- Final review: [phases/A9.REVIEW.md](phases/A9.REVIEW.md). Verdict FIX, with two open findings:
  - the kit masses don't reach the finger bodies;
  - camera-check pairs tags by image order, not by tag id.
- The real kit was built from the user's Onshape STLs (UMI soft finger + holders). It is still in the scratchpad (`$S/a9e/`, the finger-centred `alt/` version), not yet in `assets/finger_kits/`.

## How to resume (copy into a new session)

> Resume the Franka sim evaluator. Read `controlSW/docs/sim_plan/STATUS.md` first, then README.md and INTERFACES.md
> in that folder. Continue from "Next actions" below (executor + reviewer agents for code changes, local commits,
> never push).

## Progress: 19 of 19 phases built and committed

A0–A8 and B1–B10 are committed, plus the W10A/W10B fix waves. The only W10 item left is the end-to-end verification (PLAN.md §11).

## W10 verification (PLAN §11), run 2026-10-26/27 on `sim-eval-w10b`

| Step | Result |
|---|---|
| 1. Tests | franka_deploy 432 passed / 11 skipped. franka_sim 316 passed after fixing 3 stale UW-extrinsic test expectations (W10B item 3). env_bp subset 8 passed / 7 skipped (env_bp needed 31 missing deps reinstalled, see Gotchas). sensor_rig 422 passed + 1 load flake (`test_slow_link...[drop_newest]`, 3/3 pass alone). Regression: `check_protected` PASS, `test_iphone_deploy_unchanged` 11 passed, `iphone_pipeline_check` PASS exact (18 identical). |
| 2. Assets/config | `check-assets` OK (128 files). sim-iphone / sim-rig: `sim_tcp` OK 0.000 mm / 0.000 deg. real-iphone / real-rig FAIL as designed (Jetson paths + 10/9 critical placeholders). |
| 3. Stage 1 + camera | Both arms at +10 cm / −y: **43/45 feasible** (Panda clearance +23.8 mm, FR3 +9.6 mm). Stock fingers: 8/45 and 27/45, phone hits the arm. `eval --stage 1` on the shipped station FAILs (collisions 980): the YAMLs still carry stock / +y, see Decisions. camera-check (Panda +10/−y): tags p50 9.2 / 10.3 px. |
| 4. Stages 1b–2b, demo_replay, sim-iphone, Panda +10/−y (`--set` overrides) | **1: FAIL** joint margin 1.47° < 10° (worst demo incl. the 2 infeasible ones). **1b: FAIL** tracking 5/43 (A8 had 3/43: same finding, bridge 60° cone clamps), reflex 10 (A8: 8, eraser squeezed out at hand closing). **2a: FAIL** chunk margin 0° (the 2.2 % unreachable chunks of the infeasible demos); reachable 97.8 % PASS. **2b: INCOMPLETE** only for sample size (10 pairs < 50 with 1 demo). Every gate with data PASSes (live vs offline 3e-6 mm, stamp p95 18.6 ms). The 2b and camera-check runs need `data/raw/` (gitignored), so a worktree needs a `data` symlink to the main checkout. |
| 4. sim-rig, demo_replay, demos 0-4, Panda +10/−y | **1, 2a, 2b: PASS** (2a reachable 100 %, margin 42°; 2b 67 pairs, rig stamp p95 0.16 ms). **1b: INCOMPLETE**: every demo's start is refused, "FK 150-178 mm from home (max 100 mm)". The bridge clamps the start target to the 60° cone and the (placeholder) workspace box on every tick (clamp backlog = the FK gap); the rig housing puts all 5 start poses outside the cone (68-72°), the iPhone only 5/45 at 52-60 mm. With `--jobs 3` some runs also failed the camera preflight (27-28 of 30 packets, load). |
| 4. bp_run001, Stage 3, sim-iphone, 1 pose × 1 episode | Runs end to end on the laptop GPU (2.3 of 4 GB, 1 min 46 s). **FAIL**: tracking p95 8.1 mm (≤ 5), gripper lag 208 steps (empty hand closes fully, see W10A_hand.WORKLOG), obs motion 5.4 mm (≤ 3, placeholder). PASS: collisions 0, reflex 0, inference p95 0.41 s, receiver age p95 0.11 s. Success 0/1 (judge and sim_gt agree; informational with rendered images). |
| 5. Sweep `block_raise_smoke` (link ideal/wifi_good × policy fake/demo_replay, Stage 3) | Mechanics work: 4 runs, index, summary.html, 3 min 56 s. All FAIL as expected: fake only on gripper lag 2 steps; demo_replay on 1 reflex, tracking ~23 mm, obs motion 10-12 mm (open-loop replay in a closed loop). |
| 6. policy-contract | **bp_run001: CONTRACT PASS** (16/16; load 7.7 s, inference 0.25-0.29 s, K=16). fake and replay are covered by `test_policy_contract.py` (green). |
| 7. viewer | **Not run:** needs a display (the user). |
| W10 fixes found by the runs | (a) Stage 2b/3 children ignored station-group overrides, and Stage 2b `replay_run` also dropped sweep `extra_sets` and run-group overrides (B9 finding 3, `cd2dac6`). (b) Stage 1b crashed on a Stage 1 subset (`--stage all --episodes N`); it now maps demos by name (`cca6efb`). (c) An eval-made stage1.json's wiring sets undid the campaign's per-job port shift: every parallel job hit bridge port 5565 (`cca6efb`). (d) `fitted` provenance status for the iPhone UW extrinsic (W10B finding 5, `f112457`). Tests: 44 locked process tests, campaign/1b 22, eval 17, station 67, regression guard PASS. |

Read: the stack runs end to end at the recommended design, with the real policy too. The FAILs are findings, not regressions: 2 infeasible demos, the bridge cone/box clamps (1b tracking on the iPhone; every sim-rig start refused), reflexes at hand closing, and the real policy's tracking and observation motion in Stage 3. Summaries and logs are in [results/w10_verify_2026-10-26/](results/w10_verify_2026-10-26/). The full run dirs were in job scratch and are gone with the job.

## Stage 1 result: hardware design answer

| Arm | Finger extension | Phone side | Demos feasible /45 | Worst phone↔arm clearance |
|---|---|---|---|---|
| Panda | stock | −y | 8 | −59 mm (hits) |
| Panda | **+10 cm** | **−y** | **43** | **+24 mm** |
| FR3 | stock | −y | 27 | −62 mm (hits) |
| FR3 | +10 cm | −y | 43 | +10 mm |

The arm is the **Panda** (user decision 2026-10-27, on the advisor's recommendation): FR3 rows are for reference only.

(These are 2026-10-26 numbers with the W10 TCP code. The 2026-10-18 sweep gave 42/43 and a full ext × side grid: see [results/stage1/](results/stage1/).)

## Decisions waiting on the user

1. **Confirm the tooling design (D9):** printed fingers +10 cm, phone on −y. Then set `tooling.finger_adapter.extension_m: 0.10` and `tooling.tcp_x_along_hand: "-y"` in both tooling YAMLs (with provenance `decision`). Until then, default sim runs use the stock design and fail Stage 1.
2. **Bridge cone (60°) and workspace box (placeholder):** they drive the iPhone 1b tracking FAILs and refuse every sim-rig 1b start (demos tilt 68-72°). Options: widen the cone for the rig/eval, give a measured workspace box, or let the campaign start inside the cone. Related: a refused start is filed as `system_fault` (INCOMPLETE), not as a safety-clamp finding.
3. **What counts as done for 1b / 2a:** the `min`-over-all-demos gates fail on the 2 infeasible demos. Options: exclude the Stage-1-infeasible demos from those gates, or accept them as FAIL-by-finding.
4. Whether to push `sim-eval`.

## Next actions (in order)

1. A9:
   1. Fix the two review findings.
   2. Copy the real kit into `assets/finger_kits/iphumi_umi_soft_v1/`.
   3. Run the real-kit checks P0–P3 (finger_model.md section 7).
   4. Then remove the w10b worktree: `git worktree remove .claude/worktrees/w10b` (remove its `data` symlink first).
2. After D9: update the tooling YAMLs, rerun `eval --profile sim-iphone --stage 1 --out DIR` without `--set` (expect 43/45 + clearance PASS).
3. After the cone/box decision: rerun sim-rig `--stage 1b` (and the iPhone 1b) and a longer bp_run001 Stage 3 (`--poses 3 --episodes-per-pose 3`; 4 GB GPU is enough).
4. §11 step 7: a viewer run (user, display): copy deploy.yaml with `profiles.sim-iphone.bridge.override.backend.mujoco.viewer: true`.
5. Still open: the Stage 1 1.1–1.5° worst-margin demo; whether the real policy's Stage 3 tracking 8 mm / obs motion 5.4 mm is the policy or the sim (compare with a real-robot run on hardware day).

## Gotchas learned

- **Background sessions can't edit the main checkout:** work in a worktree. Gitignored inputs (`data/raw/`) need a symlink there.
- **env_bp lost 31 packages on or after 10-25** (typing_extensions, packaging, pygments, pydantic, flask, …; they used to come from the shared `~/.local`). They were reinstalled into env_bp with `--no-deps` at the `freeze3.txt` pins, and `pip check` is now clean. If it breaks again, check `~/.local`.
- **Orphaned launcher processes leave `/dev/shm/rmq_<user>_<pid>_*`** and make preflight `shm_clean` fail every process test. Check the pid is dead, then remove its segments.
- **Rate limits:** on 429, wait, then SendMessage the same agent to continue from its WORKLOG. A user-stopped agent cannot be resumed.
- **The suite lock (`$LOCK`) is often busy:** run lock-wrapped tests with `run_in_background`.
- **Timing tests flake under load** (sensor_rig slow-link, wireless stamp warnings). Rerun alone before calling it a regression.
- **Reviewers miss cross-config issues:** test both camera sides, both sensors (iPhone and rig), and a Stage 1 made by `eval` as well as by the `stage1` CLI (two of the W10 bugs only showed up that way).
- **Disk:** about 5 GB free. Outputs go to scratch; copy only small files into `results/`.
- **Scratch env paths** (`/tmp/claude-1000/.../299e5dab.../scratchpad/env_*`) are hardcoded in README.md, `station/host/pc_laptop.yaml` and the sim profiles in `deploy.yaml`. If the scratchpad is purged, recreate the envs and update those paths.
