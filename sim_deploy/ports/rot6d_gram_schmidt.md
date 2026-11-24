# Handoff: orthonormalize the policy's rot6d before the C4 check (for rnoro/final-deployment)

For the agent that owns `rnoro/final-deployment` (`~/Capstone-final-deployment`). Written 2026-11-22 from
`rnoro/main-sim` (sim_deploy), where the change is applied to the sim's copy of `deployment/` and evaluated
closed-loop. Nothing in `rnoro/final-deployment` has been changed. Ask the user before committing or pushing there.

## 1. The problem

A diffusion checkpoint outputs each waypoint's orientation as rot6d (two 3-vectors, the first two ROWS of R). The
network never outputs exactly perpendicular unit vectors. Two places read that output:

| where | what it does with a skewed row |
|---|---|
| `policy_deploy/policy/base_policy.py` `Policy.make_chunk` → `ActionChunk` (C4 validator, `contracts/_rot6d.py rot6d_problem`, `contracts/c4_limits.yaml rot6d_dot_max: 0.1`, `rot6d_norm_tol: 0.1`) | **refuses** the chunk if any row's \|cos angle\| > 0.1 (~5.7° off perpendicular) or a length is off by > 0.1 → `policy_server._guard` raises `PolicyFatal` ("own reply broke C4") → the manager stops the episode (FATAL manager) |
| `policy_deploy/manager/pose_math.py matrix_from_rot6d` (used by `pose_converter`, `policy_step`) | **Gram-Schmidt**, the same math as behavior_prompting `common/pose_util.py rot6d_to_mat` (= UMI's): b1 = a1/\|a1\|, b2 = normalize(a2 − (b1·a2) b1), b3 = b1 × b2 |

So the manager is written to accept the model's skewed rows (as UMI/behavior_prompting do on their robots), but
the C4 check one step earlier kills the episode first. In the sim (same code, `deployment/` = f632576 + sim
hooks) this ended rollouts of PICK_TEST1 run_006, run_012 (all 20 rollouts of an earlier run) and run_020 (both
smoke rollouts, \|cos\| 0.19) after 6–50 s. The 2026-11-20 real trials did not hit it (no `PolicyFatal` in their
logs) but they were stopped by hand after 16–27 s; the real arm runs the same code and is exposed the same way.

## 2. The change (as applied in the sim copy)

File: `deployment/policy_deploy/policy_deploy/policy/base_policy.py` only. Reference implementation:
`~/Capstone-main-sim/deployment/policy_deploy/policy_deploy/policy/base_policy.py` (diff it against your file;
the header lines `copied from ... / changes:` are the sim's provenance convention, adapt to yours).

1. `orthonormalize_rot6d(rot6d) -> (fixed, worst_cos, worst_len)`: `fixed = _gram_schmidt_rows(raw)` in float64,
   returned as float32 C-contiguous in the input's shape. `_gram_schmidt_rows` is a local copy of
   `manager/pose_math.py` `matrix_from_rot6d`'s first two rows (same math, same `_EPS = 1e-12`; max diff 0 on 500 random
   rows): the policy process may not import `manager` (`tests/test_import_rules.py`), so do not import `pose_math`.
   Keep the two in step by hand.
2. `Policy.make_chunk` calls it first and builds the `ActionChunk` from the orthonormalized rows; it logs one INFO
   line for a chunk the raw check would have refused (raw worst \|cos\| and worst length error).
3. Nothing else changes: C4's `rot6d_problem`, `c4_limits.yaml`, the manager, the PC, contracts, wire format.

Properties (checked in the sim, 2026-11-22):
- **Robot motion unchanged for every chunk that passes today:** `matrix_from_rot6d(fixed) == matrix_from_rot6d(raw)`
  (max abs diff 3e-8, float32 rounding); Gram-Schmidt is idempotent, and the manager applies it again anyway.
- **Exact inputs unchanged** (hold_still, replay): max abs diff 6e-8.
- **Genuinely broken output is still refused** by C4 (`PolicyFatal`, as before): a zero first vector → "first
  vector has length 0"; parallel vectors → "second vector has length 0"; NaN/inf → "not finite" (the
  `ActionChunk` finiteness check; note `rot6d_problem` itself lets NaN through, which is pre-existing and covered
  by that check).
- Applies to every policy kind via `make_chunk` (diffusion, promptable diffusion, hold_still, replay).

## 3. What to do in rnoro/final-deployment

1. Port the change to `deployment/policy_deploy/policy_deploy/policy/base_policy.py` (copy `orthonormalize_rot6d`,
   the two imports, the logger, and the `make_chunk` body).
2. Tests (your branch has them; the sim branch does not add tests):
   - `tests/policy/test_make_chunk.py`: a skewed row (e.g. `[1,0,0, 0.2,0.98,0]`, \|cos\| ~ 0.2) now builds a valid
     chunk; its `matrix_from_rot6d` equals the raw row's; identity rows unchanged; zero / parallel / NaN rows still
     raise the C4 error.
   - `tests/contracts/test_c4.py::test_chunk_rot6d_norm_and_perpendicular` stays as is (the validator is unchanged:
     it still refuses a skewed chunk built directly, which is now only possible outside `make_chunk`).
3. Docs: `policy_deploy/STRUCTURE.md` (C4 section ~line 257–272: say the policy orthonormalizes before the check,
   and why), `DIRECTIONS.md` if it describes PolicyFatal causes.
4. Real-arm check before the next trials: run the usual dry runs; on the first trial, `grep "rot6d orthonormalized"`
   in the policy log shows how often the raw check would have fired.
5. Keep `rot6d_dot_max` / `rot6d_norm_tol` at 0.1: after the change every chunk from `make_chunk` has |cos| ~ 1e-7,
   so the check only fires on broken output.

## 4. Sim evaluation results (2026-11-22)

| | before (raw check) | after (Gram-Schmidt) |
|---|---|---|
| run_020, rollouts 0 + 10 (smoke) | 2/2 PolicyFatal (\|cos\| 0.19) at ~25–50 s | 0/4 PolicyFatal (rollouts 0, 1, 10, 15) |
| run_006 / run_012 | PolicyFatal in earlier W3 rollouts (run_012: all 20) | 0/8 PolicyFatal |
| chunks the raw check would have refused | — | 17 in 12 rollouts (run_020: 13, run_012: 4, run_006: 0); worst raw \|cos\| 0.30 |

Evals: `sim_evals/PICK_TEST1/run_0{06,12,20}*/2026-11-22_16*_gs_check/` (`grep "rot6d orthonormalized"` in
`session/*/logs/policy.log`). With the fix the rollouts run on until the next limit: every one then stopped at the
PC gate's 60+2° tilt cone (a policy behaviour, the same on the real arm; see
`sim-experiments/03_policy_pipeline/W3_closed_loop/README.md` "Pre-launch findings"). So the fix removes a crash
that is not the policy's fault; it does not by itself make the current checkpoints succeed.
