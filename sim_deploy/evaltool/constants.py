"""The scoring constants, defined once (SCHEMA.md "Scoring constants", user decisions 2026-11-21: all defaults;
success = lift only, user decision 2026-11-22; + the eraser clear of the table, user decision 2026-11-22, score v4).

Everything that decides a score lives here; score.json records these values (``constants`` block) so a score can be
read back against the rules it was made with. The eval driver's live judge (sim_deploy/eval/driver.py JUDGE) and the
protocol block hashed into eval.yaml protocol_sha (sim_deploy/eval/gt_recorder.py SCORING) take their values from
here, so the driver stops a rollout exactly when this rule counts it a success. Values not fixed by SCHEMA.md are
marked "evaltool decision" and explained in README.md ("Decisions").
"""

from __future__ import annotations

import re

import numpy as np

SCORE_SCHEMA = "sim_deploy.score/v4"        # v4: success also needs the eraser clear of the table; gpu_fault validity
RECORD_SCHEMA = "sim_deploy.rollout/v2"         # eval.yaml record_schema of folders that have gt.jsonl/rollout.json

# --- success (SCHEMA; lift only, grasp not required: user decision 2026-11-22) ----------------------------------
LIFT_M = 0.05              # eraser centre z >= settled z + 0.05 m ...
HOLD_S = 1.0               # ... held continuously >= 1.0 s
HOLD_TOL_S = 0.001         # evaltool decision: times are rounded to 1 ms in gt.jsonl / trace.json; the driver uses it too
# ... and in every row of that hold the eraser's lowest corner (its box at the row's pose) >= table top + CLEAR_M:
# lifted off the table, not tipped up on an edge or leaning (user decision 2026-11-22: "lifting off the ground")
CLEAR_M = 0.01
SUCCESS_NEEDS_GRASP = False
SUCCESS_RULE = ("eraser centre z >= settled z + lift_m and the eraser box's lowest corner >= table top + clear_m, "
                "both in every row of a run held continuously >= hold_s (rows 50 ms apart, hold_tol_s tolerance); "
                "grasp not required (user decisions 2026-11-22)")
# success within these times (t_success, s from the rollout start) is reported next to the timeout's rate
SUCCESS_WITHIN_S = (20.0, 30.0)
# lift_without_grasp (flag on a successful rollout, reported for a human check: one-finger carry or physics artifact):
# no row of the success hold window (the run of rows with lift >= LIFT_M that completed the hold, from its first row
# to t_success inclusive) satisfies the grasp rule below.
LIFT_WITHOUT_GRASP_RULE = ("success and no row from the start of the success hold run to t_success (inclusive) has "
                           "both sides touching the eraser")
TIMEOUT_S = 60.0           # per rollout (the driver's budget; recorded, not used to cut a record)
GT_RATE_HZ = 20.0

# grasp (a reported metric, not part of success): both sides (left and right pad or holder) touch the eraser in the
# same gt row
GRASP_GEOM = re.compile(r"^(pad|finger_holder)_(l|r)")      # same rule as sim_deploy/eval/gt_recorder.py PAD

# --- secondary metrics (evaltool decisions, README.md) -----------------------------------------------------------
LIFTED_M = 0.02            # "lifted" = grasp and >= 2 cm up (the driver's judge.lifted_m), reported only
GAP_CLOSE_DROP_M = 0.005   # close_error: the gap first drops 5 mm below open (SCHEMA)
FIRST_SECOND_S = 1.0       # first_second_progress window (SCHEMA)

# --- validity ----------------------------------------------------------------------------------------------------
# loop_late counts late 1 kHz cycles of the (sim) PC loop over the rollout (deployment/PLAN.md C2 field). Sim runs
# on a loaded laptop show 3-4 % late cycles as normal; > 10 % (100 per second) means the sim could not keep real
# time and the rollout is not a fair test of the policy. evaltool decision.
LOOP_LATE_MAX_PER_S = 100.0
# stop reasons that make a rollout invalid (not the policy's doing): the run was stopped by a person or broke
INVALID_STOP_REASONS = ("user", "error")
# real trials (source: real): the operator's stop ("user") is how a failing trial ends, so only "error" is invalid;
# loop_late is never a validity rule there (the real PC reports up to ~289 per rollout), only a timing metric.
INVALID_STOP_REASONS_REAL = ("error",)
# a fault text that is the sim falling behind wall time, not a gate rule: invalid as well. evaltool decision.
SIM_LAG_FAULT = re.compile(r"communication_constraints_violation|behind wall time")
# the arm was never commanded: chunks were sent but the PC accepted none of their waypoints for the whole rollout
# (a timeout). Seen 2026-11-22 after a GPU fault (inference ~3 s, every waypoint late and skipped). Slow inference as
# such is not a validity rule (user decision 2026-11-22): only a rollout in which nothing ever reached the arm.
# A fault (e.g. PolicyFatal on the first chunk) stays a policy failure. Sim only.
NO_ACCEPTED_STOP_REASONS = ("timeout",)

# a GPU fault (kernel "NVRM: Xid") seen by the batch after a run.sh call: the batch writes <eval>/gpu_fault.json with
# the first Xid time (xid_time) and, once a later batch start passes the GPU check, cleared (boot after the Xid, else
# the check time); every sim rollout of that eval whose [wall_start, wall_end] overlaps [xid_time, cleared) is invalid
# ("gpu_fault") and re-run (2026-11-22: after Xid 62 inference took ~3 s and no waypoint reached the arm). Sim only.
GPU_FAULT_FILE = "gpu_fault.json"

# --- fault classification (franka_control safety_gate.cpp texts) ------------------------------------------------
GATE_RULES = (
    ("box", re.compile(r"outside workspace box")),
    ("cone", re.compile(r"from approach axis")),
    ("lead", re.compile(r"max_lead_s")),
    ("speed", re.compile(r"v_max|w_max")),
    ("width", re.compile(r"gripper width")),
    ("nonfinite", re.compile(r"not finite")),
    ("reflex", re.compile(r"libfranka|reflex")),
)
CLAMP_LINE = re.compile(r"clamped into the workspace box")

# old eval outcomes (driver before the record sim_deploy.episode/v1, now sim_deploy.rollout/v2) -> SCHEMA stop_reason:
# the same map as the driver's sim_deploy/eval/gt_recorder.py STOP_REASON (C2 stopped / the eraser off the table / the
# manager ending it = fault); any other outcome (home / start pose / c refused, not run) = error
OUTCOME_TO_STOP_REASON = {
    "success": "success", "timeout": "timeout", "fault": "fault", "stopped": "fault", "fail": "fault",
    "interrupted": "user", "crashed": "error", "sim_error": "error",
}
OTHER_OUTCOME_STOP_REASON = "error"


def eraser_lowest_z(pose, box_m) -> float:
    """World z of the lowest corner of the eraser box (full extents ``box_m`` along its body x, y, z; model.build
    eraser_box) at ``pose`` [x, y, z, qw, qx, qy, qz]."""
    w, x, y, z = (float(v) for v in pose[3:7])
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    r2 = np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])   # row 3 of R (world z)
    return float(pose[2]) - float(np.abs(r2) @ (np.asarray(box_m, float) / 2))


def clear_of_table(pose, box_m, table_top_z: float) -> bool:
    """The success rule's second part: the eraser's lowest corner >= table top + CLEAR_M (the driver and
    evaltool score call this on the same gt row values)."""
    return eraser_lowest_z(pose, box_m) >= float(table_top_z) + CLEAR_M


def as_dict() -> dict:
    return {"lift_m": LIFT_M, "hold_s": HOLD_S, "hold_tol_s": HOLD_TOL_S, "clear_m": CLEAR_M,
            "success_within_s": list(SUCCESS_WITHIN_S), "gpu_fault_file": GPU_FAULT_FILE,
            "success_needs_grasp": SUCCESS_NEEDS_GRASP,
            "success_rule": SUCCESS_RULE, "lift_without_grasp_rule": LIFT_WITHOUT_GRASP_RULE, "timeout_s": TIMEOUT_S,
            "grasp_geom": GRASP_GEOM.pattern, "lifted_m": LIFTED_M, "gap_close_drop_m": GAP_CLOSE_DROP_M,
            "first_second_s": FIRST_SECOND_S, "loop_late_max_per_s": LOOP_LATE_MAX_PER_S,
            "invalid_stop_reasons": list(INVALID_STOP_REASONS),
            "invalid_stop_reasons_real": list(INVALID_STOP_REASONS_REAL), "sim_lag_fault": SIM_LAG_FAULT.pattern,
            "no_accepted_stop_reasons": list(NO_ACCEPTED_STOP_REASONS)}
