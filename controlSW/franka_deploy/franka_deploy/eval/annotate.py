"""Success annotation for block_raise rollouts.

SUCCESS RULE (block_raise): the eraser is lifted >= 3 cm above its resting height and held for >= 2 s without
slipping (the fingers stay within 2 mm of the grasp width while lifted), then the episode may end. The live operator
label (y/n/s) is kept as is; this file adds a post-hoc annotation from the logged video and logs.

Failure categories: missed_grasp, slip, no_approach, collision_box_contact, safety_stop, timeout, system_fault.
system_fault (process died, Ctrl+C, controller/bridge fault, inference error) is excluded from policy success rates
but reported.

    python -m franka_deploy annotate <episode_dir> --annotator NAME [--success yes|no] [--grasp-t S] [--lift-t S]
        [--category CAT] [--note TEXT] [--auto]
Without flags on a terminal: prints the episode timeline + the automatic suggestion (from the trace: phases of the
measured width + TCP height) and the video paths, then asks. --auto writes the suggestion only (source "auto").
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

CATEGORIES = ("missed_grasp", "slip", "no_approach", "collision_box_contact", "safety_stop", "timeout", "system_fault", "none")
LIFT_MIN_M, HOLD_MIN_S, SLIP_TOL_M = 0.03, 2.0, 0.002


def auto_suggest(ep_dir: str, up_axis=(0.0, 0.0, 1.0)) -> dict:
    from franka_deploy.eval.metrics import phases

    lab = json.load(open(os.path.join(ep_dir, "label.json")))
    label = str(lab.get("label"))
    s = json.load(open(os.path.join(ep_dir, "summary.json")))
    sug = {"source": "auto", "live_label": label}
    if label.startswith("aborted:") or label.startswith("exception:"):
        sug.update(success=None, failure_category="system_fault", detail=label)
        return sug
    tr_p = os.path.join(ep_dir, "controller_trace.npz")
    if not os.path.exists(tr_p):
        sug.update(success=None, failure_category="system_fault", detail="no controller trace")
        return sug
    tr = np.load(tr_p)
    t, fk, w = tr["t"], tr["fk"], tr["width_measured"]
    h = (fk[:, :3] - fk[0, :3]) @ np.asarray(up_axis, float)
    ph = phases(w, h, t)
    if ph is None:
        cat = "no_approach" if np.ptp(fk[:, :3], axis=0).max() < 0.02 else "missed_grasp"
        sug.update(success=False, failure_category=cat, detail="gripper never closed")
        return sug
    held_mask = (t - t[0] >= ph["t_grasp"]) & (t - t[0] <= ph["t_release"])
    lifted = h - h[np.searchsorted(t - t[0], ph["t_grasp"])] >= LIFT_MIN_M
    hold_s = float(np.sum(np.diff(t, prepend=t[0])[held_mask & lifted]))
    slip = bool(np.any(w[held_mask & lifted] < ph["grasp_width_m"] - SLIP_TOL_M)) if np.any(held_mask & lifted) else False
    ok = ph["lift_height_m"] >= LIFT_MIN_M and hold_s >= HOLD_MIN_S and not slip
    cat = "none" if ok else ("slip" if slip else ("collision_box_contact" if s.get("workspace_contacts") else
                                                  ("timeout" if ph["lift_height_m"] < LIFT_MIN_M else "slip")))
    sug.update(success=bool(ok), grasp_t=ph["t_grasp"], lift_t=ph["t_lift_peak"], lift_height_m=ph["lift_height_m"],
               hold_s=hold_s, failure_category=cat)
    return sug


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--success", choices=["yes", "no", "unknown"], default=None)
    ap.add_argument("--grasp-t", type=float, default=None)
    ap.add_argument("--lift-t", type=float, default=None)
    ap.add_argument("--category", choices=CATEGORIES, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--annotator", required=True, help="who annotates (required; never defaulted from $USER)")
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args(argv)
    d = a.episode_dir
    sug = auto_suggest(d)
    ann = dict(sug)
    if not a.auto:
        manual = {}
        if a.success is not None:
            manual["success"] = {"yes": True, "no": False, "unknown": None}[a.success]
        for k, v in (("grasp_t", a.grasp_t), ("lift_t", a.lift_t), ("failure_category", a.category)):
            if v is not None:
                manual[k] = v
        if not manual and sys.stdin.isatty():
            print(json.dumps(sug, indent=1))
            print("videos:", glob.glob(os.path.join(d, "*.zarr", "*.mp4")))
            r = input("success? [y/n/u, enter = suggestion] ").strip().lower()
            if r:
                manual["success"] = {"y": True, "n": False}.get(r)
            c = input(f"failure category {CATEGORIES} [enter = suggestion] ").strip()
            if c:
                manual["failure_category"] = c
        if manual:
            ann.update(manual, source="manual", annotator=a.annotator, auto_suggestion=sug)
    ann["note"] = a.note
    with open(os.path.join(d, "annotation.json"), "w") as f:
        json.dump(ann, f, indent=1, default=str)
    print(json.dumps(ann, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
