"""Post-mortem completion of an episode that ended abnormally (task killed, controller killed, Ctrl+C).

Run by the launcher after teardown (and safe to re-run): merges the controller's 1 s trace parts, writes
summary.json / label.json (label "aborted:<reason>" unless the task already finalized it), camera_stats.txt and the
bridge.jsonl slice from the session logs. events.jsonl is already written incrementally by the task.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np


def _events(d):
    p = os.path.join(d, "events.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for ln in open(p):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def salvage_episode(d: str, reason: str, bridge_log: str | None = None, camera_log: str | None = None) -> dict:
    from franka_deploy.controllers.safe_controller import merge_trace_parts

    done = {}
    tr, parts = os.path.join(d, "controller_trace.npz"), os.path.join(d, "controller_trace_parts")
    if not os.path.exists(tr) and os.path.isdir(parts):
        done["trace_merged"] = merge_trace_parts(parts, tr)
    lab_p = os.path.join(d, "label.json")
    lab = json.load(open(lab_p)) if os.path.exists(lab_p) else {"label": "aborted:unfinalized"}
    finalized = os.path.exists(os.path.join(d, "summary.json")) and not str(lab.get("label", "")).startswith("aborted:unfinalized")
    evs = _events(d)
    t0 = next((e["t"] for e in evs if e["kind"] == "episode_start"), evs[0]["t"] if evs else None)
    t1 = evs[-1]["t"] if evs else t0
    if os.path.exists(tr):
        z = np.load(tr)
        if "t" in z.files and len(z["t"]):
            t0 = min(t0, float(z["t"][0])) if t0 is not None else float(z["t"][0])
            t1 = max(t1 or 0, float(z["t"][-1]))
    if not finalized:
        label = f"aborted:{reason}"
        json.dump({"label": label, "is_successful": None, "notes": lab.get("notes", []), "salvaged": True}, open(lab_p, "w"))
        taps = glob.glob(os.path.join(d, "tap", "call_*.npz"))
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
        dur = float((t1 or 0) - (t0 or 0))
        json.dump({"episode_idx": int(os.path.basename(d).split("_")[-1]), "duration_s": dur, "label": label,
                   "policy_calls": len(taps), "salvaged": True, "abort_reason": reason, "size_mb": size / 1e6,
                   "log_rate_mb_per_min": size / 1e6 / max(dur / 60, 1e-6),
                   "workspace_contacts": sum(1 for e in evs if e["kind"] == "workspace_contact")},
                  open(os.path.join(d, "summary.json"), "w"), indent=1)
        done["summary_written"] = True
        if bridge_log and os.path.exists(bridge_log) and not os.path.exists(os.path.join(d, "bridge.jsonl")) and t0 is not None:
            with open(bridge_log) as fi, open(os.path.join(d, "bridge.jsonl"), "w") as fo:
                for ln in fi:
                    try:
                        t = json.loads(ln).get("t", 0)
                    except Exception:
                        continue
                    if t0 - 1 <= t <= t1 + 1:
                        fo.write(ln)
        if camera_log and os.path.exists(camera_log) and not os.path.exists(os.path.join(d, "camera_stats.txt")):
            lines = [ln for ln in open(camera_log, errors="replace").read().replace("\r", "\n").splitlines() if "rig:" in ln]
            open(os.path.join(d, "camera_stats.txt"), "w").write("\n".join(lines[-3:]) + "\n")
    return done


def salvage_session(session: str, reason: str) -> dict:
    out = {}
    logs = os.path.join(session, "logs")
    for d in sorted(glob.glob(os.path.join(session, "task", "episodes", "*", "*", "*", "episode_*"))):
        out[d] = salvage_episode(d, reason, os.path.join(logs, "bridge.jsonl"), os.path.join(logs, "camera.log"))
    return out
