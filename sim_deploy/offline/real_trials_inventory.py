"""W1 / S6 question 1 + 4 (logs part): inventory of the real-arm trials of 2026-11-20 and what the logs say was executed.

For every recorded trial folder (``~/Capstone/data/*/policy/*/deploy/2026-11-20_*_probe_live``) and every real session
(``~/policy_deploy_sessions/2026-11-20_HH-MM-SS``, not ``_mock``/``_pi_collect``/``_frames_alone``): checkpoint,
settings (precision, diffusion steps, exec_h, gripper_map, prompt), outcome, duration, chunks, the manager.log clamp
lines (absolute commanded TCP z below the box floor: count, min z, first/last time after the episode start),
accepted/skipped/shortened waypoints, image_age / frames / policy / cycle ms, anchor_dev_mm (p50/p95), the start pose
(TCP xyz in the base, tilt from straight down, headings), the probe set at the episode start.

Run (light env, CPU, a few seconds)::

    cd ~/Capstone-main-sim && unset PYTHONPATH && ~/envs/sim_policy_deploy/bin/python -m sim_deploy.offline.real_trials_inventory

Output: ``sim_evals/_pipeline/W1/real_trials/inventory.json`` and ``inventory.md`` (the table, also printed).
"""
from __future__ import annotations

import sys

import numpy as np

from sim_deploy.offline.real_trials_common import (OUT, SESSIONS, clamps, copy_version, episode_for_trial,
                                                   episode_t0_ns, pct, pose_summary, session, set_times, trials,
                                                   wall, write_json)


def session_names() -> list[str]:
    return sorted(p.name for p in SESSIONS.glob("2026-11-20_*")
                  if not any(p.name.endswith(s) for s in ("_mock", "_pi_collect", "_frames_alone")))


def episode_block(sess: dict, ei: int) -> dict:
    e = sess["episodes"][ei]
    eid = e["episode_id"]
    t0 = episode_t0_ns(sess, eid)
    steps = [s for s in sess["steps"] if s["episode_id"] == eid]
    sent = [s for s in steps if s.get("sent")]
    cl = [c for c in clamps(sess["name"]) if any(s["chunk_id"] == c["chunk_id"] for s in steps)]
    zs = [v for c in cl for (_, ax, v, _) in c["waypoints"] if ax == "z"]
    other_axes = sorted({ax for c in cl for (_, ax, _, _) in c["waypoints"] if ax != "z"})
    by_id = {s["chunk_id"]: s for s in steps}
    t_rel = [(by_id[c["chunk_id"]]["t_anchor_ns"] - t0) / 1e9 for c in cl if t0 is not None]
    ws = wall(e["wall_start"])
    clamp_k = [k for c in cl for (k, ax, _, _) in c["waypoints"] if ax == "z"]
    return {
        "episode_id": eid, "outcome": e["outcome"], "reason": e.get("reason"), "wall_start": e["wall_start"],
        "wall_end": e["wall_end"], "duration_s": round(e["duration_s"], 2), "steps_sent_episodes_jsonl": e["steps_sent"],
        "chunks_logged": len(steps), "chunks_sent": len(sent),
        "skips": e.get("skips"), "late": e.get("late"), "odd_chunks": e.get("odd_chunks"),
        "gate_shortened_pc": e.get("shortened"), "pulled_back": e.get("pulled_back"), "loop_late": e.get("loop_late"),
        "width_clips": e.get("width_clips"),
        "waypoints": {"accepted": sum(s["accepted"] for s in sent), "skipped": sum(s["skipped"] for s in sent),
                      "shortened": sum(s["shortened"] for s in sent), "rejected": sum(s["rejected"] for s in sent)},
        "per_chunk": {"accepted_mean": round(float(np.mean([s["accepted"] for s in sent])), 2) if sent else None,
                      "skipped_mean": round(float(np.mean([s["skipped"] for s in sent])), 2) if sent else None},
        "ms": {k: {"p50": pct([s.get(k) for s in sent], 50), "p95": pct([s.get(k) for s in sent], 95)}
               for k in ("image_age_ms", "frames_ms", "policy_ms", "cycle_ms", "margin_ms", "c3_rtt_ms")},
        "anchor_dev_mm": {"p50": pct([s.get("anchor_dev_mm") for s in sent], 50),
                          "p95": pct([s.get("anchor_dev_mm") for s in sent], 95),
                          "max": pct([s.get("anchor_dev_mm") for s in sent], 100)},
        "anchor_source": sorted({s.get("anchor_source") for s in sent}),
        "skip_reasons": sorted({s.get("skip_reason") for s in steps if not s.get("sent")}),
        "clamped": {"chunks": len(cl), "min_z_m": min(zs) if zs else None, "other_axes": other_axes,
                    "first_s_after_start": round(min(t_rel), 2) if t_rel else None,
                    "last_s_after_start": round(max(t_rel), 2) if t_rel else None,
                    "first_chunk_id": cl[0]["chunk_id"] if cl else None,
                    "waypoint_index_hist": {int(k): clamp_k.count(k) for k in sorted(set(clamp_k))}},
        "start_pose": pose_summary(e["start_pose"]), "start_pose_4x4": e["start_pose"],
        "t0_ns": t0, "wall_start_epoch": ws,
    }


def station_block(sess: dict) -> dict:
    st, su = sess["station"], sess["startup"] or {}
    pol, man = st.get("policy") or {}, st.get("manager") or {}
    sset = su.get("settings", {})
    pinfo = su.get("policy", {})
    return {"checkpoint": pol.get("checkpoint"), "precision": pol.get("precision"),
            "diffusion_steps": pol.get("diffusion_steps"), "exec_h": man.get("exec_h"),
            "prompt_dataset": sset.get("prompt_dataset"), "task_name": man.get("task_name"),
            "prompt_index": man.get("prompt_index", pol.get("prompt_index")),
            "gripper_map": sset.get("gripper_map"), "gripper_map_in_station_yaml": "gripper_map" in pol,
            "warmup_gripper_width_m": sset.get("warmup_gripper_width_m"),
            "epoch": pinfo.get("epoch"), "diffusion_steps_checkpoint": pinfo.get("diffusion_steps_checkpoint"),
            "model_target": pinfo.get("model_target"),
            "workspace_box": (sess["session"] or {}).get("settings", {}).get("workspace_box"),
            "F_T_TCP": (sess["session"] or {}).get("settings", {}).get("frame", {}).get("F_T_TCP"),
            "laptop_commit": (sess["session"] or {}).get("laptop_commit")}


def main() -> int:
    tr = trials()
    sessions = {n: session(n) for n in session_names()}
    rows, used = [], set()
    for t in tr:
        sess = sessions[t["session"]]
        times = set_times(t["folder"])
        ei, e = episode_for_trial(t, sess, times)
        used.add((t["session"], ei))
        ep = episode_block(sess, ei)
        ws, we = wall(e["wall_start"]), wall(e["wall_end"])
        i_start = int(np.searchsorted(times, ws)) - 1      # the last set written before the episode start
        rows.append({**t, "episode_index": ei, "station": station_block(sess), "episode": ep,
                     "probe": {"t_first": float(times[0]), "t_last": float(times[-1]),
                               "set_at_start": max(i_start, 0),
                               "set_at_start_age_s": round(float(ws - times[max(i_start, 0)]), 3),
                               "sets_during_episode": int(((times >= ws) & (times <= we)).sum()),
                               "first_set_s_before_start": round(float(ws - times[0]), 2),
                               "last_set_s_after_end": round(float(times[-1] - we), 2),
                               "rate_hz": round(float((len(times) - 1) / (times[-1] - times[0])), 2)}})
    unrecorded = []
    for n, sess in sessions.items():
        if not sess["episodes"]:
            unrecorded.append({"session": n, "episodes": 0, "station": station_block(sess),
                               "note": "no episode (startup only / aborted before start)"})
            continue
        for ei in range(len(sess["episodes"])):
            if (n, ei) not in used:
                unrecorded.append({"session": n, "episode_index": ei, "station": station_block(sess),
                                   "episode": episode_block(sess, ei), "note": "no probe recording"})
    out = {"copy_version": copy_version(), "trials": rows, "sessions_without_probe": unrecorded,
           "notes": ["start_pose = episodes.jsonl (the C2 target pose at the C3 start, robot base frame, TCP = "
                     "session F_T_TCP)", "clamp z = the ABSOLUTE commanded TCP z before clamping, from manager.log",
                     "accepted/skipped/shortened = C3 set_waypoints replies summed over the sent chunks "
                     "(steps.jsonl); gate_shortened_pc = the PC gate's counter delta (episodes.jsonl)"]}
    write_json(OUT / "inventory.json", out)
    md = table(rows, unrecorded)
    (OUT / "inventory.md").write_text(md)
    print(md)
    return 0


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(rows, unrecorded) -> str:
    h = ("| trial | session (ep) | ckpt (epoch) | prec/steps/exec_h | width map | dur s | outcome | chunks | clamped "
         "(min z m, first s) | wp acc/skip/short | gate short | img age p50/p95 | policy ms p50/p95 | anchor dev mm "
         "p50/p95 | start TCP xyz m, tilt, heading z |\n|" + "---|" * 15 + "\n")
    lines = []
    for r in rows + [u for u in unrecorded if "episode" in u]:
        e, s = r["episode"], r["station"]
        ck = s["checkpoint"].split("/policy/")[1].split("/")[0] if s["checkpoint"] else "-"
        gm = "B (0.051/0.090<->0.030/0.040)" if s["gripper_map"] else "none"
        sp = e["start_pose"]
        c = e["clamped"]
        lines.append(
            f"| {r.get('label', '(no probe)')} | {r['session']} ({e['episode_id']}) | {ck} ({s['epoch']}) | "
            f"{s['precision']}/{s['diffusion_steps']}/{s['exec_h']} | {gm} | {e['duration_s']:.1f} | "
            f"{e['outcome']}: {(e['reason'] or '')[-40:]} | {e['chunks_sent']} | "
            f"{c['chunks']} ({_f(c['min_z_m'], 3)}, {_f(c['first_s_after_start'])}) | "
            f"{e['waypoints']['accepted']}/{e['waypoints']['skipped']}/{e['waypoints']['shortened']} | "
            f"{e['gate_shortened_pc']} | {_f(e['ms']['image_age_ms']['p50'])}/{_f(e['ms']['image_age_ms']['p95'])} | "
            f"{_f(e['ms']['policy_ms']['p50'])}/{_f(e['ms']['policy_ms']['p95'])} | "
            f"{_f(e['anchor_dev_mm']['p50'])}/{_f(e['anchor_dev_mm']['p95'])} | "
            f"({', '.join(f'{v:.3f}' for v in sp['xyz_m'])}), {sp['tilt_deg']}, {sp['heading_tcp_z_deg']} |")
    return h + "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
