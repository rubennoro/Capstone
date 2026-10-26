"""Stage 1b wrapper: A8's full-stack demo-replay campaign (``franka_sim.campaign.run_campaign``, I-CAMPAIGN), gated by
``criteria.stage1b``.

Why: Stage 1 (A3) says whether each demo is kinematically feasible with the station's tooling; Stage 1b replays every demo
through the whole stack (launcher, bridge, MuJoCo arm + hand + scene, render sender, ``policy: demo_replay``) and asks
whether the controller tracks it without faults. This module is thin: it finds the Stage 1 result, calls
``run_campaign(config, profile, stage1_json, demos, speeds, out_dir, sets, jobs)`` and maps its ``campaign.json`` rows
into flat metrics. A8 owns the campaign (speeds, alignment, the tracking window); nothing here re-measures it.

Metrics (gates are the bases of ``criteria.stage1b`` ``*_min`` / ``*_max`` keys; ``track_pos_m`` / ``track_rot_deg`` are the
campaign's tracking tolerances, not gates):
  track_ok_fraction   feasible, fault-free runs with ``track_ok`` / feasible, fault-free runs, at the primary (first) speed
  reflex              reflex count over the primary-speed runs
  per speed: ``track_ok_fraction_s<v>``, Wilson 95 % CI, ``reflex_s<v>``, ``pad_table_s<v>``, ``system_faults_s<v>``.
A system fault (launcher / sender / bridge died, timeouts) is plumbing, not tracking: any fault makes the stage INCOMPLETE.

Inputs: ``--stage1 DIR`` (a Stage 1 run dir, or a stage1.json / .json.gz file), else the Stage 1 result of the same
``--stage all`` run; ``--demos`` / ``--episodes`` pick demos (default all); ``--speeds`` (default ``sim.stage1b.speeds``,
else ``DEFAULTS``); ``--jobs``.
"""

from __future__ import annotations

import json
from pathlib import Path

from franka_sim.eval import runner as rn

DEFAULTS = {"speeds": [0.5]}          # code default when the station's sim group has no stage1b block (A8 spec: speed 0.5 first)


def _stage1_json(R: rn.Run, results: dict) -> Path:
    from franka_sim.eval import stage1 as s1w

    d = getattr(R.args, "stage1", None)
    if d:
        p = Path(d)
        if p.is_file():
            return p
        q = s1w.stage1_json_in(p)
        if q is not None:
            return q
        raise rn.StageIncomplete(f"--stage1 {d}: no stage1.json there")
    if results.get("stage1_json"):
        return Path(results["stage1_json"])
    raise rn.StageIncomplete("Stage 1b needs a Stage 1 result: --stage1 DIR (or run --stage all)")


def speeds_of(R: rn.Run) -> list:
    a = getattr(R.args, "speeds", None)
    if a:
        return [float(x) for x in str(a).split(",") if x.strip()]
    cfg = R.station.get("sim.stage1b", {}) or {}
    return [float(x) for x in (cfg.get("speeds") or DEFAULTS["speeds"])]


def _f(row: dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _is_fault(row: dict) -> bool:
    v = _f(row, "system_fault", "fault", "faults", default=None)
    if isinstance(v, (list, tuple)):
        return bool(v)
    return bool(v) or str(_f(row, "status", "class", default="")).lower() in ("system_fault", "fault")


def _is_infeasible(row: dict) -> bool:
    if "stage1_infeasible" in row:
        return bool(row["stage1_infeasible"])
    labels = row.get("labels") or row.get("label") or []
    return "stage1_infeasible" in (labels if isinstance(labels, (list, tuple)) else [labels])


def _count(row: dict, *keys) -> int:
    v = _f(row, *keys, default=0)
    return len(v) if isinstance(v, (list, tuple)) else int(v or 0)


def metrics_from(camp: dict, speeds: list) -> tuple[dict, list, list]:
    """(metrics, table rows, notes) from campaign.json content."""
    from franka_deploy.eval.metrics import wilson

    rows = list(camp.get("rows") or camp.get("runs") or [])
    notes, table = [], []
    m: dict = {"n_runs": len(rows), "speeds": list(speeds), "campaign_schema": camp.get("schema")}
    by_speed: dict = {}
    for r in rows:
        by_speed.setdefault(float(_f(r, "speed", "speed_scale", default=speeds[0] if speeds else 1.0)), []).append(r)
    for sp in sorted(by_speed, key=lambda x: (x not in speeds, speeds.index(x) if x in speeds else x)):
        rs = by_speed[sp]
        faults = [r for r in rs if _is_fault(r)]
        infeas = [r for r in rs if _is_infeasible(r)]
        den = [r for r in rs if not _is_fault(r) and not _is_infeasible(r)]
        k = sum(bool(_f(r, "track_ok", default=False)) for r in den)
        p, lo, hi = wilson(k, len(den))
        tag = f"s{sp:g}"
        m.update({f"track_ok_fraction_{tag}": p if den else None, f"track_ok_ci95_lo_{tag}": lo if den else None,
                  f"track_ok_ci95_hi_{tag}": hi if den else None, f"track_ok_k_{tag}": k, f"track_ok_n_{tag}": len(den),
                  f"n_runs_{tag}": len(rs), f"stage1_infeasible_{tag}": len(infeas), f"system_faults_{tag}": len(faults),
                  f"reflex_{tag}": sum(_count(r, "reflexes", "reflex", "n_reflexes") for r in rs),
                  f"pad_table_{tag}": sum(_count(r, "pad_table", "pad_table_contacts", "n_pad_table") for r in rs)})
        failing = [str(_f(r, "demo", "demo_index", default="?")) for r in den if not _f(r, "track_ok", default=False)]
        if failing:
            notes.append(f"speed {sp:g}: tracking failed on demos {', '.join(failing)}")
        if infeas:
            notes.append(f"speed {sp:g}: stage1_infeasible demos (run, excluded from the tracking denominator): "
                         + ", ".join(str(_f(r, "demo", "demo_index", default="?")) for r in infeas))
        if faults:
            notes.append(f"speed {sp:g}: system faults on demos "
                         + ", ".join(f"{_f(r, 'demo', 'demo_index', default='?')} ({_f(r, 'fault_reason', 'fault', 'reason', default='?')})"
                                     for r in faults))
        for r in rs:
            table.append({"speed": sp, "demo": _f(r, "demo", "demo_index"), "track_ok": _f(r, "track_ok"),
                          "track_fraction": _f(r, "track_fraction", "track_ok_fraction"),
                          "stage1_infeasible": _is_infeasible(r), "system_fault": _is_fault(r),
                          "reflexes": _count(r, "reflexes", "reflex", "n_reflexes"),
                          "pad_table": _count(r, "pad_table", "pad_table_contacts", "n_pad_table"),
                          "session": _f(r, "session", "session_dir")})
    primary = speeds[0] if speeds else (sorted(by_speed)[0] if by_speed else None)
    tag = None if primary is None else f"s{float(primary):g}"
    m["primary_speed"] = primary
    m["track_ok_fraction"] = m.get(f"track_ok_fraction_{tag}")
    m["reflex"] = m.get(f"reflex_{tag}")
    m["pad_table_contacts"] = m.get(f"pad_table_{tag}")
    m["system_faults"] = int(sum(v for k, v in m.items() if k.startswith("system_faults_s")))
    if m["system_faults"]:
        m["incomplete_reason"] = f"{m['system_faults']} campaign run(s) ended in a system fault (plumbing, not tracking)"
    if not rows:
        m["incomplete_reason"] = "campaign.json has no runs"
    for k in ("fingerprint", "config_fingerprint", "overrides", "sets"):
        if camp.get(k) is not None:
            m[f"campaign_{k}"] = camp[k]
    notes.append(f"gated at the primary speed {primary}; reflexes count as failures (reflex gate), pad_table contacts are reported "
                 "apart (A6 review finding 5)")
    return m, table, notes


def run(R: rn.Run, d: Path, results: dict) -> dict:
    s1 = _stage1_json(R, results)
    speeds = speeds_of(R)
    demos = R.demo_list()
    if demos is None and getattr(R.args, "episodes", None):
        demos = list(range(int(R.args.episodes)))
    try:
        from franka_sim.campaign import run_campaign
    except ImportError as e:
        raise rn.StageIncomplete(f"franka_sim.campaign (A8) is not available: {e}") from e
    camp = run_campaign(config=str(R.cfg["config_path"]), profile=R.profile, stage1_json=str(s1),
                        demos="all" if demos is None else demos, speeds=speeds, out_dir=str(d / "campaign"),
                        sets=tuple(R.user_sets), jobs=max(1, int(getattr(R.args, "jobs", 1) or 1)))
    if not isinstance(camp, dict):
        cj = d / "campaign" / "campaign.json"
        camp = json.loads(cj.read_text()) if cj.is_file() else {}
    m, table, notes = metrics_from(camp, speeds)
    notes = [f"stage1.json: {s1}", f"campaign dir: {d / 'campaign'} (campaign.json, campaign.txt)"] + notes
    if R.user_sets:
        notes.append("run overrides: " + ", ".join(R.user_sets))
    return {"metrics": m, "rows": table, "notes": notes}
