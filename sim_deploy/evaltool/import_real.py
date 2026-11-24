"""Import real-arm trials as eval-like folders (SCHEMA.md, `sim_deploy.rollout/v2`, source real).

    python -m sim_deploy.evaltool import-real <trial_dir> [<trial_dir> ...] [--labels labels.yaml] [--out DIR]

A trial dir is a probe recording, ``~/Capstone/data/<task>/policy/<run>/deploy/<date>_<HH-MM>_trialN_probe_live/``
(policy_deploy TRIALS.md). Its ``NOTES.md`` names the policy_deploy session; the session's rollout (a line of its
``manager/episodes.jsonl``: policy_deploy calls a rollout an episode) whose time window overlaps the probe's PNG sets
most is the trial's rollout (or the one with the labels' ``episode_id``, the manager's id).

Output, per session: ``<out>/<task>/<run>/<session>/`` (default ``<out>`` = ``sim_evals/_real``)

- ``eval.yaml``: record_schema, source / launcher ``real``, checkpoint and the settings the session ran with, protocol
  (+ sha), the imported trials;
- ``rollouts.jsonl``: one line per imported rollout (= its rollout.json);
- ``rollouts/NNN/``, NNN = the rollout's line in the session's ``manager/episodes.jsonl`` (0-based):
  ``rollout.json`` (SCHEMA fields), ``session_rollout.json``, ``steps.jsonl``, ``NOTES.md`` (the trial's), and
  ``gt.jsonl`` only when the session saved the tap (``manager.save_tap``): the TCP path from each sent step's anchor;
- ``session/``: copies of station.yaml, manager/{session.yaml, episodes.jsonl, events.jsonl}, logs/manager.log,
  policy/startup.json.

What real trials can give: outcome / stop_reason from the operator's labels (null without), clamps (manager.log lines
"clamped into the workspace box" in the rollout's wall-time window), C2 counter deltas (shortened, pulled_back,
loop_late) and the gate's rejection from manager/episodes.jsonl, step timing from steps.jsonl, the TCP path only with
the tap.
Everything sim-only (placement, eraser pose, contacts, loadavg) is null and listed under ``missing``.

Labels (``--labels``, or ``<trial_dir>/labels.yaml``; the template: ``sim_deploy/evaltool/labels_template.yaml``):
either one trial's fields at the top level (then only one trial folder per call), or ``trials: {<task>/<run>/<trial
folder name>: {fields}}``; a ``trials:`` entry wins over the trial folder's own file. ``draft: true`` labels are kept
as ``outcome_draft`` / ``stop_reason_draft``; ``outcome`` / ``stop_reason`` stay null until the label is confirmed.

Reads only; writes only under ``<out>``. Light env (numpy, yaml), CPU, seconds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO
from sim_deploy.offline.real_trials_common import CLAMP_RE, WP_RE, episode_for_trial, load_jsonl, set_times, wall

RECORD_SCHEMA = "sim_deploy.rollout/v2"
DEFAULT_OUT = REPO / "sim_evals/_real"
SESSIONS = Path.home() / "policy_deploy_sessions"
OUTCOMES = ("success", "fail")
STOP_REASONS = ("success", "timeout", "fault", "user", "error")
CLAMP_SLACK_S = 1.0          # manager.log lines up to this long after wall_end still count (the last chunk's line)
SESSION_COPIES = ("station.yaml", "manager/session.yaml", "manager/episodes.jsonl", "manager/events.jsonl",
                  "logs/manager.log", "policy/startup.json")
SIM_ONLY = {"seed_key": "no seeds on the arm", "placement": "not measured on the arm",
            "eraser_settled_pose": "not measured on the arm", "start_pose_q": "the PC's joint start is not logged on "
            "the laptop (start_pose_tcp has the C2 target pose)", "loadavg_start": "not logged on the arm",
            "loadavg_end": "not logged on the arm"}


class TrialError(Exception):
    pass


# ------------------------------------------------------------------ finding the session and the rollout

def trial_info(d: Path) -> dict:
    m = re.search(r"_trial(\d+)_", d.name)
    if len(d.parts) < 5 or d.parent.name != "deploy" or d.parents[2].name != "policy":
        raise TrialError(f"{d}: not <task>/policy/<run>/deploy/<trial folder>")
    run, task = d.parents[1].name, d.parents[3].name
    return {"task": task, "run": run, "trial": int(m.group(1)) if m else None, "folder": str(d),
            "key": f"{task}/{run}/{d.name}", "label": f"{task}/{run[:7]}/trial{m.group(1) if m else '?'}",
            "notes": (d / "NOTES.md").read_text() if (d / "NOTES.md").is_file() else None}


def find_session(info: dict, labels: dict, times: np.ndarray | None) -> Path:
    if labels.get("session"):
        p = Path(str(labels["session"])).expanduser()
        return p if p.is_absolute() else SESSIONS / p
    m = re.search(r"(~?/\S*policy_deploy_sessions/[^\s)/]+)", info["notes"] or "")
    if m:
        p = Path(m.group(1)).expanduser()
        if p.is_dir():
            return p
    if times is not None and len(times):                    # no NOTES: the session with a rollout over the sets
        for s in sorted(SESSIONS.glob("2*")):
            for e in load_jsonl(s / "manager/episodes.jsonl"):
                if wall(e["wall_start"]) <= times[-1] and wall(e["wall_end"]) >= times[0]:
                    return s
    raise TrialError(f"{info['folder']}: no session (NOTES.md names none; give `session:` in the labels)")


def probe_times(d: Path) -> np.ndarray | None:
    if not any(d.glob("*_main_0.png")):
        return None
    try:
        return set_times(d)
    except (AssertionError, FileNotFoundError) as exc:
        print(f"  warning: probe set times of {d.name}: {exc}", file=sys.stderr)
        return None


def pick_rollout(info: dict, rollouts: list[dict], labels: dict, times: np.ndarray | None) -> int:
    """The index of the trial's rollout in the session's manager/episodes.jsonl lines (``rollouts``)."""
    if not rollouts:
        raise TrialError(f"{info['folder']}: the session has no rollout (manager/episodes.jsonl is empty)")
    if labels.get("episode_id") is not None:
        for i, e in enumerate(rollouts):
            if e["episode_id"] == int(labels["episode_id"]):
                return i
        raise TrialError(f"{info['folder']}: episode_id {labels['episode_id']} not in the session")
    if times is None:
        if len(rollouts) > 1:
            raise TrialError(f"{info['folder']}: no probe sets and {len(rollouts)} rollouts: give `episode_id`")
        return 0
    i, _ = episode_for_trial(info, {"episodes": rollouts}, times)      # offline helper, its own key
    e = rollouts[i]
    if not ((times >= wall(e["wall_start"])) & (times <= wall(e["wall_end"]))).any():
        raise TrialError(f"{info['folder']}: no session rollout overlaps the probe sets: give `episode_id`")
    return i


# ------------------------------------------------------------------ the per-rollout numbers

def clamp_lines(log: Path, t_a: float, t_b: float, chunk_ids: set[int]) -> dict:
    zs, axes, first, n, n_chunk = [], set(), None, 0, 0
    for line in log.read_text().splitlines() if log.is_file() else []:
        m = CLAMP_RE.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f").timestamp()
        if int(m.group(2)) in chunk_ids:
            n_chunk += 1
        if not t_a <= t <= t_b + CLAMP_SLACK_S:
            continue
        n += 1
        first = t - t_a if first is None else first
        for _, ax, v, _ in WP_RE.findall(m.group(3)):
            axes.add(ax)
            if ax == "z":
                zs.append(float(v))
    return {"count": n, "count_by_chunk_id": n_chunk, "min_commanded_z_m": min(zs) if zs else None,
            "axes": sorted(axes), "first_s_after_start": round(first, 2) if first is not None else None}


def step_summary(steps: list[dict]) -> dict:
    """The sim driver's _step_summary keys, plus policy_ms p95 and the C3 reply sums."""
    sent = [s for s in steps if s.get("sent")]
    reasons: dict[str, int] = {}
    for s in steps:
        if not s.get("sent"):
            k = s.get("skip_reason") or "?"
            reasons[k] = reasons.get(k, 0) + 1

    def q(key, p):
        v = [s[key] for s in sent if s.get(key) is not None]
        return round(float(np.percentile(v, p)), 2) if v else None

    margins = [s["margin_ms"] for s in sent if s.get("margin_ms") is not None]
    return {"sent": len(sent), "skipped": len(steps) - len(sent), "skip_reasons": reasons,
            "policy_ms_p50": q("policy_ms", 50), "policy_ms_p95": q("policy_ms", 95),
            "cycle_ms_p50": q("cycle_ms", 50), "image_age_ms_p50": q("image_age_ms", 50),
            "image_age_ms_p95": q("image_age_ms", 95),
            "margin_ms_min": round(min(margins), 2) if margins else None, "margin_ms_p50": q("margin_ms", 50),
            "c3_rtt_ms_p50": q("c3_rtt_ms", 50), "anchor_dev_mm_p50": q("anchor_dev_mm", 50),
            "anchor_dev_mm_p95": q("anchor_dev_mm", 95),
            "waypoints": {k: int(sum(s.get(k, 0) for s in sent))
                          for k in ("accepted", "skipped", "shortened", "rejected")},
            "rejected": int(sum(s.get("rejected", 0) for s in steps)),
            "shortened": int(sum(s.get("shortened", 0) for s in steps))}


def fault_rule(reason: str) -> str | None:
    r = reason.lower()
    if "gate rejected" not in r:
        return None
    for rule, words in (("box", ("workspace box",)), ("cone", ("cone",)), ("speed", ("speed", "velocity")),
                        ("lead", ("lead",))):
        if any(w in r for w in words):
            return rule
    return "other"


def stop_reason_from_log(e: dict) -> str:
    r = (e.get("reason") or "").lower()
    if e["outcome"] == "success":
        return "success"
    if "user stop" in r or r == "operator":
        return "user"          # the arm's user stop, or the operator's n key
    if "fault" in r or "gate rejected" in r:
        return "fault"
    if "timeout" in r or "timed out" in r:
        return "timeout"
    return "error"


def c2_block(e: dict) -> dict:
    rule = fault_rule(e.get("reason") or "")
    return {"accepted": None, "shortened": e.get("shortened"), "pulled_back": e.get("pulled_back"),
            "rejected": 1 if rule else 0, "loop_late": e.get("loop_late"),
            "modes": {"fault": 1} if "pc reports fault" in (e.get("reason") or "").lower() else None,
            "errors": None, "fault_rule": rule,
            "source": "manager/episodes.jsonl: the PC gate's counter deltas over the rollout; rejected from its reason"}


def tap_path(tap_dir: Path, t0_ns: int | None) -> tuple[list[dict], dict] | None:
    """gt.jsonl-equivalent rows from the tap: per sent step the anchor pose (the TCP pose the observation was made
    relative to, base frame) at its image time, and the newest gripper width; plus the commanded waypoints' min z."""
    files = sorted(tap_dir.glob("chunk_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        return None
    rows, wp_z = [], []
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            T = np.asarray(z["anchor_T"], dtype=np.float64)
            t_ns = int(z["anchor_t_ns"])
            gap = float(np.asarray(z["gripper_width_m"]).reshape(-1)[-1])
            src = str(z["anchor_source"])
            if "waypoints_O_T_TCP" in z.files and z["waypoints_O_T_TCP"].size:
                wp_z.append(float(np.asarray(z["waypoints_O_T_TCP"])[:, 2, 3].min()))
        t0 = t0_ns if t0_ns is not None else t_ns
        rows.append({"t": round((t_ns - t0) / 1e9, 3), "eraser_pose": None,
                     "tcp_pose": [round(float(v), 5) for v in (*T[:3, 3], *_quat(T[:3, :3]))], "gap_m": round(gap, 5),
                     "contacts_eraser": None, "contacts_table": None, "min_z_fingers_rig_m": None, "c2_mode": None,
                     "reflex": None, "chunk_id": int(f.stem.split("_")[1]), "anchor_source": src})
    zs = [r["tcp_pose"][2] for r in rows]
    return rows, {"rows": len(rows), "min_tcp_z_m": min(zs), "min_commanded_tcp_z_m": min(wp_z) if wp_z else None,
                  "source": "tap anchors (one row per sent step, not 20 Hz): tcp_pose is the anchor = the C2 TARGET pose "
                            "(measured only after an anchor fallback, see anchor_source), not the measured TCP"}


def _quat(R: np.ndarray) -> list[float]:
    """[qw, qx, qy, qz] of a rotation matrix (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return (q if q[0] >= 0 else -q).tolist()


def rollout_t0_ns(events: list[dict], episode_id: int) -> int | None:
    """The manager's start event of the rollout with this episode_id (policy_deploy's names)."""
    for ev in events:
        if ev.get("kind") == "episode_start" and ev.get("detail") == f"episode {episode_id}":
            return int(ev["t_ns"])
    return None


# ------------------------------------------------------------------ labels

def load_labels(path: Path | None) -> dict:
    """{trial key or '*': fields}; '*' = a single-trial file (fields at the top level)."""
    if path is None or not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    if "trials" in doc:
        default_draft = doc.get("draft")
        out = {}
        for k, v in (doc["trials"] or {}).items():
            v = dict(v or {})
            if default_draft is not None:
                v.setdefault("draft", default_draft)
            out[str(k).strip("/")] = {**v, "_file": str(path)}
        return out
    return {"*": {**doc, "_file": str(path)}}


def labels_for(info: dict, cli: dict) -> dict:
    own = load_labels(Path(info["folder"]) / "labels.yaml").get("*", {})
    lab = cli.get(info["key"]) or cli.get(Path(info["folder"]).name) or own or cli.get("*") or {}
    for k, allowed in (("outcome", OUTCOMES), ("stop_reason", STOP_REASONS)):
        if lab.get(k) is not None and lab[k] not in allowed:
            raise TrialError(f"{info['key']}: labels {k} {lab[k]!r} not one of {allowed}")
    return lab


# ------------------------------------------------------------------ one trial

def station_settings(sess_dir: Path) -> dict:
    def y(p):
        return yaml.safe_load(p.read_text()) if p.is_file() else {}

    st = y(sess_dir / "station.yaml") or {}
    ses = y(sess_dir / "manager/session.yaml") or {}
    su = json.loads((sess_dir / "policy/startup.json").read_text()) if (sess_dir / "policy/startup.json").is_file() \
        else {}
    pol, man, sset, ms = st.get("policy") or {}, st.get("manager") or {}, su.get("settings") or {}, \
        ses.get("settings") or {}
    pinfo = su.get("policy") or {}
    kind = pol.get("kind")
    if kind == "diffusion" and sset.get("prompt_dataset"):
        kind = "prompting"
    return {
        "checkpoint": sset.get("checkpoint") or pol.get("checkpoint"), "kind": kind,
        "precision": sset.get("precision", pol.get("precision")),
        "diffusion_steps": sset.get("diffusion_steps", pol.get("diffusion_steps")),
        "diffusion_steps_checkpoint": pinfo.get("diffusion_steps_checkpoint"), "epoch": pinfo.get("epoch"),
        "exec_h": ms.get("exec_h", man.get("exec_h")), "lookahead_ms": ms.get("lookahead_ms"),
        "prompt_dataset": sset.get("prompt_dataset"), "task_name": ms.get("task_name"),
        "prompt_index": ms.get("prompt_index"), "gripper_map": sset.get("gripper_map"),
        "warmup_gripper_width_m": sset.get("warmup_gripper_width_m"), "device": sset.get("device"),
        "workspace_box": ms.get("workspace_box"),     # already shrunk by manager.workspace_clamp_m (settings.py)
        "save_tap": ms.get("save_tap"), "tap_images_every": ms.get("tap_images_every"),
        "F_T_TCP": (ms.get("frame") or {}).get("F_T_TCP"),
        "laptop_commit": ses.get("laptop_commit"), "pc_commit": ses.get("pc_commit"),
        "pc_limits": {k: (ses.get("hello") or {}).get(k) for k in ("gripper_width_max_m", "max_lead_s",
                                                                    "watchdog_limit_s")},
    }


def protocol_of(s: dict) -> dict:
    """The SCHEMA protocol keys that exist on the arm; the sim-only ones null (so real never matches a sim batch)."""
    return {"source": "real", "seed_rule": None, "timeout_s": None, "precision": s["precision"],
            "diffusion_steps": s["diffusion_steps"], "exec_h": s["exec_h"], "look": "real", "model_sha256": None,
            "q_start_rad": None, "judge": "operator labels",
            "real_limits": {"box": s["workspace_box"], "v_max": None, "w_max": None, "cone": None,
                            **s["pc_limits"]}}


def sha16(doc: dict) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def import_trial(d: Path, cli_labels: dict, out_root: Path) -> dict:
    info = trial_info(d)
    lab = labels_for(info, cli_labels)
    times = probe_times(d)
    sess_dir = find_session(info, lab, times)
    rollouts = load_jsonl(sess_dir / "manager/episodes.jsonl")
    ei = pick_rollout(info, rollouts, lab, times)
    e = rollouts[ei]
    eid = e["episode_id"]
    events = load_jsonl(sess_dir / "manager/events.jsonl")
    steps = [s for s in load_jsonl(sess_dir / "manager/steps.jsonl") if s.get("episode_id") == eid]
    t_a, t_b = wall(e["wall_start"]), wall(e["wall_end"])
    cl = clamp_lines(sess_dir / "logs/manager.log", t_a, t_b, {s["chunk_id"] for s in steps})
    settings = station_settings(sess_dir)

    out = out_root / info["task"] / info["run"] / sess_dir.name
    rollout_dir = out / "rollouts" / f"{ei:03d}"
    for old in out.glob("rollouts/*/rollout.json"):        # this trial matched another rollout before
        if old.parent.name != f"{ei:03d}" and json.loads(old.read_text())["trial"]["key"] == info["key"]:
            shutil.rmtree(old.parent)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "session_rollout.json").write_text(json.dumps(e, indent=1))
    (rollout_dir / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    if info["notes"] is not None:
        (rollout_dir / "NOTES.md").write_text(info["notes"])
    for rel in SESSION_COPIES:
        src = sess_dir / rel
        if src.is_file():
            (out / "session" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / "session" / rel)

    missing = dict(SIM_ONLY)
    tap = sess_dir / "manager/tap" / f"ep_{eid:04d}"
    path = tap_path(tap, rollout_t0_ns(events, eid)) if tap.is_dir() else None
    if path:
        rows, path_summary = path
        (rollout_dir / "gt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        path_summary = None
        (rollout_dir / "gt.jsonl").unlink(missing_ok=True)
        missing["path"] = "manager.save_tap was off: no TCP path, no min z"
    draft = bool(lab.get("draft"))
    verdict = None if draft else lab.get("outcome")      # an unconfirmed (draft) label is not a verdict
    if lab.get("outcome") is None:
        missing["outcome"] = "no operator label"
    elif draft:
        missing["outcome"] = "draft label (labels.draft true): confirm it, set draft false and re-import"
    mp4 = d / f"{d.name}.mp4"
    rec = {
        "record_schema": RECORD_SCHEMA, "source": "real", "launcher": "real", "batch_id": None, "n": ei,
        **{k: None for k in SIM_ONLY},
        "outcome": verdict, "stop_reason": None if draft else lab.get("stop_reason"),
        "outcome_draft": lab.get("outcome") if draft else None,
        "stop_reason_draft": lab.get("stop_reason") if draft else None,
        "stop_reason_from_log": stop_reason_from_log(e),
        "success": None if verdict is None else verdict == "success",
        "grasped": None if draft else lab.get("grasped"), "lifted": None if draft else lab.get("lifted"),
        "labels": {k: v for k, v in lab.items() if k not in ("session", "episode_id")} or None,
        "duration_s": round(e["duration_s"], 3),
        "start_pose_tcp": e.get("start_pose"),
        "clamps": cl["count"], "clamps_detail": cl,
        "c2": c2_block(e), "loop_late": e.get("loop_late"),
        "step_summary": step_summary(steps),
        "session_counts": {k: e.get(k) for k in ("steps_sent", "skips", "late", "anchor_fallbacks", "odd_chunks",
                                                 "width_clips")},
        "path": path_summary, "min_z_tcp_m": path_summary["min_tcp_z_m"] if path_summary else None,
        "session": {"dir": str(sess_dir), "episode_id": eid, "rollout_index": ei, "pd_outcome": e["outcome"],
                    "pd_reason": e.get("reason"), "wall_start": e["wall_start"], "wall_end": e["wall_end"]},
        "trial": {k: info[k] for k in ("task", "run", "trial", "folder", "key", "label")} |
                 {"probe_sets": int(len(times)) if times is not None else 0,
                  "video": str(mp4) if mp4.is_file() else None},
        "missing": missing,
    }
    (rollout_dir / "rollout.json").write_text(json.dumps(rec, indent=1))
    write_eval(out, info, settings, sess_dir)
    return rec


def write_eval(out: Path, info: dict, settings: dict, sess_dir: Path) -> None:
    recs = [json.loads(p.read_text()) for p in sorted(out.glob("rollouts/*/rollout.json"))]
    (out / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    protocol = protocol_of(settings)
    doc = {
        "schema": "sim_deploy.eval/v2", "record_schema": RECORD_SCHEMA, "source": "real", "launcher": "real",
        "batch_id": None, "created": sess_dir.name, "imported": datetime.now().isoformat(timespec="seconds"),
        "command": "python -m sim_deploy.evaltool import-real " + " ".join(r["trial"]["folder"] for r in recs),
        "tag": "real",
        "policy": {"kind": settings["kind"], "task": info["task"], "run_folder": info["run"],
                   "checkpoint": settings["checkpoint"], "dataset": settings["prompt_dataset"],
                   "settings": {k: v for k, v in settings.items() if k not in ("checkpoint", "kind")}},
        "rollouts": len(recs), "look": {"name": "real"},
        "protocol": protocol, "protocol_sha": sha16(protocol),
        "session_dir": str(sess_dir),
        "git": {"laptop": settings["laptop_commit"], "pc": settings["pc_commit"]},
        "trials": [{"n": r["n"], "episode_id": r["session"]["episode_id"], **r["trial"]} for r in recs],
    }
    doc = json.loads(json.dumps(doc))       # no YAML anchors for the shared lists
    (out / "eval.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))


# ------------------------------------------------------------------ CLI

def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("trial_dirs", nargs="+", help="<task>/policy/<run>/deploy/<date>_<HH-MM>_trialN_probe_live")
    p.add_argument("--labels", default=None, help="labels.yaml (one trial, or trials: {<task>/<run>/<folder>: ...})")
    p.add_argument("--out", default=None, help=f"root of the eval-like folders (default {DEFAULT_OUT})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim_deploy.evaltool import-real", description=__doc__.split("\n")[0])
    add_arguments(p)
    a = p.parse_args(argv)
    cli = load_labels(Path(a.labels).expanduser()) if a.labels else {}
    if "*" in cli and len(a.trial_dirs) > 1:
        print(f"import-real: {a.labels} labels one trial; give one trial folder, or use `trials:` keys",
              file=sys.stderr)
        return 2
    if a.labels and not cli:
        print(f"import-real: {a.labels}: no labels read", file=sys.stderr)
        return 2
    out_root = Path(a.out).expanduser() if a.out else DEFAULT_OUT
    bad = 0
    for t in a.trial_dirs:
        d = Path(t).expanduser().resolve()
        try:
            if not d.is_dir():
                raise TrialError(f"{d}: not a folder")
            r = import_trial(d, cli, out_root)
        except (TrialError, OSError, ValueError, KeyError) as exc:
            print(f"import-real: FAIL {t}: {exc}", file=sys.stderr)
            bad += 1
            continue
        lab = r["labels"] or {}
        print(f"{r['trial']['label']:28s} {Path(r['session']['dir']).name} episode_id {r['session']['episode_id']} -> "
              f"{r['trial']['task']}/{r['trial']['run']}/{Path(r['session']['dir']).name}/rollouts/{r['n']:03d}: "
              f"outcome {r['outcome'] or r['outcome_draft']}{' (draft)' if lab.get('draft') else ''}, "
              f"stop {r['stop_reason'] or r['stop_reason_draft']} "
              f"(log: {r['stop_reason_from_log']}), clamps {r['clamps']}, {r['duration_s']:.1f} s, "
              f"path {'yes' if r['path'] else 'no'}")
    print(f"import-real: {len(a.trial_dirs) - bad} imported, {bad} failed; under {out_root}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
