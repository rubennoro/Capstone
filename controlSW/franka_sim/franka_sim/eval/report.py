"""``sim_report/v1``: report.json + a self-contained report.html per run dir, and DIR/summary.{json,html} over run dirs.

report.json: {schema, stage, policy_id, policy_kind, profile, sensor, robot, station_fingerprint, criteria_version,
image_source, seed, started, duration_s, metrics, gates, verdict, notes, rows}. ``rows`` (optional) are a few per-demo
rows ([{demo, ...numbers}]) for the HTML. The HTML has inline CSS only (no external assets, no scripts) and a banner
when ``image_source == "render"`` (rendered images: the success rate is not a real-world estimate).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

SCHEMA = "sim_report/v1"
REQUIRED = ("schema", "stage", "policy_id", "policy_kind", "profile", "sensor", "robot", "station_fingerprint",
            "criteria_version", "image_source", "seed", "started", "duration_s", "metrics", "gates", "verdict", "notes")
RENDER_BANNER = "rendered images: success is not a real-world estimate"


def make_report(*, stage, policy_id, policy_kind, profile, sensor, robot, station_fingerprint, image_source, seed, started,
                duration_s, metrics, evaluation, notes=(), rows=None) -> dict:
    rep = {"schema": SCHEMA, "stage": str(stage), "policy_id": policy_id, "policy_kind": policy_kind, "profile": profile,
           "sensor": sensor, "robot": robot, "station_fingerprint": station_fingerprint,
           "criteria_version": evaluation["criteria_version"], "image_source": image_source, "seed": seed,
           "started": started, "duration_s": round(float(duration_s), 2), "metrics": metrics,
           "gates": evaluation["gates"], "verdict": evaluation["verdict"], "notes": list(notes)}
    if rows:
        rep["rows"] = rows
    return rep


def validate(rep: dict) -> list[str]:
    """Problems with a report dict against sim_report/v1 (empty = valid)."""
    errs = [f"missing key {k}" for k in REQUIRED if k not in rep]
    if rep.get("schema") != SCHEMA:
        errs.append(f"schema {rep.get('schema')!r} != {SCHEMA}")
    if rep.get("verdict") not in ("PASS", "FAIL", "INCOMPLETE"):
        errs.append(f"verdict {rep.get('verdict')!r}")
    for n, g in (rep.get("gates") or {}).items():
        for k in ("value", "threshold", "op", "pass"):
            if k not in g:
                errs.append(f"gate {n}: missing {k}")
        if g.get("pass") not in (True, False, None):
            errs.append(f"gate {n}: pass {g.get('pass')!r}")
    return errs


def _clean(o):
    """JSON-safe: numpy scalars/arrays, NaN/inf -> None."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(o, np.generic):
            o = o.item()
        elif isinstance(o, np.ndarray):
            o = o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(_clean(obj), indent=1) + "\n")


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


CSS = """body{font-family:system-ui,sans-serif;margin:24px;color:#1b1f23;background:#fff}h1{font-size:20px;margin:0 0 4px}
h2{font-size:15px;margin:22px 0 6px}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #d0d7de;padding:4px 9px;text-align:left}
th{background:#f3f4f6}.badge{display:inline-block;padding:2px 12px;border-radius:12px;color:#fff;font-weight:600}
.PASS{background:#1a7f37}.FAIL{background:#cf222e}.INCOMPLETE{background:#9a6700}.ok{color:#1a7f37}.bad{color:#cf222e;font-weight:600}
.na{color:#9a6700}.banner{background:#fff8c5;border:1px solid #d4a72c;padding:8px 12px;margin:10px 0;font-weight:600}
.meta{color:#57606a;font-size:12px}code{background:#f3f4f6;padding:1px 4px}"""


def render_html(rep: dict) -> str:
    e = html.escape
    head = (f"<h1>{e(str(rep['policy_id']))} on {e(str(rep['sensor']))} / {e(str(rep['robot']))}, stage {e(str(rep['stage']))} "
            f"<span class='badge {e(rep['verdict'])}'>{e(rep['verdict'])}</span></h1>")
    meta = (f"<div class='meta'>profile <code>{e(str(rep['profile']))}</code> &middot; station fingerprint "
            f"<code>{e(str(rep['station_fingerprint']))}</code> &middot; criteria {e(str(rep['criteria_version']))} &middot; "
            f"image source {e(str(rep['image_source']))} &middot; seed {e(str(rep['seed']))} &middot; started "
            f"{e(str(rep['started']))} &middot; {rep['duration_s']} s</div>")
    banner = f"<div class='banner'>{RENDER_BANNER}</div>" if rep.get("image_source") == "render" else ""
    rows = []
    for name, g in rep["gates"].items():
        p = g["pass"]
        cls, txt = ("ok", "pass") if p is True else (("bad", "FAIL") if p is False else ("na", "incomplete"))
        rows.append(f"<tr><td>{e(name)}</td><td>{e(_fmt(g['value']))}</td><td>{e(g['op'])} {e(_fmt(g['threshold']))}</td>"
                    f"<td class='{cls}'>{txt}</td></tr>")
    gates = ("<h2>Gates</h2><table><tr><th>gate</th><th>value</th><th>threshold</th><th>result</th></tr>" + "".join(rows)
             + "</table>") if rows else "<h2>Gates</h2><p>none</p>"
    demo_rows = ""
    if rep.get("rows"):
        cols = list(rep["rows"][0].keys())
        demo_rows = ("<h2>Per demo</h2><table><tr>" + "".join(f"<th>{e(c)}</th>" for c in cols) + "</tr>"
                     + "".join("<tr>" + "".join(f"<td>{e(_fmt(r.get(c)))}</td>" for c in cols) + "</tr>"
                               for r in rep["rows"][:12]) + "</table>")
    scal = [(k, v) for k, v in rep["metrics"].items() if not isinstance(v, (dict, list))]
    met = ("<h2>Metrics</h2><table><tr><th>metric</th><th>value</th></tr>"
           + "".join(f"<tr><td>{e(k)}</td><td>{e(_fmt(v))}</td></tr>" for k, v in scal) + "</table>")
    notes = ("<h2>Notes</h2><ul>" + "".join(f"<li>{e(str(n))}</li>" for n in rep["notes"]) + "</ul>") if rep["notes"] else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{e(str(rep['policy_id']))} stage {e(str(rep['stage']))}</title>"
            f"<style>{CSS}</style></head><body>{head}{meta}{banner}{gates}{demo_rows}{met}{notes}</body></html>\n")


def write_report(run_dir, rep: dict) -> None:
    run_dir = Path(run_dir)
    write_json(run_dir / "report.json", rep)
    (run_dir / "report.html").write_text(render_html(_clean(rep)))


# ------------------------------------------------------------------------------------------------ summary
def _key_gates(rep: dict, n: int = 3) -> str:
    items = [(k, g) for k, g in rep["gates"].items() if g["pass"] is not True] or list(rep["gates"].items())
    return "; ".join(f"{k.split('.', 1)[-1]}={_fmt(g['value'])} ({'ok' if g['pass'] else 'FAIL' if g['pass'] is False else 'n/a'})"
                     for k, g in items[:n])


def summarize(out_dir) -> dict:
    """One row per run dir under ``out_dir`` that holds a valid report.json."""
    out_dir = Path(out_dir)
    rows = []
    for p in sorted(out_dir.glob("*/report.json")):
        try:
            rep = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if rep.get("schema") != SCHEMA:
            continue
        rows.append({"run": p.parent.name, "policy": rep["policy_id"], "profile": rep["profile"], "sensor": rep["sensor"],
                     "robot": rep["robot"], "stage": rep["stage"], "verdict": rep["verdict"], "key_gates": _key_gates(rep)})
    return {"schema": "sim_summary/v1", "n_runs": len(rows), "rows": rows}


def _rate(m: dict) -> str | None:
    """'0.67 [0.21, 0.94] (2/3)' from a report's success metrics, None when it has none."""
    if m.get("success_rate") is None:
        return None
    return (f"{_fmt(m['success_rate'])} [{_fmt(m.get('success_ci95_lo'))}, {_fmt(m.get('success_ci95_hi'))}] "
            f"({m.get('success_k')}/{m.get('success_n')})")


def comparison(out_dir) -> dict | None:
    """The sweep comparison: one column per (point, stage) with the sweep axis values as headers; rows = verdict, success rate
    (Wilson CI) and every gate (value + pass). None when ``out_dir`` has no ``sweep_index.json``."""
    out_dir = Path(out_dir)
    ip = out_dir / "sweep_index.json"
    if not ip.exists():
        return None
    try:
        idx = json.loads(ip.read_text())
    except (OSError, ValueError):
        return None
    cols = []
    for pt in idx.get("points", []):
        for st, ent in (pt.get("stages") or {}).items():
            rp = out_dir / ent["run_dir"] / "report.json"
            try:
                rep = json.loads(rp.read_text())
            except (OSError, ValueError):
                rep = None
            cols.append({"point": pt["i"], "stage": st, "axes": pt["axes"], "run": ent["run_dir"], "report": rep})
    gate_names = []
    for c in cols:
        for g in (c["report"] or {}).get("gates", {}):
            if g not in gate_names:
                gate_names.append(g)
    rows = [{"name": "verdict", "cells": [{"text": (c["report"] or {}).get("verdict", "not run")} for c in cols]},
            {"name": "success_rate (95% CI)", "cells": [{"text": _rate((c["report"] or {}).get("metrics", {})) or "n/a"} for c in cols]}]
    for g in gate_names:
        cells = []
        for c in cols:
            gg = ((c["report"] or {}).get("gates") or {}).get(g)
            cells.append({"text": "not run" if gg is None else _fmt(gg["value"]), "pass": None if gg is None else gg["pass"]})
        rows.append({"name": g, "cells": cells})
    return {"axes": idx.get("axes", []), "columns": [{k: c[k] for k in ("point", "stage", "axes", "run")} for c in cols], "rows": rows,
            "multi_stage": len(idx.get("stages", [])) > 1}


def comparison_html(cmp: dict) -> str:
    e = html.escape
    head = "".join(f"<tr><th>{e(str(a))}</th>" + "".join(f"<th>{e(str(c['axes'].get(a)))}</th>" for c in cmp["columns"]) + "</tr>"
                   for a in cmp["axes"])
    if cmp["multi_stage"]:
        head += "<tr><th>stage</th>" + "".join(f"<th>{e(c['stage'])}</th>" for c in cmp["columns"]) + "</tr>"
    head += "<tr><th>run</th>" + "".join(f"<th><a href='{e(c['run'])}/report.html'>{e(c['run'][:18])}</a></th>" for c in cmp["columns"]) + "</tr>"
    body = ""
    for r in cmp["rows"]:
        cells = ""
        for c in r["cells"]:
            if r["name"] == "verdict":
                cells += f"<td><span class='badge {e(c['text'].replace(' ', ''))}'>{e(c['text'])}</span></td>" if c["text"] in (
                    "PASS", "FAIL", "INCOMPLETE") else f"<td>{e(c['text'])}</td>"
            elif "pass" in c:
                cls = "ok" if c["pass"] is True else "bad" if c["pass"] is False else "na"
                cells += f"<td class='{cls}'>{e(c['text'])}</td>"
            else:
                cells += f"<td>{e(c['text'])}</td>"
        body += f"<tr><td>{e(r['name'])}</td>{cells}</tr>"
    return f"<h2>Sweep comparison</h2><table>{head}{body}</table>"


def write_summary(out_dir) -> dict:
    out_dir = Path(out_dir)
    s = summarize(out_dir)
    cmp = comparison(out_dir)
    if cmp is not None:
        s["comparison"] = cmp
    write_json(out_dir / "summary.json", s)
    e = html.escape
    cols = ["run", "policy", "profile", "sensor", "robot", "stage", "verdict", "key_gates"]
    body = "".join("<tr>" + "".join(
        (f"<td><a href='{e(r['run'])}/report.html'>{e(r['run'])}</a></td>" if c == "run" else
         f"<td><span class='badge {e(r[c])}'>{e(r[c])}</span></td>" if c == "verdict" else f"<td>{e(str(r[c]))}</td>")
        for c in cols) + "</tr>" for r in s["rows"])
    (out_dir / "summary.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>sim eval summary</title><style>{CSS}</style></head><body>"
        f"<h1>Sim evaluation summary</h1><div class='meta'>{s['n_runs']} run(s)</div><table><tr>"
        + "".join(f"<th>{c}</th>" for c in cols) + f"</tr>{body}</table>{comparison_html(cmp) if cmp else ''}</body></html>\n")
    return s
