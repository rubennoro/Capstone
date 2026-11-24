"""metrics.json (rates with Wilson intervals), summary.html and the sim_evals/index.csv row."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

import numpy as np

INDEX_FIELDS = ("date", "task", "run_folder", "policy", "kind", "dataset", "rollouts", "seed", "tag", "look", "successes",
                "success_rate", "ci95_lo", "ci95_hi", "grasp_rate", "lift_rate", "faults", "stops", "timeouts",
                "eval_dir")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _q(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def metrics(rollouts: list[dict]) -> dict:
    """Rates over the rated rollouts: an "interrupted" one (the run was stopped during it, E7) is listed in
    rollouts.jsonl and counted in outcomes / interrupted, but is in no denominator."""
    recorded = rollouts
    rollouts = [e for e in recorded if e.get("outcome") != "interrupted"]
    n = len(rollouts)
    out = {"rollouts": n, "rollouts_recorded": len(recorded),
           "interrupted": len(recorded) - n}
    for key in ("success", "grasped", "lifted"):
        k = sum(1 for e in rollouts if e.get(key))
        lo, hi = wilson(k, n)
        out[key] = {"k": k, "rate": k / n if n else None, "ci95": [lo, hi]}
    out["outcomes"] = {}
    for e in recorded:
        out["outcomes"][e["outcome"]] = out["outcomes"].get(e["outcome"], 0) + 1
    out["faults"] = sum(1 for e in rollouts if e.get("fault"))
    out["stops"] = sum(1 for e in rollouts if e.get("stopped"))
    out["timeouts"] = sum(1 for e in rollouts if e["outcome"] == "timeout")
    for key in ("t_grasp_s", "t_lift_s", "t_success_s", "duration_s"):
        vals = [e.get(key) for e in rollouts if e.get(key) is not None]
        out[key] = {"n": len(vals), "p50": _q(vals, 50), "mean": float(np.mean(vals)) if vals else None}
    st = [e.get("steps") or {} for e in rollouts]
    out["steps"] = {
        "sent": int(sum(s.get("sent", 0) for s in st)),
        "skipped": int(sum(s.get("skipped", 0) for s in st)),
        "skip_reasons": _merge([s.get("skip_reasons", {}) for s in st]),
        "policy_ms_p50": _q([s.get("policy_ms_p50") for s in st], 50),
        "policy_ms_p95_max": max([s["policy_ms_p95"] for s in st if s.get("policy_ms_p95") is not None], default=None),
        "cycle_ms_p50": _q([s.get("cycle_ms_p50") for s in st], 50),
        "margin_ms_min": min([s["margin_ms_min"] for s in st if s.get("margin_ms_min") is not None], default=None),
        "margin_ms_p50": _q([s.get("margin_ms_p50") for s in st], 50),
    }
    return out


def _merge(ds):
    out: dict = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _pct(r) -> str:
    return "-" if r is None else f"{100 * r:.0f} %"


def _home_text(h: dict) -> str:
    """P2: how rollouts started (the start pose after h + p, or the deploy home after h)."""
    st = h.get("start_pose")
    if h.get("variant") == "start" and st:
        t = st["tcp_O_m"]
        return (f"start pose (h, then the PC's p): TCP O ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}), "
                f"{st['height_above_table_m'] * 100:.1f} cm above the table, tilted {st['tilt_deg']:.1f} deg, "
                f"{st['heading_words']}; q_start_rad {[round(x, 4) for x in st['q_start_rad']]}")
    return f"{h.get('variant', '?')} home (h only)"


def summary_html(eval_doc: dict, met: dict, rollouts: list[dict]) -> str:
    e = html.escape
    s = met["success"]
    rows = []
    for ro in rollouts:
        vid = ro.get("video")
        rows.append(
            "<tr>" + "".join(f"<td>{x}</td>" for x in (
                ro["n"], ro["seed_key"], e(ro["outcome"]), "yes" if ro.get("grasped") else "",
                "yes" if ro.get("lifted") else "", f"{ro['max_lift_m'] * 100:.1f}" if ro.get("max_lift_m") is not None
                else "-", "-" if ro.get("t_grasp_s") is None else f"{ro['t_grasp_s']:.1f}",
                f"{ro.get('duration_s', 0):.1f}", (ro.get("steps") or {}).get("sent", "-"),
                (ro.get("steps") or {}).get("skipped", "-"), e(ro.get("fault") or ro.get("note") or ""),
                f'<a href="{e(vid)}">mp4</a>' if vid else "")) + "</tr>")
    st = met["steps"]
    pol = eval_doc["policy"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>sim eval {e(pol['run_folder'])}</title>
<style>
:root{{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--ok:#1a7f37;--bad:#b42318}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--ok:#4ac26b;--bad:#f0645a}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1100px;padding:0 16px}}
h1{{font-size:20px;margin:0 0 4px}} .mut{{color:var(--mut)}} table{{border-collapse:collapse;width:100%;margin:12px 0}}
td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;font-variant-numeric:tabular-nums}}
.big{{font-size:28px;font-weight:600}} .kv td:first-child{{color:var(--mut);width:220px}}
</style></head><body>
<h1>{e(pol['task'])} / {e(pol['run_folder'])}</h1>
<div class="mut">{e(eval_doc['created'])} · tag {e(eval_doc['tag'])} · look {e((eval_doc.get('look') or {}).get('name', '?'))} · policy {e(pol['raw'])} ({e(pol['kind'])})</div>
<p><span class="big">{s['k']} / {met['rollouts']}</span> successes ({_pct(s['rate'])}, 95 % CI {_pct(s['ci95'][0])} to
{_pct(s['ci95'][1])}) · grasped {met['grasped']['k']} · lifted {met['lifted']['k']} · faults {met['faults']} ·
stops {met['stops']} · timeouts {met['timeouts']}</p>
<table class="kv">
<tr><td>success rule</td><td>eraser centre {eval_doc['judge']['lift_m'] * 100:.0f} cm above its start for
{eval_doc['judge']['hold_s']:.1f} s; timeout {eval_doc['judge']['timeout_s']:.0f} s</td></tr>
<tr><td>steps sent / skipped</td><td>{st['sent']} / {st['skipped']} {e(json.dumps(st['skip_reasons']))}</td></tr>
<tr><td>policy ms p50 / worst p95</td><td>{st['policy_ms_p50']} / {st['policy_ms_p95_max']}</td></tr>
<tr><td>margin ms p50 / min</td><td>{st['margin_ms_p50']} / {st['margin_ms_min']}</td></tr>
<tr><td>rollout start</td><td>{e(_home_text(eval_doc.get('home') or {}))}</td></tr>
<tr><td>checkpoint</td><td>{e(str(pol.get('checkpoint')))}</td></tr>
<tr><td>commits</td><td>sim_deploy {e(eval_doc['git']['sim_deploy'])} · training {e(str(eval_doc['git']['training']))}</td></tr>
<tr><td>look</td><td>{e((eval_doc.get('look') or {}).get('name', '?'))} ({e((eval_doc.get('look') or {}).get('station', ''))})</td></tr>
<tr><td>policy settings</td><td>{e(_settings_text(eval_doc))}</td></tr>
<tr><td>model</td><td>{e(eval_doc['model']['sha256'][:16])}…</td></tr>
</table>
<table><tr><th>#</th><th>seed</th><th>outcome</th><th>grasp</th><th>lift</th><th>max lift cm</th><th>t grasp s</th>
<th>time s</th><th>steps</th><th>skipped</th><th>fault / note</th><th>video</th></tr>
{''.join(rows)}</table>
</body></html>
"""


def _settings_text(eval_doc: dict) -> str:
    pol = eval_doc.get("policy") or {}
    sp, sm = pol.get("station_policy") or {}, pol.get("station_manager") or {}
    return (f"precision {sp.get('precision', '?')} · diffusion steps {sp.get('diffusion_steps', '?')} · "
            f"exec_h {sm.get('exec_h', '?')}")


def append_index(index_csv: Path, row: dict) -> None:
    """One row per eval folder: a resumed eval (E7, --resume) replaces its earlier row."""
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if index_csv.exists():
        with open(index_csv, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("eval_dir") != row.get("eval_dir")]
    rows.append({k: row.get(k, "") for k in INDEX_FIELDS})
    tmp = index_csv.with_name(index_csv.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(index_csv)
