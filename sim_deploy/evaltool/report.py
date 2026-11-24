"""Cross-eval report: select eval folders, score them (in memory; no eval folder is written), group, write
``<out>/{report.md, report.html, rollouts.csv, checkpoints.csv, kind_x_look.csv}``.

    build_report(root, select, batch, protocol, group, out) -> dict

Grouping keys: checkpoint (= <task>/<run_folder>), kind (BP / plain / hold_still / replay / unknown, from
sim_deploy.ui.catalog: BP = no TRAIN_CONFIG = umi_policy_dunetp, plain = umi_policy_dunet), look (eval.yaml look.name,
"none" for folders older than the looks), tag, batch, protocol.

Per group also (2026-11-22): success within 20 s / 30 s (t_success) besides the timeout's rate; the start split of a
start set (demo starts the checkpoint trained on / its held-out demo / the real-trial poses); policy_ms and cycle_ms
p50 / p90 pooled over every sent chunk, next to the real arm's (``real_timing``: the imported trials under
sim_evals/_real, else REAL_TIMING_FALLBACK).
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from sim_deploy import REPO
from sim_deploy.eval.report import wilson
from sim_deploy.evaltool import constants as C
from sim_deploy.evaltool.score import _json, _jsonl, _q, load_eval, score_eval

SIM_EVALS = REPO / "sim_evals"
SKIP_DIRS = ("_dry_run", "_reports")
# W1 step 3 (sim-experiments/03_policy_pipeline/W1_offline): first chunk on real training episodes, row 6, k = 15
W1_OPENLOOP_SUMMARY = SIM_EVALS / "_pipeline" / "W1" / "openloop" / "summary.json"
GROUP_KEYS = ("checkpoint", "kind", "look", "origin", "tag", "batch", "protocol")
DATA_ROOT = Path.home() / "Capstone" / "data"          # <task>/policy/<run_folder> (real imports have no policy.raw)
KIND_NAMES = {"prompting": "BP", "plain": "plain"}
REAL_ROOT = SIM_EVALS / "_real"
# The real arm's chunk timing (2026-11-20 trials, policy_deploy steps.jsonl, sent chunks) when sim_evals/_real has no
# steps for a checkpoint: run_006 (BP) policy p50 119 ms / cycle 132 ms; the plain runs ~97 / ~110 ms.
REAL_TIMING_FALLBACK = {"PICK_TEST1/run_006_2026-11-18": {"policy_ms_p50": 119.0, "cycle_ms_p50": 132.0},
                        "PICK_TEST1/run_022_2026-11-19": {"policy_ms_p50": 97.0, "cycle_ms_p50": 110.0},
                        "PICK_DOUBLE/run_002_2026-11-19": {"policy_ms_p50": 97.0, "cycle_ms_p50": 110.0}}
START_GROUPS = (("demo_trained", "demos trained"), ("demo_held_out", "demo held out"), ("real_trial", "real-trial poses"))


# ----------------------------------------------------------------------------------------------------- selection

def discover(root: Path) -> list[Path]:
    out = []
    for y in sorted(Path(root).rglob("eval.yaml")):
        rel = y.parent.relative_to(root).parts
        if any(p in SKIP_DIRS for p in rel) or "session" in rel or "rollouts" in rel:
            continue
        out.append(y.parent)
    return out


_KIND_CACHE: dict = {}


def kind_of_eval(doc: dict) -> str:
    pol = doc.get("policy") or {}
    raw = pol.get("raw") or (str(DATA_ROOT / str(pol.get("task")) / "policy" / str(pol.get("run_folder")))
                             if pol.get("task") and pol.get("run_folder") else "")
    if raw and raw in _KIND_CACHE:
        return _KIND_CACHE[raw]
    k = None
    run = Path(str(raw)).expanduser()
    if run.is_dir():
        try:
            from sim_deploy.ui import catalog
            rj = catalog._read_json
            cfg, _ = catalog._train_config(run, rj(run / "job.json"), rj(run / "train_meta.json"))
            k = KIND_NAMES.get(catalog.kind_of(cfg))
        except Exception:                                  # noqa: BLE001 - the catalog is another part's code
            k = None
    if k is None:
        k = KIND_NAMES.get(pol.get("kind"), pol.get("kind") or "unknown")
        if k == "diffusion":                               # real import: kind of the policy_deploy station only
            k = "unknown"
    if raw:
        _KIND_CACHE[raw] = k
    return k


def select(root: Path, pattern: str | None, batch: str | None, protocol: str | None) -> list[tuple[Path, dict]]:
    rx = re.compile(pattern) if pattern else None
    out = []
    for d in discover(root):
        try:
            doc, _ = load_eval(d)
        except Exception:                                  # noqa: BLE001 - unreadable eval.yaml: skip it
            continue
        if doc.get("dry_run"):
            continue
        if rx and not rx.search(str(doc.get("tag") or "")):
            continue
        if batch and str(doc.get("batch_id")) != batch:
            continue
        if protocol and not str(doc.get("protocol_sha") or "").startswith(protocol):
            continue
        out.append((d, doc))
    return out


def batch_rollout_filter(batch: str, batches_root: Path | None = None) -> dict | None:
    """{resolved eval dir: set of rollout n that count} from sim_evals/_batches/<batch>/manifest.jsonl
    (sim_deploy/eval/batch.py): a retry run counts only its ``use_rollouts``; those rollouts of the run it retries
    (and of lower retry levels) are replaced, i.e. dropped. None without a manifest (then every rollout counts).
    A retry rollout replaces the original only if the retry eval recorded it."""
    mf = (batches_root or SIM_EVALS / "_batches") / batch / "manifest.jsonl"
    if not mf.is_file():
        return None
    planned, ev = {}, {}
    for ln in mf.read_text().splitlines():
        try:
            m = json.loads(ln)
        except ValueError:
            continue
        if m.get("event") == "planned" and isinstance(m.get("run"), dict):
            planned[m["run"]["key"]] = m["run"]
        elif m.get("event") == "run" and m.get("eval_dir"):
            ev[m["key"]] = str(Path(m["eval_dir"]).resolve())
    recorded: dict = {}
    for d in set(ev.values()):
        try:
            recorded[d] = {int(r.get("n", -1)) for r in load_eval(Path(d))[1] if r.get("outcome") != "not_run"}
        except (OSError, ValueError):
            recorded[d] = set()
    allowed: dict = {}
    chains: dict = {}
    for key, r in planned.items():
        if key in ev:
            chains.setdefault(r.get("retry_of") or key, []).append((key.count("|retry"), key, r))
    for base, chain in chains.items():
        owner: dict = {}                                   # rollout n -> eval dir that counts it
        for _lvl, key, r in sorted(chain, key=lambda x: x[0]):
            d = ev[key]
            use = r.get("use_rollouts")
            ns = recorded.get(d, set()) if use is None else {int(n) for n in use} & recorded.get(d, set())
            for n in ns:
                owner[n] = d
            allowed.setdefault(d, set())
        for n, d in owner.items():
            allowed[d].add(n)
    return allowed


# ----------------------------------------------------------------------------------------------------- rows

def _f(x, nd=4):
    return "" if x is None else (round(x, nd) if isinstance(x, float) else x)


def rollout_rows(root: Path, chosen: list, allowed: dict | None = None) -> tuple[list[dict], int]:
    """(rows, rollouts dropped by the batch manifest)."""
    rows, dropped = [], 0
    for d, doc in chosen:
        try:
            s = score_eval(d, keep_ms=True)
        except Exception as e:                             # noqa: BLE001 - report the folder, go on
            print(f"[evaltool] {d}: not scored ({type(e).__name__}: {e})")
            continue
        m = s["eval"]
        kind = kind_of_eval(doc)
        ckpt = f"{m['task']}/{m['run_folder']}"
        keep = None if allowed is None else allowed.get(str(Path(d).resolve()), set())
        for e in s["rollouts"]:
            if keep is not None and e["n"] not in keep:
                dropped += 1
                continue
            t = e.get("timing") or {}
            rows.append({
                "checkpoint": ckpt, "kind": kind, "look": m["look"] or "none", "tag": m["tag"],
                "batch": m["batch_id"] or "", "protocol": m["protocol_sha"] or "", "created": m["created"],
                "eval_dir": str(Path(d).resolve().relative_to(Path(root).resolve()))
                if Path(d).resolve().is_relative_to(Path(root).resolve()) else str(d),
                "n": e["n"], "seed_key": json.dumps(e["seed_key"]) if e["seed_key"] is not None else "",
                "origin": e.get("origin", "sim"), "source": e["source"] or "",
                "start": e.get("start_group") or "", "start_id": e.get("start_id") or "",
                "held_out": "" if e.get("held_out") is None else int(e["held_out"]),
                "valid": int(e["validity"]["valid"]), "labelled": int(e.get("labelled", True)),
                "outcome_draft": e.get("outcome_draft") or "",
                "invalid_reasons": "; ".join(e["validity"]["reasons"]), "stop_reason": e["stop_reason"] or "",
                "success": "" if e["success"] is None else int(e["success"]),
                "grasp": "" if e["grasp"] is None else int(e["grasp"]),
                "lifted": "" if e["lifted"] is None else int(e["lifted"]),
                "lift_without_grasp": "" if e.get("lift_without_grasp") is None else int(e["lift_without_grasp"]),
                "t_grasp_s": _f(e["t_grasp"], 3), "t_success_s": _f(e["t_success"], 3),
                "max_lift_m": _f(e["max_lift_m"]), "min_z_tcp_m": _f(e["min_z"]["tcp_m"]),
                "min_z_fingers_rig_m": _f(e["min_z"]["fingers_rig_m"]),
                "table_contact_rows": (e.get("table_contacts") or {}).get("rows", ""),
                "table_contact_geoms": " ".join((e.get("table_contacts") or {}).get("geoms", [])),
                "approach_error_m": _f(e["approach_error_m"]), "close_error_m": _f(e["close_error_m"]),
                "first_second_progress_m": _f(e["first_second_progress_m"]),
                "faults": " ".join(f"{k}:{v}" for k, v in sorted(e["faults"].items())),
                "fault_text": e["fault_text"] or "",
                "shortened": _f(e["gate"]["shortened"]), "pulled_back": _f(e["gate"]["pulled_back"]),
                "width_clips": _f(e["gate"]["width_clips"]), "clamps": _f(e["clamps"]),
                "policy_ms_p50": _f(t.get("policy_ms_p50"), 1), "policy_ms_p95": _f(t.get("policy_ms_p95"), 1),
                "cycle_ms_p50": _f(t.get("cycle_ms_p50"), 1),
                "_policy_ms": (e.get("_ms") or {}).get("policy", []), "_cycle_ms": (e.get("_ms") or {}).get("cycle", []),
                "margin_ms_min": _f(t.get("margin_ms_min"), 1),
                "loop_late": _f(e["validity"]["loop_late"]),
                "loop_late_per_s": _f(e["validity"]["loop_late_per_s"], 1),
                "duration_s": _f(e["duration_s"], 2), "video": e["video"] or "",
            })
    return rows, dropped


# ----------------------------------------------------------------------------------------------------- groups

def _w1_reference() -> dict:
    """{checkpoint: {"policy_bf16_8_cm": median, "demo_cm": median, "n": k}} from W1's open-loop summary."""
    try:
        d = json.loads(W1_OPENLOOP_SUMMARY.read_text())
    except (OSError, ValueError):
        return {}
    out = {}
    for rid, r in (d.get("runs") or {}).items():
        ap = r.get("approach_row6") or []
        pol = [a["bf16_8"]["k15_toward_grasp_cm"] for a in ap if a.get("bf16_8")]
        gt = [a["gt"]["k15_toward_grasp_cm"] for a in ap if a.get("gt")]
        if pol or gt:
            out[rid] = {"policy_bf16_8_cm": float(np.median(pol)) if pol else None,
                        "demo_cm": float(np.median(gt)) if gt else None, "n": len(pol)}
    return out


def real_timing(root: Path = REAL_ROOT) -> dict:
    """{checkpoint: {policy_ms_p50, policy_ms_p90, cycle_ms_p50, cycle_ms_p90, chunks, rollouts, trials, source}}: the
    real arm's sent chunks pooled per checkpoint from sim_evals/_real/<task>/<run>/<session>/rollouts/*/steps.jsonl;
    ``rollouts`` = steps.jsonl files, ``trials`` = distinct real trials among them (rollout.json trial.key; a trial
    can have several rollouts). Checkpoints without any get REAL_TIMING_FALLBACK (p50 only)."""
    pol: dict = {}
    cyc: dict = {}
    rollouts: dict = {}
    trials: dict = {}
    for f in sorted(Path(root).glob("*/*/*/rollouts/*/steps.jsonl")):
        task, run = f.parts[-6], f.parts[-5]
        ck = f"{task}/{run}"
        sent = [s for s in _jsonl(f) if s.get("sent")]
        pol.setdefault(ck, []).extend(s["policy_ms"] for s in sent if s.get("policy_ms") is not None)
        cyc.setdefault(ck, []).extend(s["cycle_ms"] for s in sent if s.get("cycle_ms") is not None)
        rollouts[ck] = rollouts.get(ck, 0) + 1
        rj = _json(f.parent / "rollout.json") or {}
        tkey = (rj.get("trial") or {}).get("key") or str(f.parents[2])   # no trial key: its session folder
        trials.setdefault(ck, set()).add(tkey)
    out = {ck: {"policy_ms_p50": _q(pol[ck], 50), "policy_ms_p90": _q(pol[ck], 90),
                "cycle_ms_p50": _q(cyc.get(ck, []), 50), "cycle_ms_p90": _q(cyc.get(ck, []), 90),
                "chunks": len(pol[ck]), "rollouts": rollouts[ck], "trials": len(trials[ck]), "source": "sim_evals/_real steps.jsonl"}
           for ck in pol if pol[ck]}
    for ck, v in REAL_TIMING_FALLBACK.items():
        if ck not in out:
            out[ck] = {**v, "policy_ms_p90": None, "cycle_ms_p90": None, "chunks": None, "rollouts": None, "trials": None,
                       "source": "documented value (report.py REAL_TIMING_FALLBACK)"}
    return out


def _rate_of(rs: list, pred=None) -> dict:
    v = [r for r in rs if r["valid"] and r["labelled"] and r["success"] != ""]
    k = sum(1 for r in v if (pred(r) if pred else int(r["success"])))
    lo, hi = wilson(k, len(v))
    return {"k": k, "n": len(v), "rate": round(k / len(v), 3) if v else "", "lo": round(lo, 3), "hi": round(hi, 3)}


def _cell(x: dict) -> str:
    return f"{x['k']}/{x['n']}" + (f" ({_pct(x['rate'])})" if x["n"] else "")


def _median(xs):
    v = [x for x in xs if x not in ("", None)]
    return float(np.median(v)) if v else None


def group_rows(rows: list, keys: list, w1: dict | None = None) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for gk, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        valid = [r for r in rs if r["valid"] and r["labelled"] and r["success"] != ""]
        n, k = len(valid), sum(int(r["success"]) for r in valid)
        unl = sum(1 for r in rs if r["valid"] and not r["labelled"])
        met = [r for r in rs if r["valid"]]               # metrics: every valid rollout, labelled or not
        lo, hi = wilson(k, n)
        g = sum(int(r["grasp"] or 0) for r in valid)
        faults: dict = {}
        for r in met:
            for kv in r["faults"].split():
                a, b = kv.split(":")
                faults[a] = faults.get(a, 0) + int(b)
        row = dict(zip(keys, gk))
        row.update({
            "rollouts": len(rs), "valid": len(met), "rated": n, "invalid": sum(1 for r in rs if not r["valid"]),
            "unlabelled": unl,
            "successes": k,
            "success_rate": round(k / n, 3) if n else "", "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
            "grasp_rate": round(g / n, 3) if n else "",
            "lift_without_grasp": sum(1 for r in valid if r["lift_without_grasp"] == 1),
            "t_success_p50_s": _f(_median([r["t_success_s"] for r in met]), 2),
            "max_lift_p50_m": _f(_median([r["max_lift_m"] for r in met])),
            "min_z_tcp_p50_m": _f(_median([r["min_z_tcp_m"] for r in met])),
            "approach_error_p50_m": _f(_median([r["approach_error_m"] for r in met])),
            "close_error_p50_m": _f(_median([r["close_error_m"] for r in met])),
            "first_second_progress_p50_m": _f(_median([r["first_second_progress_m"] for r in met])),
            "table_contact_rollouts": sum(1 for r in met if r["table_contact_rows"] not in ("", 0)),
            "faults": " ".join(f"{a}:{b}" for a, b in sorted(faults.items())),
            "clamps": sum(int(r["clamps"] or 0) for r in met),
            "policy_ms_p50": _f(_median([r["policy_ms_p50"] for r in met]), 1),
            "evals": len({r["eval_dir"] for r in rs}),
        })
        for s in C.SUCCESS_WITHIN_S:                        # success reached within s of the rollout start
            x = _rate_of(rs, lambda r, s=s: int(r["success"]) and r["t_success_s"] != ""
                         and float(r["t_success_s"]) <= s + C.HOLD_TOL_S)
            row[f"success_{s:g}s"], row[f"success_{s:g}s_rate"] = x["k"], x["rate"]
        for g, _name in START_GROUPS:                       # start-set split (empty for random placements)
            x = _rate_of([r for r in rs if r.get("start") == g])
            row[f"{g}_successes"], row[f"{g}_rated"], row[f"{g}_rate"] = x["k"], x["n"], x["rate"]
        pm = [v for r in met for v in r.get("_policy_ms") or []]
        cm = [v for r in met for v in r.get("_cycle_ms") or []]
        row.update({"sim_policy_ms_p50": _f(_q(pm, 50), 1), "sim_policy_ms_p90": _f(_q(pm, 90), 1),
                    "sim_cycle_ms_p50": _f(_q(cm, 50), 1), "sim_cycle_ms_p90": _f(_q(cm, 90), 1),
                    "sim_chunks": len(pm)})
        if w1 is not None and "checkpoint" in keys:
            ref = w1.get(row["checkpoint"]) or {}
            row["w1_first_chunk_toward_grasp_cm"] = _f(ref.get("policy_bf16_8_cm"), 2)
            row["w1_demo_first_chunk_toward_grasp_cm"] = _f(ref.get("demo_cm"), 2)
        out.append(row)
    return out


def _binom_two_sided(b: int, c: int) -> float | None:
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_looks(rows: list, a: str = "robot", b: str = "train") -> list[dict]:
    """Per checkpoint: valid rollouts of look a and look b with the same seed_key (the newest rollout per
    (checkpoint, look, seed_key) when repeated); exact McNemar p on the discordant pairs."""
    newest: dict = {}
    for r in rows:
        if not r["valid"] or not r["labelled"] or r["success"] == "" or not r["seed_key"] or r["look"] not in (a, b):
            continue
        key = (r["checkpoint"], r["look"], r["seed_key"])
        if key not in newest or str(r["created"]) > str(newest[key]["created"]):
            newest[key] = r
    out = []
    for ck in sorted({k[0] for k in newest}):
        seeds = sorted({k[2] for k in newest if k[0] == ck})
        pairs = [(newest[(ck, a, s)], newest[(ck, b, s)]) for s in seeds if (ck, a, s) in newest and (ck, b, s) in newest]
        if not pairs:
            continue
        both = sum(1 for x, y in pairs if x["success"] and y["success"])
        only_a = sum(1 for x, y in pairs if x["success"] and not y["success"])
        only_b = sum(1 for x, y in pairs if y["success"] and not x["success"])
        out.append({"checkpoint": ck, "pairs": len(pairs), f"{a}_successes": both + only_a,
                    f"{b}_successes": both + only_b, "both": both, f"{a}_only": only_a, f"{b}_only": only_b,
                    "neither": len(pairs) - both - only_a - only_b,
                    "mcnemar_exact_p": _f(_binom_two_sided(only_a, only_b), 3)})
    return out


# ----------------------------------------------------------------------------------------------------- writing

def _csv(path: Path, rows: list, fields: list | None = None) -> None:
    fields = fields or ([k for k in rows[0].keys() if not k.startswith("_")] if rows else [])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _pct(x) -> str:
    return "-" if x in ("", None) else f"{100 * float(x):.0f} %"


def _md_table(rows: list, cols: list) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _rate_cell(r) -> str:
    return (f"{r['successes']}/{r['rated']} ({_pct(r['success_rate'])}, CI {_pct(r['ci95_lo'])}-"
            f"{_pct(r['ci95_hi'])})")


CK_COLS_MD = ["success (Wilson 95 %)", "invalid", "unlabelled", "grasp", "lift w/o grasp", "t_success p50 s",
              "approach err p50 m",
              "close err p50 m", "1st-s progress p50 m", "W1 first chunk cm (policy / demo)", "faults", "clamps"]


def _ck_md_rows(groups: list, keys: list) -> list:
    out = []
    for g in groups:
        r = {k: g[k] for k in keys}
        r.update({"success (Wilson 95 %)": _rate_cell(g), "invalid": g["invalid"], "unlabelled": g["unlabelled"], "grasp": _pct(g["grasp_rate"]),
                  "lift w/o grasp": g["lift_without_grasp"],
                  "t_success p50 s": g["t_success_p50_s"], "approach err p50 m": g["approach_error_p50_m"],
                  "close err p50 m": g["close_error_p50_m"],
                  "1st-s progress p50 m": g["first_second_progress_p50_m"],
                  "W1 first chunk cm (policy / demo)":
                      f"{g.get('w1_first_chunk_toward_grasp_cm', '')} / {g.get('w1_demo_first_chunk_toward_grasp_cm', '')}"
                      if "w1_first_chunk_toward_grasp_cm" in g else "",
                  "faults": g["faults"], "clamps": g["clamps"]})
        out.append(r)
    return out


def _html_table(rows: list, cols: list, ci: bool = False) -> str:
    e = html.escape
    head = "".join(f"<th>{e(c)}</th>" for c in cols) + ("<th>95 % CI</th>" if ci else "")
    body = []
    for r in rows:
        cells = "".join(f"<td>{e(str(r.get(c, '')))}</td>" for c in cols)
        if ci:
            lo, hi, p = r.get("_lo"), r.get("_hi"), r.get("_p")
            bar = ""
            if lo is not None:
                bar = (f'<div class="ci"><span style="left:{100 * lo:.1f}%;width:{max(0.5, 100 * (hi - lo)):.1f}%">'
                       f'</span>' + (f'<i style="left:{100 * p:.1f}%"></i>' if p not in ("", None) else "") + "</div>")
            cells += f"<td>{bar}</td>"
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _with_ci(md_rows: list, groups: list) -> list:
    return [dict(m, _lo=g["ci95_lo"], _hi=g["ci95_hi"], _p=g["success_rate"]) if g["rated"] else dict(m)
            for m, g in zip(md_rows, groups)]


def build_report(root: Path, pattern: str | None, batch: str | None, protocol: str | None, group: list,
                 out: Path) -> dict:
    root, out = Path(root), Path(out)
    bad = [k for k in group if k not in GROUP_KEYS]
    if bad:
        raise ValueError(f"--group {bad}: expected some of {', '.join(GROUP_KEYS)}")
    chosen = select(root, pattern, batch, protocol)
    allowed = batch_rollout_filter(batch) if batch else None
    rows, dropped = rollout_rows(root, chosen, allowed)
    w1 = _w1_reference()
    groups = group_rows(rows, group, w1)
    kxl = group_rows(rows, ["kind", "look"])
    kinds = group_rows(rows, ["kind"])
    pairs = paired_looks(rows)
    real_t = real_timing()
    # start split + within-time success, per group
    has_sets = any(r.get("start") for r in rows)
    st_cols = list(group) + ["success (60 s)"] + [f"<= {s:g} s" for s in C.SUCCESS_WITHIN_S] + (
        [name for _g, name in START_GROUPS] if has_sets else [])
    st_rows = []
    for g in groups:
        x = {k: g[k] for k in group}
        x["success (60 s)"] = f"{g['successes']}/{g['rated']} ({_pct(g['success_rate'])})"
        for s in C.SUCCESS_WITHIN_S:
            x[f"<= {s:g} s"] = f"{g[f'success_{s:g}s']}/{g['rated']} ({_pct(g[f'success_{s:g}s_rate'])})"
        for gk, name in START_GROUPS:
            x[name] = (f"{g[f'{gk}_successes']}/{g[f'{gk}_rated']}" + (f" ({_pct(g[f'{gk}_rate'])})"
                                                                       if g[f"{gk}_rated"] else ""))
        st_rows.append(x)
    # timing vs the real arm, per checkpoint (pooled over the selection)
    tm_cols = ["checkpoint", "sim chunks", "sim policy ms p50 / p90", "sim cycle ms p50 / p90",
               "real policy ms p50 / p90", "real cycle ms p50 / p90", "real source"]
    tm_rows = []
    for g in group_rows(rows, ["checkpoint"]):
        rt = real_t.get(g["checkpoint"]) or {}

        def pp(a, b):
            return f"{_f(a, 0) if a is not None else '-'} / {_f(b, 0) if b is not None else '-'}"
        tm_rows.append({"checkpoint": g["checkpoint"], "sim chunks": g["sim_chunks"],
                        "sim policy ms p50 / p90": pp(g["sim_policy_ms_p50"] or None, g["sim_policy_ms_p90"] or None),
                        "sim cycle ms p50 / p90": pp(g["sim_cycle_ms_p50"] or None, g["sim_cycle_ms_p90"] or None),
                        "real policy ms p50 / p90": pp(rt.get("policy_ms_p50"), rt.get("policy_ms_p90")) if rt else "",
                        "real cycle ms p50 / p90": pp(rt.get("cycle_ms_p50"), rt.get("cycle_ms_p90")) if rt else "",
                        "real source": (f"{rt['source']}" + (f", {rt['trials']} trials ({rt['rollouts']} rollouts), "
                                                             f"{rt['chunks']} chunks" if rt.get("trials") else "")) if rt else "no real trial"})
    out.mkdir(parents=True, exist_ok=True)
    _csv(out / "rollouts.csv", rows)
    _csv(out / "checkpoints.csv", groups)
    _csv(out / "kind_x_look.csv", kxl)

    n_inv = sum(1 for r in rows if not r["valid"])
    n_unl = sum(1 for r in rows if r["valid"] and not r["labelled"])
    inv_reasons: dict = {}
    for r in rows:
        for x in filter(None, r["invalid_reasons"].split("; ")):
            key = x.split(" ")[0] if not x.startswith("stop_reason") else x
            inv_reasons[key] = inv_reasons.get(key, 0) + 1
    sources: dict = {}
    for r in rows:
        sources[r["source"] or "none"] = sources.get(r["source"] or "none", 0) + 1
    created = datetime.now().isoformat(timespec="seconds")
    sel = {"root": str(root), "select": pattern, "batch": batch, "protocol": protocol, "group": group}
    if batch:
        sel["batch_manifest"] = ("read: " + f"{dropped} rollouts not counted (retry extras / replaced)"
                                 if allowed is not None else "none found: every rollout counts")
    bp_plain = [g for g in kinds if g["kind"] in ("BP", "plain")]
    pair_cols = ["checkpoint", "pairs", "robot_successes", "train_successes", "both", "robot_only", "train_only",
                 "neither", "mcnemar_exact_p"]
    ck_md = _ck_md_rows(groups, group)
    kxl_md = _ck_md_rows(kxl, ["kind", "look"])
    kind_md = _ck_md_rows(bp_plain, ["kind"])
    cols_ck = group + CK_COLS_MD
    # successes without a two-sided grasp in the hold window (score.json lift_without_grasp): a human checks each one
    # (one-finger carry or a physics artifact)
    lwg = [r for r in rows if r["lift_without_grasp"] == 1]
    lwg_evals: dict = {}
    for r in rows:
        if r["success"] == 1:
            x = lwg_evals.setdefault(r["eval_dir"], {"eval_dir": r["eval_dir"], "checkpoint": r["checkpoint"],
                                                     "look": r["look"], "successes": 0, "lift_without_grasp": 0})
            x["successes"] += 1
            x["lift_without_grasp"] += r["lift_without_grasp"] == 1
    lwg_evals_rows = [x for x in lwg_evals.values() if x["lift_without_grasp"]]
    lwg_eval_cols = ["eval_dir", "checkpoint", "look", "successes", "lift_without_grasp"]
    lwg_cols = ["eval_dir", "n", "valid", "t_success_s", "max_lift_m", "grasp", "t_grasp_s", "video"]
    lwg_md = ["## Lift without grasp (check by hand)", "",
              f"{len(lwg)} successful rollouts in which no row of the success hold window has both fingers on the "
              f"eraser: {C.LIFT_WITHOUT_GRASP_RULE}. Watch each one (one-finger carry or a physics artifact). "
              "`grasp` = a two-sided grasp at any time of the rollout.", "",
              _md_table(lwg_evals_rows, lwg_eval_cols), _md_table(lwg, lwg_cols)]

    md = [f"# Sim eval report: {out.name}", "",
          f"{created} · selection {json.dumps(sel)} · {len(chosen)} eval folders, {len(rows)} rollouts "
          f"({len(rows) - n_inv - n_unl} rated, {n_unl} unlabelled real trials, {n_inv} invalid: "
          f"{json.dumps(inv_reasons)}) · scored from "
          f"{json.dumps(sources)}", "",
          f"Rules: success = the eraser centre >= {C.LIFT_M * 100:.0f} cm above its settled height and its lowest "
          f"corner >= {C.CLEAR_M * 100:.0f} cm above the table, held {C.HOLD_S:.1f} s, grasp not required (score "
          f"{C.SCORE_SCHEMA}; folders with only trace.json: the lift alone; real trials: the operator's label, the "
          f"eraser lifted clear of the table); grasp = both fingers on the eraser in one row (reported); invalid = sim: stop by a person / an "
          f"error, a sim-lag fault, loop_late > {C.LOOP_LATE_MAX_PER_S:.0f}/s, no waypoint accepted, or started after a "
          f"GPU fault (kernel Xid); real: stop_reason error "
          f"(excluded from every rate, counted); unlabelled real trials are counted, not rated. Metrics: sim_deploy/evaltool/README.md.",
          "", f"## By {', '.join(group)}", "", _md_table(ck_md, cols_ck), *lwg_md,
          "## BP vs plain (pooled over the selection)", "",
          _md_table(kind_md, ["kind"] + CK_COLS_MD[:4]),
          "## kind x look", "", _md_table(kxl_md, ["kind", "look"] + CK_COLS_MD[:4]),
          "## robot vs train, paired on seed_key", "", _md_table(pairs, pair_cols),
          "## Success within time and by start", "",
          f"Success reached within {' / '.join(f'{s:g}' for s in C.SUCCESS_WITHIN_S)} s of the rollout start (t_success) "
          "next to the timeout's rate. Start split (start sets only, `eval/start_set.py`): the shared demo starts the "
          "checkpoint trained on, its one held-out (val) demo, and the eraser poses fitted to the real-arm trials.", "",
          _md_table(st_rows, st_cols),
          "## Timing vs the real arm", "",
          "Sent chunks pooled per checkpoint (steps.jsonl): sim over the selection; real = the 2026-11-20 arm trials "
          "imported under sim_evals/_real, else the documented p50 (report.py REAL_TIMING_FALLBACK). The sim's cycle "
          "(~200 ms) is longer than the real arm's (~110-130 ms): the policy replans less often in the sim (known gap, "
          "reported, not fixed: user decision 2026-11-22).", "",
          _md_table(tm_rows, tm_cols),
          "W1 first chunk = median over W1's row-6 samples of the k = 15 TCP displacement toward the demo's grasp "
          "point (bf16/8 policy; demo = the recorded action), cm: compare with 1st-s progress (m, toward the "
          f"eraser, first {C.FIRST_SECOND_S:.0f} s in the sim). Source: {W1_OPENLOOP_SUMMARY}.", ""]
    (out / "report.md").write_text("\n".join(md))

    e = html.escape
    css = """:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--acc:#2f6fdf;--accbg:#d6e3fa}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--acc:#7aa7ff;--accbg:#27344d}}
:root[data-theme="dark"]{--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--acc:#7aa7ff;--accbg:#27344d}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:24px auto;max-width:1300px;padding:0 16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 6px}.mut{color:var(--mut)}
.wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;margin:8px 0}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--mut);font-weight:500}.ci{position:relative;width:160px;height:10px;background:var(--line);border-radius:5px}
.ci span{position:absolute;top:0;height:10px;background:var(--accbg);border:1px solid var(--acc);border-radius:5px;box-sizing:border-box}
.ci i{position:absolute;top:-2px;width:2px;height:14px;background:var(--acc);margin-left:-1px}"""
    ro_cols = ["checkpoint", "kind", "look", "origin", "tag", "n", "seed_key", "start", "held_out", "start_id",
               "valid", "labelled", "stop_reason",
               "outcome_draft", "success", "grasp", "lift_without_grasp",
               "t_success_s", "max_lift_m", "min_z_tcp_m", "approach_error_m", "close_error_m",
               "first_second_progress_m", "faults", "clamps", "loop_late", "policy_ms_p50", "invalid_reasons",
               "eval_dir"]
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sim eval report</title><style>{css}</style></head><body>
<h1>Sim eval report: {e(out.name)}</h1>
<div class="mut">{e(created)} · selection {e(json.dumps(sel))} · {len(chosen)} eval folders · {len(rows)} rollouts,
{len(rows) - n_inv - n_unl} rated, {n_unl} unlabelled, {n_inv} invalid {e(json.dumps(inv_reasons))} · scored from {e(json.dumps(sources))}</div>
<p class="mut">{e(md[4])}</p>
<h2>By {e(', '.join(group))}</h2><div class="wrap">{_html_table(_with_ci(ck_md, groups), cols_ck, ci=True)}</div>
<h2>Lift without grasp (check by hand)</h2><p class="mut">{e(lwg_md[2])}</p>
<div class="wrap">{_html_table(lwg_evals_rows, lwg_eval_cols)}</div><div class="wrap">{_html_table(lwg, lwg_cols)}</div>
<h2>BP vs plain (pooled)</h2><div class="wrap">{_html_table(_with_ci(kind_md, bp_plain), ['kind'] + CK_COLS_MD[:4], ci=True)}</div>
<h2>kind x look</h2><div class="wrap">{_html_table(_with_ci(kxl_md, kxl), ['kind', 'look'] + CK_COLS_MD[:4], ci=True)}</div>
<h2>robot vs train, paired on seed_key</h2><div class="wrap">{_html_table(pairs, pair_cols)}</div>
<h2>Success within time and by start</h2><div class="wrap">{_html_table(st_rows, st_cols)}</div>
<h2>Timing vs the real arm</h2><div class="wrap">{_html_table(tm_rows, tm_cols)}</div>
<p class="mut">{e(md[-2])}</p>
<h2>Rollouts</h2><div class="wrap">{_html_table(rows, ro_cols)}</div>
</body></html>
"""
    (out / "report.html").write_text(page)
    return {"evals": len(chosen), "rollouts": len(rows), "invalid": n_inv, "dropped": dropped,
            "lift_without_grasp": len(lwg), "out": str(out)}
