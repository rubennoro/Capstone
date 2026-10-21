"""Pure gate evaluation: ``evaluate(criteria, metrics, stage) -> {stage, criteria_version, gates, verdict}``.

Why: every stage reports the same way: a flat metrics dict is compared against ``run/criteria/<id>.yaml`` (section
``stage<N>``). No I/O, no imports beyond the stdlib, so it can be tested without any simulator.

Rules (keys of ``criteria[<section>]``):
  * ``<name>_min``: value >= threshold; ``<name>_max``: value <= threshold; a boolean threshold: value must equal it
    (``True`` means the metric must be true). The metric looked up is ``<name>`` without the suffix, e.g.
    ``collisions_max`` -> metrics["collisions"]. A key without a suffix is looked up as is.
  * ``<name>_at_least``: an evidence requirement (``>=``) whose shortfall means "not enough data" (``pass: None``,
    INCOMPLETE), never a FAIL, e.g. ``n_pairs_at_least: 50``.
  * A threshold dict selects by ``metrics["protocol"]`` (``{v1: 30, v2: 5}``, protocol "v1"/"v2"); a dict of ``k4/k8/k15``
    keys (position error at chunk step k) fans out into one gate per key, metric ``<name>_<key>`` (e.g.
    ``pred_vs_demo_pos_err_mm_p95_max: {k4: 15}`` -> gate ``pred_vs_demo_pos_err_mm_p95_k4_max`` on metric
    ``pred_vs_demo_pos_err_mm_p95_k4``).
  * Missing (None / absent / NaN) metric: the gate is ``pass: None`` (INCOMPLETE), never a pass.
Verdict: FAIL if any gate fails, else INCOMPLETE if any gate is incomplete (or there are no gates), else PASS.
Metrics that are not asked for by the criteria are ignored; criteria keys the stage does not measure at all
(e.g. stage3 keys while evaluating stage 2) are never looked at, because only the ``stage`` section is read.
"""

from __future__ import annotations

import math
import re

STAGE_SECTION = {"1": "stage1", "1b": "stage1b", "2a": "stage2", "2b": "stage2", "2": "stage2", "3": "stage3"}
# criteria keys of the shared `stage2` section that belong to one sub-stage only (the other sub-stage's keys are skipped)
STAGE_KEYS = {
    "2a": {"chunk_reachable_fraction_min", "chunk_joint_margin_deg_min", "pred_vs_demo_pos_err_mm_p95_max"},
    "2b": {"live_vs_offline_pos_err_mm_p50_max", "live_vs_offline_pos_err_mm_p95_max", "n_pairs_at_least",
           "frame_selection_errors_max", "cam_stamp_err_ms_p95_max"},
}
K_KEYS = re.compile(r"^k\d+$")


def _missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _split(key: str) -> tuple[str, str]:
    for suf, op in (("_at_least", ">=*"), ("_min", ">="), ("_max", "<=")):
        if key.endswith(suf):
            return key[: -len(suf)], op
    return key, "=="


def _gate(name: str, metric: str, op: str, thr, metrics: dict) -> dict:
    v = metrics.get(metric)
    if _missing(v):
        return {"value": None, "threshold": thr, "op": op, "pass": None, "metric": metric}
    if op == ">=*":         # evidence requirement: below the threshold = insufficient data, not a failure
        if float(v) >= float(thr):
            return {"value": v if isinstance(v, (bool, str, int)) else float(v), "threshold": thr, "op": ">=", "pass": True, "metric": metric}
        return {"value": float(v), "threshold": thr, "op": ">=", "pass": None, "metric": metric,
                "note": "insufficient data (INCOMPLETE, not FAIL)"}
    if isinstance(thr, bool):
        ok = bool(v) == thr
        op = "is"
    elif op == ">=":
        ok = float(v) >= float(thr)
    elif op == "<=":
        ok = float(v) <= float(thr)
    else:
        ok = float(v) == float(thr)
    return {"value": v if isinstance(v, (bool, str, int)) else float(v), "threshold": thr, "op": op, "pass": bool(ok),
            "metric": metric}


def evaluate(criteria: dict, metrics: dict, stage: str) -> dict:
    stage = str(stage)
    sec = STAGE_SECTION.get(stage)
    if sec is None:
        raise ValueError(f"unknown stage {stage!r} (known: {', '.join(STAGE_SECTION)})")
    spec = dict(criteria.get(sec) or {})
    if stage in STAGE_KEYS:
        spec = {k: v for k, v in spec.items() if k in STAGE_KEYS[stage]}
    gates: dict[str, dict] = {}
    for key, thr in spec.items():
        base, op = _split(key)
        if isinstance(thr, dict) and thr and all(K_KEYS.match(str(k)) for k in thr):
            for kk, t in thr.items():
                gates[f"{sec}.{base}_{kk}{key[len(base):]}"] = _gate(key, f"{base}_{kk}", op, t, metrics)
        elif isinstance(thr, dict):
            proto = metrics.get("protocol")
            if proto is None or proto not in thr:
                gates[f"{sec}.{key}"] = {"value": None, "threshold": thr, "op": op, "pass": None, "metric": base,
                                         "note": f"no threshold for protocol {proto!r}"}
            else:
                gates[f"{sec}.{key}"] = {**_gate(key, base, op, thr[proto], metrics), "protocol": proto}
        else:
            gates[f"{sec}.{key}"] = _gate(key, base, op, thr, metrics)
    passes = [g["pass"] for g in gates.values()]
    if any(p is False for p in passes):
        verdict = "FAIL"
    elif not passes or any(p is None for p in passes):
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    return {"stage": stage, "criteria_version": str(criteria.get("schema", "run.criteria/v1")) + ":" + str(criteria.get("id", "")),
            "gates": gates, "verdict": verdict}
