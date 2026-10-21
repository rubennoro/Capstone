"""Sweeps: the cartesian grid of a spec (``run/sweeps/<id>.yaml``), one eval run per point, with resume.

Spec keys: ``axes`` (a list; each ``{group: <run or station group>, ids: [...]}`` or ``{set: <group.path>, values: [...]}``),
``stage`` (1, 2a, 2b, 3 or all), ``max_runs`` (cap on NEW runs per invocation), ``seed``, ``n_episodes`` (Stage 3: episodes per
pose; Stage 2a/2b: demos), ``n_poses`` (Stage 3), ``resume``.

Expansion is deterministic and row-major (the last axis varies fastest); point i gets seed ``seed + i``. Each point is resolved
with ``runner.resolve(args, run_overrides, extra_sets)``, so a ``set`` axis changes the run dir hash and a ``group`` axis the
station fingerprint: points never collide. ``OUT/sweep_index.json`` maps point -> run dir and is rewritten after every point,
so a process killed between points resumes where it stopped: a point whose run dir holds a ``report.json`` with a verdict other
than INCOMPLETE is skipped. ``OUT/summary.html`` compares the points column by column.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import time
from pathlib import Path

import yaml

from franka_sim.eval import report as rpt
from franka_sim.eval import runner as rn

SWEEP_DIR = rn.REPO_ROOT / "controlSW" / "configs" / "run" / "sweeps"


def load_spec(ref: str) -> dict:
    p = Path(ref)
    if not p.is_file():
        p = SWEEP_DIR / f"{ref}.yaml"
    if not p.is_file():
        rn.usage_error(f"--sweep {ref!r}: not a file and not run/sweeps/{ref}.yaml")
    spec = yaml.safe_load(p.read_text())
    spec.setdefault("id", p.stem)
    return spec


def axis_label(ax: dict) -> str:
    return ax["group"] if "group" in ax else ax["set"]


def axis_values(ax: dict) -> list:
    if "group" in ax:
        return list(ax["ids"])
    if "set" in ax:
        return list(ax["values"])
    raise ValueError(f"sweep axis needs group+ids or set+values: {ax}")


def expand(spec: dict) -> list:
    """[{i, seed, axes: {label: value}, run_overrides: {group: id}, extra_sets: ['path=value']}] in row-major order."""
    axes = spec.get("axes") or []
    seed0 = int(spec.get("seed", 0))
    pts = []
    if not axes:
        return pts
    for i, combo in enumerate(itertools.product(*[axis_values(a) for a in axes])):
        ro, es, lab = {}, [], {}
        for ax, v in zip(axes, combo):
            lab[axis_label(ax)] = v
            if "group" in ax:
                ro[ax["group"]] = v
            else:
                es.append(f"{ax['set']}={v}")
        pts.append({"i": i, "seed": seed0 + i, "axes": lab, "run_overrides": ro, "extra_sets": es})
    return pts


def _args_for(args, spec: dict, pt: dict):
    a = copy.copy(args)
    a.seed = pt["seed"]
    if spec.get("n_poses") and not getattr(a, "poses", None):
        a.poses = int(spec["n_poses"])
    return a


def _stages(args, spec: dict) -> list:
    st = str(spec.get("stage", args.stage))
    return list(rn.STAGES) if st == "all" else [st]


def _write_index(out: Path, idx: dict):
    tmp = out / "sweep_index.json.tmp"
    tmp.write_text(json.dumps(idx, indent=1, default=str) + "\n")
    os.replace(tmp, out / "sweep_index.json")


def finished_verdict(run_dir: Path) -> str | None:
    """The verdict in ``run_dir/report.json`` (None when absent or unreadable)."""
    p = Path(run_dir) / "report.json"
    try:
        return json.loads(p.read_text()).get("verdict")
    except (OSError, ValueError):
        return None


def run_sweep(args, stage_fn=None) -> int:
    """Run (or resume) the sweep named by ``args.sweep``; returns the eval exit code over every point's verdict.
    ``stage_fn(R, stage, results) -> report`` is injectable for tests (default ``runner.run_one_stage``)."""
    stage_fn = stage_fn or rn.run_one_stage
    spec = load_spec(args.sweep)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = _stages(args, spec)
    max_runs = args.max_runs if getattr(args, "max_runs", None) is not None else spec.get("max_runs")
    resume = bool(spec.get("resume", True))
    pts = expand(spec)
    n_ep = spec.get("n_episodes")
    idx = {"schema": "sim_sweep_index/v1", "sweep": spec["id"], "stages": stages, "axes": [axis_label(a) for a in spec.get("axes", [])],
           "points": []}
    new_runs, verdicts = 0, []
    for pt in pts:
        a = _args_for(args, spec, pt)
        if n_ep:
            if stages == ["3"] and not getattr(a, "episodes_per_pose", None):
                a.episodes_per_pose = int(n_ep)
            elif stages != ["3"] and not getattr(a, "episodes", None) and not getattr(a, "demos", None):
                a.episodes = int(n_ep)
        R = rn.resolve(a, pt["run_overrides"], pt["extra_sets"])
        entry = {"i": pt["i"], "seed": pt["seed"], "axes": pt["axes"], "stages": {}}
        idx["points"].append(entry)
        for st in stages:
            d = R.run_dir(st)
            entry["stages"][st] = {"run_dir": d.name}
            v = finished_verdict(d) if resume else None
            if v is not None and v != "INCOMPLETE":
                entry["stages"][st].update(verdict=v, skipped=True)
                verdicts.append(v)
                continue
            if max_runs is not None and new_runs >= int(max_runs):
                entry["stages"][st]["verdict"] = v          # pending (INCOMPLETE earlier or never run)
                continue
            rep = stage_fn(R, st, {})
            new_runs += 1
            entry["stages"][st].update(verdict=rep["verdict"], skipped=False)
            verdicts.append(rep["verdict"])
            _write_index(out, idx)
            rpt.write_summary(out)
    _write_index(out, idx)
    rpt.write_summary(out)
    pending = sum(1 for p in idx["points"] for s in p["stages"].values() if s.get("verdict") in (None, "INCOMPLETE") and not s.get("skipped"))
    print(f"[sweep] {spec['id']}: {len(pts)} point(s), {new_runs} new run(s), {pending} pending/incomplete; index {out / 'sweep_index.json'}", flush=True)
    if pending:                 # points left unrun (max_runs) or INCOMPLETE: the sweep is not finished, whatever the rest say
        verdicts.append("INCOMPLETE")
    return rn.exit_code(verdicts)
