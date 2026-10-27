"""Eval runner: resolve the profile's station + policy, run stages, write run dirs, criteria gates and the summary.

Run dir: ``OUT/<stage>_<policy>_<fingerprint[:8]>_<sha8(user sets)>/`` with station_resolved.yaml, criteria.yaml,
metrics.json, report.json, report.html and the stage artefacts. ``OUT/summary.{json,html}`` has one row per run dir.
The runner (and the sim side) runs in the real-env interpreter (mujoco); torch work is a subprocess with the BP python
(``tools.predict_demos``); the BP side never imports mujoco and this side never imports torch.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from franka_sim.eval import criteria as crit
from franka_sim.eval import report as rpt

# franka_sim.model appends franka_deploy to sys.path as well; the runner needs it before importing the model.
SIM_PKG = Path(__file__).resolve().parents[2]                   # controlSW/franka_sim
DEPLOY_PKG = SIM_PKG.parent / "franka_deploy"
for _p in (DEPLOY_PKG,):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

from franka_deploy.station import StationError, load_station  # noqa: E402
from franka_deploy.launch import launcher as L  # noqa: E402
from franka_deploy.paths import REPO_ROOT  # noqa: E402

STAGES = ("1", "1b", "2a", "2b")                                # what "--stage all" runs (1b = A8's full-stack campaign)
ALL_STAGES = STAGES + ("3",)                                    # Stage 3 (closed loop) is always asked for explicitly
EXIT_PASS, EXIT_INCOMPLETE, EXIT_USAGE, EXIT_FAIL = 0, 1, 2, 3   # 2 = usage / config error (as argparse), never a verdict
STATION_ID_GROUPS = ("robot", "tooling", "sensor", "scene", "host")
STATION_GROUPS = ("robot", "tooling", "sensor", "scene", "host", "criteria", "sim")   # groups Stage 1 / the sim model load
DEFAULT_DEPLOY = DEPLOY_PKG / "configs" / "deploy.yaml"


def usage_error(msg: str):
    """A usage / configuration error: message on stderr, exit code 2 (1 means INCOMPLETE, 3 FAIL)."""
    import sys as _sys

    print(f"eval: error: {msg}", file=_sys.stderr)
    raise SystemExit(EXIT_USAGE)


class StageIncomplete(Exception):
    """A stage could not produce its metrics (launcher failed, sender died, ...): the verdict is INCOMPLETE."""


@dataclass
class Run:
    """Everything a stage needs, resolved once per (profile, policy, sets)."""
    args: object
    cfg: dict
    station: object
    policy: dict
    ids: dict
    run_ids: dict
    sets: list
    user_sets: list
    out: Path
    seed: int = 0
    notes: list = field(default_factory=list)
    tag: str = ""            # extra run-dir identity that is not a station set (e.g. the Stage 3 stub scenario)

    @property
    def profile(self) -> str:
        return self.cfg["profile"]

    @property
    def python_bp(self) -> str:
        return self.cfg["python"]["bp"]

    @property
    def python_re(self) -> str:
        return self.cfg["python"]["realenv"]

    @property
    def zarr(self) -> str:
        return str(self.policy.get("prompt_zarr") or self.cfg["prompt_zarr"])

    @property
    def criteria(self) -> dict:
        return self.station.get("criteria")

    @property
    def sensor(self) -> str:
        return self.ids.get("sensor", "?")

    @property
    def image_source(self) -> str:
        return str(self.station.get("sim.image_source", "unknown"))

    @property
    def protocol(self) -> str:
        return L.wire_protocol(self.station.get("sensor.transport.protocol"))

    def station_sets(self, groups=STATION_GROUPS) -> list:
        return [s for s in self.sets if str(s).partition("=")[0].partition(".")[0] in groups]

    def run_dir(self, stage: str) -> Path:
        h = hashlib.sha256("|".join(self.user_sets + ([f"tag={self.tag}"] if self.tag else [])).encode()).hexdigest()[:8]
        return self.out / f"{stage}_{self.policy.get('id')}_{self.station.fingerprint()[:8]}_{h}"

    def demo_list(self):
        a = self.args
        return [int(x) for x in a.demos.split(",")] if getattr(a, "demos", None) else None


def resolve(args, run_overrides: dict | None = None, extra_sets=None) -> Run:
    """Profile + policy + sets -> Run. ``run_overrides`` = {group: id} (a run group such as link / policy / sim, or a station
    group such as robot / sensor), ``extra_sets`` = more ``group.path=value`` overrides (sweep points); both count as user
    sets for the run-dir hash. Callers that pass neither get exactly the old behaviour."""
    cfg = L.load_config(args.deploy_config or DEFAULT_DEPLOY, args.profile)
    if "station" not in cfg:
        usage_error(f"profile {args.profile!r} has no station: eval needs a station profile (sim-*)")
    run_ids = dict(cfg.get("run", {}))
    station_ids = dict(cfg["station"])
    if args.policy:
        run_ids["policy"] = args.policy
    for g, i in (run_overrides or {}).items():
        (station_ids if g in STATION_ID_GROUPS else run_ids)[g] = i
    cfg["station"] = station_ids
    user_sets = list(getattr(args, "set", None) or []) + [str(x) for x in (extra_sets or [])]
    auto_tcp = (cfg.pop("_sim_tcp", None) or {}).get("set")   # W10B: the launcher's derived TCP for the profile's DEFAULT
    # station; the user sets (extension, side, robot ...) may change the geometry, so it is dropped here and every launcher
    # child re-derives it from the full station (launcher.sim_tcp_fill), an explicit user tooling.F_T_TCP set still wins
    sets = [x for x in cfg.get("station_set", []) if x != auto_tcp] + user_sets
    try:
        station = load_station({"station": station_ids, "run": run_ids}, sets=sets)
    except StationError as e:
        usage_error(str(e))
    cfg["station_set"] = sets
    if args.policy or run_overrides:
        cfg["run"] = run_ids
    policy = L.resolve_policy(cfg, station)
    out = Path(args.out).resolve()
    try:
        if out.is_relative_to(REPO_ROOT):
            usage_error(f"refusing to write inside the repo: {out}")
    except AttributeError:      # pragma: no cover
        pass
    out.mkdir(parents=True, exist_ok=True)
    return Run(args=args, cfg=cfg, station=station, policy=policy, ids=dict(cfg["station"]), run_ids=run_ids, sets=sets,
               user_sets=user_sets, out=out, seed=int(getattr(args, "seed", 0) or 0), tag=str(getattr(args, "sim_stub", None) or ""))


def child_env(extra: dict | None = None) -> dict:
    """Environment for a stage subprocess: franka_deploy (+ franka_sim) on PYTHONPATH, no stray user path."""
    e = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    e["PYTHONPATH"] = os.pathsep.join([str(DEPLOY_PKG), str(SIM_PKG)])
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.update(extra or {})
    return e


def write_policy_yaml(R: Run, path: Path) -> Path:
    """The resolved policy group as YAML for predict_demos (python = the BP interpreter for the server kinds)."""
    pol = {k: v for k, v in R.policy.items() if k != "provenance"}
    kind = pol.get("kind")
    if kind in ("bp_server", "command"):
        pol["python"] = L.policy_python(R.cfg, R.policy)
        pol.setdefault("checkpoint", R.cfg.get("checkpoint"))
        pol.setdefault("prompt_zarr", R.cfg.get("prompt_zarr"))
    path.write_text(yaml.safe_dump(pol))
    return path


STAGE_FN = {"1": "stage1", "1b": "stage1b", "2a": "stage2a", "2b": "stage2b", "3": "stage3"}


def run_one_stage(R: Run, stage: str, results: dict) -> dict:
    """Run one stage into its run dir; returns the report dict. Never raises for a stage failure: INCOMPLETE + notes."""
    import importlib

    mod = importlib.import_module(f"franka_sim.eval.{STAGE_FN[stage]}")
    d = R.run_dir(stage)
    d.mkdir(parents=True, exist_ok=True)
    (d / "station_resolved.yaml").write_text(R.station.to_yaml())
    (d / "criteria.yaml").write_text(yaml.safe_dump(R.criteria))
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    notes = list(R.notes)
    n_notes0 = len(R.notes)
    rows = None
    metrics: dict = {}
    try:
        res = mod.run(R, d, results)
        metrics, rows = res["metrics"], res.get("rows")
        notes += res.get("notes", [])
    except StageIncomplete as e:
        notes.append(f"INCOMPLETE: {e}")
    except Exception as e:      # noqa: BLE001 - a crashed stage is reported, not raised
        notes.append(f"INCOMPLETE: stage {stage} crashed: {e!r}")
        (d / "error.txt").write_text(traceback.format_exc())
    notes += R.notes[n_notes0:]         # notes a stage added to the run context belong to this stage only
    del R.notes[n_notes0:]
    metrics.setdefault("protocol", R.protocol)
    ev = crit.evaluate(R.criteria, metrics, stage)
    sec = crit.STAGE_SECTION.get(stage, "")
    ph = sorted(p for p, _ in R.station.placeholders()
                if p.startswith(("criteria.limits.", "criteria.success.", f"criteria.{sec}.")))
    if ph:
        notes.append("criteria values that are open user decisions (placeholders): " + ", ".join(ph))
    if metrics.get("incomplete_reason"):       # a stage that stopped early: whatever it measured is listed, the verdict is INCOMPLETE
        ev["verdict"] = "INCOMPLETE"
        notes.append(f"INCOMPLETE: {metrics['incomplete_reason']}")
    rep = rpt.make_report(stage=stage, policy_id=R.policy.get("id"), policy_kind=R.policy.get("kind"), profile=R.profile,
                          sensor=R.sensor, robot=R.ids.get("robot"), station_fingerprint=R.station.fingerprint(),
                          image_source=R.image_source, seed=R.seed, started=started, duration_s=time.time() - t0,
                          metrics=metrics, evaluation=ev, notes=notes, rows=rows)
    rpt.write_json(d / "metrics.json", metrics)
    rpt.write_report(d, rep)
    print(f"[eval] stage {stage}: {rep['verdict']}  ({d})", flush=True)
    for n, g in rep["gates"].items():
        print(f"         {n:55s} {g['value']!s:>12}  {g['op']} {g['threshold']}  {g['pass']}", flush=True)
    for n in notes:
        if str(n).startswith("INCOMPLETE"):
            print(f"         {n}", flush=True)
    results[stage] = {"dir": d, "report": rep}
    return rep


def exit_code(verdicts) -> int:
    """0 = every stage PASS; 3 = at least one FAIL; 1 = no FAIL but at least one INCOMPLETE (a FAIL is a definite result, so
    it wins over a stage that could not finish); 2 is reserved for usage / config errors (``usage_error``)."""
    v = list(verdicts)
    return EXIT_FAIL if "FAIL" in v else EXIT_INCOMPLETE if "INCOMPLETE" in v else EXIT_PASS


def run(args) -> int:
    if getattr(args, "sweep", None):
        from franka_sim.eval import sweeps

        return sweeps.run_sweep(args)
    R = resolve(args)
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    results: dict = {}
    reps = []
    for st in stages:
        reps.append(run_one_stage(R, st, results))
    rpt.write_summary(R.out)
    return exit_code(r["verdict"] for r in reps)
