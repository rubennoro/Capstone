"""Stage 1 wrapper: ``python -m franka_sim stage1`` (A3) for the profile's station, then gates from ``criteria.stage1``.

The design sweep is not repeated: the station's current tooling values (finger mode and its length, camera side) are
used; the metrics carry ``finger_mode``, ``swept_key`` / ``swept_value`` and ``flange_to_tip_m`` (A9d).
The placement search runs in a subprocess (one BLAS thread per worker, as A3's entry point sets up); its stage1.json is
kept in the run dir (``stage1/<combo>/stage1.json``) and read by Stage 2a (``--stage1``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from franka_sim.eval import runner as rn

KEY_COLUMNS = ("demo", "feasible", "reach_fraction", "min_margin_deg", "min_tooling_clearance_m", "n_collisions",
               "clamp_fraction", "slowdown")


def stage1_json_in(d: Path) -> Path | None:
    """The stage1.json under a Stage 1 run dir (``<dir>/stage1.json`` or ``<dir>/stage1/<combo>/stage1.json``)."""
    d = Path(d)
    for c in (d / "stage1.json", *sorted(d.glob("stage1/*/stage1.json")), *sorted(d.glob("*/stage1.json"))):
        if c.is_file():
            return c
    return None


def run_stage1_cli(R: rn.Run, demos: list | None, out: Path, log_path: Path) -> Path:
    """Run A3's CLI for ``demos`` (zarr indices, None = all); returns the stage1.json path."""
    a = R.args
    cmd = [sys.executable, "-m", "franka_sim", "stage1", "--zarr", R.zarr, "--ids",
           ",".join(f"{k}={v}" for k, v in R.ids.items()), "--run", f"criteria={R.run_ids.get('criteria', 'block_raise')},sim=stage1",
           "--demos", "all" if demos is None else ",".join(str(x) for x in demos), "--jobs", str(max(1, int(getattr(a, "jobs", 1)))),
           "--out", str(out)]
    for s in R.station_sets():
        cmd += ["--set", s]
    if getattr(a, "quick", False):
        cmd.append("--quick")
    with open(log_path, "w") as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, env=rn.child_env({"OMP_NUM_THREADS": "1"}),
                           cwd=str(SIM_PKG_DIR))
    p = stage1_json_in(out)
    if r.returncode != 0 or p is None:
        raise rn.StageIncomplete(f"franka_sim stage1 failed (rc {r.returncode}); see {log_path}")
    return p


SIM_PKG_DIR = rn.SIM_PKG


def metrics_from(res: dict) -> tuple[dict, list]:
    s, rows = res["summary"], res["demos"]
    n = len(rows)
    m = {
        "n_demos": n, "n_feasible": s["n_feasible"], "feasible_fraction": s["n_feasible"] / max(1, n),
        "reachable_fraction": float(np.mean([r["reach_fraction"] for r in rows])),
        "joint_margin_deg": s["worst_joint_margin_all_demos_deg"],
        "joint_margin_feasible_deg": s["worst_joint_margin_feasible_deg"],
        "collisions": int(sum(r["n_collisions"] for r in rows)),
        "tooling_clearance_min_m": s["worst_tooling_clearance_m"],
        "clamp_fraction": s["clamp_fraction"], "slowdown_mean": s["slowdown_mean"], "slowdown_max": s["slowdown_max"],
        "width_p99_mapped_le_stroke": bool(all(r["mapped_p99_le_stroke"] for r in rows)),
        "width_gap_p99_m": max(r["gap_p99_m"] for r in rows),
        "recommended_placement": res["recommendation"]["grid_point"] | {"yaw_deg": res["recommendation"]["yaw_deg"]},
        "robot": res["robot"], "finger_extension_m": res["design"]["extension_m"], "camera_side": res["design"]["camera_side"],
        # A9 (absent in stage1.json written before A9d -> None): finger mode, the swept length key, flange to fingertip
        "finger_mode": res["design"].get("finger_mode"), "swept_key": res["design"].get("swept_key"),
        "swept_value": res["design"].get("swept_value"), "flange_to_tip_m": res["design"].get("flange_to_tip_m"),
    }
    table = [{"demo": r["name"][:30], "feasible": r["feasible"], "reach_fraction": round(r["reach_fraction"], 3),
              "min_margin_deg": round(r["min_margin_deg"], 1), "min_tooling_clearance_m": round(r["min_tooling_clearance_m"], 4),
              "n_collisions": r["n_collisions"], "clamp_fraction": round(r["clamp_fraction"], 3),
              "slowdown": round(r["slowdown"], 2)} for r in rows]
    return m, table


def run(R: rn.Run, d: Path, results: dict) -> dict:
    demos = R.demo_list()
    if demos is None and getattr(R.args, "episodes", None):
        demos = list(range(int(R.args.episodes)))
    p = run_stage1_cli(R, demos, d / "stage1", d / "stage1.log")
    res = json.loads(p.read_text())
    m, table = metrics_from(res)
    results["stage1_json"] = p
    if m["swept_key"] is None and m["finger_mode"] is None:     # stage1.json from before A9d
        fing = f"finger extension {m['finger_extension_m']} m"
    else:
        fing = (f"finger mode {m['finger_mode']}, "
                + (f"{m['swept_key']} {m['swept_value']} m" if m["swept_key"] else "no length axis")
                + (f", flange_to_tip {m['flange_to_tip_m']:.4f} m" if m["flange_to_tip_m"] is not None else ""))
    notes = [f"stage1.json: {p}", f"{res['quick'] and 'quick grid' or 'full grid'}; robot {res['robot']}, {fing}, "
             f"camera side {m['camera_side']}; recommended placement {m['recommended_placement']}"]
    return {"metrics": m, "rows": table, "notes": notes}
