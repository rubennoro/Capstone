"""python -m sim_deploy.model build --station sim_deploy/stations/sim_rig.yaml [--out DIR]
python -m sim_deploy.model info [--model sim_deploy/build/sim_rig/model.json]   (print the summary)"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sim_deploy import ROOT


def _summary(doc: dict) -> str:
    j = doc["joints"]
    lines = [f"model {doc['station_id']}  sha256 {doc['model_sha256']}  mujoco {doc['mujoco_version']}",
             f"  nq {doc['nq']}  nv {doc['nv']}  nu {doc['nu']}  nbody {doc['nbody']}  ngeom {doc['ngeom']}  "
             f"timestep {doc['timestep_s']} s  robot mass {doc['robot_mass_kg']:.3f} kg (hand subtree {doc['hand_subtree_mass_kg']:.3f})",
             "  qpos: " + ", ".join(f"{n}@{v['qposadr']}" for n, v in j.items()),
             f"  sites {doc['sites']}  cameras {list(doc['cameras'])}",
             f"  home q {doc['home']['q_arm_rad']}  gap {doc['home']['finger_gap_m']} m",
             "  home tcp (O) " + str([round(v, 4) for v in [r[3] for r in doc['home']['O_T_tcp'][:3]]]),
             f"  F_T_TCP z {doc['F_T_TCP'][2][3]} (frames.yaml); kit-derived differs by {doc['F_T_TCP_vs_kit']['mm']:.3f} mm "
             f"/ {doc['F_T_TCP_vs_kit']['deg']:.4f} deg",
             f"  eraser {doc['eraser']['size_m']} m, {doc['eraser']['mass_kg']} kg, default qpos "
             f"{[round(v, 4) for v in doc['eraser']['default_qpos']]}; table top z {doc['table']['top_z_m']}",
             f"  geom groups {doc['geom_group_counts']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build scene.xml + meshes/ + model.json")
    b.add_argument("--station", default=str(ROOT / "stations" / "sim_rig.yaml"))
    b.add_argument("--out", help="output folder (default sim_deploy/build/<station id>)")
    i = sub.add_parser("info", help="print a model.json summary")
    i.add_argument("--model", default=str(ROOT / "build" / "sim_rig" / "model.json"))
    a = ap.parse_args(argv)
    if a.cmd == "build":
        from sim_deploy.model.build import build

        doc = build(a.station, a.out)
    else:
        doc = json.loads(Path(a.model).read_text())
    print(_summary(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
