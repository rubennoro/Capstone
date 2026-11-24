"""Per rollout of the cone experiment evals: duration, end reason, max tilt (TCP z vs straight down), max tilt while
the TCP is low, table contacts by non-eraser geoms, lowest finger/rig z, grasp / lift. Usage: python cone_analyse.py EVAL..."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def tilt_deg(q_wxyz):
    w, x, y, z = q_wxyz
    zax = Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2]
    return float(np.degrees(np.arccos(np.clip(-zax[2], -1, 1))))


for ev in sys.argv[1:]:
    ev = Path(ev)
    print(f"\n{ev.parent.name} / {ev.name}")
    print(f"{'ro':>3} {'dur s':>6} {'outcome':>8} {'tilt0':>6} {'maxtilt':>7} {'tilt@z<8cm':>10} {'minz cm':>7} "
          f"{'tbl rows':>8} {'grasp':>5} {'lift cm':>7}  table geoms / end")
    for gt in sorted(ev.glob("rollouts/*/gt.jsonl")):
        rows = [json.loads(l) for l in gt.read_text().splitlines() if l.strip()]
        if not rows:
            continue
        rec = json.loads((gt.parent / "rollout.json").read_text())
        tilts = np.array([tilt_deg(r["tcp_pose"][3:7]) for r in rows])
        tz = np.array([r["tcp_pose"][2] for r in rows])
        low = tilts[tz < 0.08]
        tbl = [r for r in rows if r.get("contacts_table")]
        geoms = sorted({g for r in tbl for g in r["contacts_table"]})
        grasp = any({"l", "r"} <= {g.split("_")[-1][:1] for g in r.get("contacts_eraser", [])} for r in rows)
        z0 = rows[0]["eraser_pose"][2]
        lift = max(r["eraser_pose"][2] for r in rows) - z0
        sc = {}
        sp = ev / "score.json"
        end = rec.get("stop_reason") or rec.get("outcome")
        print(f"{gt.parent.name:>3} {rows[-1]['t']:6.1f} {str(rec.get('outcome')):>8} {tilts[0]:6.1f} {tilts.max():7.1f} "
              f"{(low.max() if low.size else float('nan')):10.1f} {min(r['min_z_fingers_rig_m'] for r in rows)*100:7.1f} "
              f"{len(tbl):8d} {str(grasp):>5} {lift*100:7.1f}  {','.join(geoms) or '-'} / {end}")
