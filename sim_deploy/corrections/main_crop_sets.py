"""W2 step 4 (opt-in experiment, default off): the IMX519 crop/zoom correction applied to rendered image sets.

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.corrections.main_crop_sets --look train [--poses mount mount_start robot_trials]
        [--yaml sim_deploy/corrections/main_crop_corr.yaml] [--suffix c519]

For every set of ``sim_evals/_pipeline/W2/sets/<look>/<pose>/<ref>/``: the exact JPEG bytes the sim Pi sent
(``jpg640_main_{0,1}.jpg``) decoded as render_sets does (cv2, BGR -> RGB), ``main_crop.apply`` (the fitted similarity
of the 640x480 main frame, mount -> handheld, ``fit_main.py``), then the copy's ``ImagePrep.prep_main`` (the rig file
of the station, ``rig_pi_v3_640.yaml`` via ``sim_pi.settings``). The IMX219 images and the low-dim are copied
unchanged. Output: ``sets/<look>_<suffix>/<pose>/<ref>/`` (``obs.npz``, ``main_{0,1}.png``, ``meta.json``) and index
rows with look ``<look>_<suffix>`` (``attribution_table.py`` then plans its jobs like any other look). CPU only.

This only models what a crop in the laptop's frame prep would do (PLAN.md 10 B5: a sim-only option, never on by
default); a positive result is a recommendation for the real deployment, not a change of the sim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from sim_deploy.corrections import main_crop
from sim_deploy.sim_pi import _deploy as D

REPO = Path(__file__).resolve().parents[2]
SETS = REPO / "sim_evals" / "_pipeline" / "W2" / "sets"
DEFAULT_YAML = REPO / "sim_deploy" / "corrections" / "main_crop_corr.yaml"


def _prep(station: Path):
    """The copy's ImagePrep for the station's rig file (render_sets' own loader: sim_pi.settings semantics)."""
    from sim_deploy.camera_study.render_sets import load_station_settings
    s, _, _ = load_station_settings(station)
    return D.ImagePrep(D.load_live_rig(s.rig_file)), str(s.rig_file)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--look", required=True)
    ap.add_argument("--poses", nargs="+", default=["mount", "mount_start", "robot_trials"])
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML)
    ap.add_argument("--suffix", default="c519")
    a = ap.parse_args(argv)
    crop = main_crop.load(str(a.yaml))
    idx_path = SETS / "index.json"
    idx = json.loads(idx_path.read_text())
    rows = [r for r in idx["sets"] if r["look"] == a.look and r["pose"] in a.poses]
    if not rows:
        raise SystemExit(f"no sets for look {a.look} poses {a.poses} in {idx_path}")
    new_look = f"{a.look}_{a.suffix}"
    prep, rig_file = _prep(REPO / rows[0]["station"])
    new_rows = []
    for r in rows:
        src = REPO / r["folder"]
        dst = SETS / new_look / r["pose"] / src.name
        dst.mkdir(parents=True, exist_ok=True)
        o = dict(np.load(src / "obs.npz"))
        mains = []
        for i in (0, 1):
            jpg = np.frombuffer((src / f"jpg640_main_{i}.jpg").read_bytes(), np.uint8)
            dec = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img = prep.prep_main(crop.apply(dec))
            mains.append(img)
            cv2.imwrite(str(dst / f"main_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        o["main"] = np.stack(mains).astype(np.uint8)
        np.savez(dst / "obs.npz", **o)
        for f in src.glob("obs_w*.npz"):              # width variants (robot_trials): same images, other width
            ov = dict(np.load(f))
            ov["main"] = o["main"]
            np.savez(dst / f.name, **ov)
        for f in ("uw_0.png", "uw_1.png", "real_static_sets.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        meta = {"source_set": str(src.relative_to(REPO)), "correction": str(a.yaml.relative_to(REPO)),
                "correction_yaml_sha256": __import__("hashlib").sha256(a.yaml.read_bytes()).hexdigest(),
                "rig_file": rig_file, "what": "main = ImagePrep.prep_main(main_crop.apply(decoded jpg640_main_i)); "
                "ultrawide and low-dim copied from the source set"}
        (dst / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
        nr = dict(r)
        nr.update(look=new_look, folder=str(dst.relative_to(REPO)), obs=str((dst / "obs.npz").relative_to(REPO)),
                  note=f"W2 step 4 opt-in: {r['look']} with the IMX519 crop correction ({a.yaml.name})")
        if "obs_variants" in nr:
            nr["obs_variants"] = {k: v.replace(f"/{r['look']}/", f"/{new_look}/") for k, v in nr["obs_variants"].items()}
        if "real_static_sets" in nr:
            nr["real_static_sets"] = str((dst / "real_static_sets.json").relative_to(REPO))
        new_rows.append(nr)
    idx = json.loads(idx_path.read_text())                 # re-read: another process may have written meanwhile
    idx["sets"] = sorted([s for s in idx["sets"] if s["look"] != new_look] + new_rows,
                         key=lambda s: (s["look"], s["pose"], s["reference"]))
    idx["looks"] = sorted({s["look"] for s in idx["sets"]})
    tmp = idx_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, indent=1))
    os.replace(tmp, idx_path)
    print(f"{len(new_rows)} sets -> {SETS / new_look}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
