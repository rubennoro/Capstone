"""W2 step 2: fit the Pi cameras' image formation (``camera_look.py``) to the real 640x480 frames.

    unset PYTHONPATH; cd ~/Capstone-main-sim
    PY=~/envs/sim_deploy/bin/python
    $PY -m sim_deploy.look.fit_camera_look render --station sim_deploy/stations/sim_rig_look_real.yaml   # GL (EGL)
    $PY -m sim_deploy.look.fit_camera_look fit    --look sim_rig_look_real [--write-yaml]                # CPU
    $PY -m sim_deploy.look.fit_camera_look images --look sim_rig_look_real                               # CPU

Real: the 15 references of ``sim_evals/_camera_study/B/layout.json`` (B1), ``reference/ep<ii>_<label>/raw640_{main,
uw}_{0,1}.png``: the Pi recordings (PyAV decode) INTER_AREA to 640x480, what the Pi streams before its JPEG.

``render``: the same views in the sim, HANDHELD camera pose (``camera_study.render_sets``: ``Study``, ``plan_scenes``,
``load_handheld_render_model``: training's T_cam_tcp + E7's fix, arm removed), the sim Pi chain up to the 640x480
frame (``CameraChain.frame``), WITHOUT any camera look, no JPEG. Eraser placement: ``sim_evals/_pipeline/W2/sets/
placement.json`` (step 3) when it exists, else ``Study.fit_placement()`` (in memory; saved beside the renders). Per
image also a label image (a second render of the same views with each geom class drawn in a flat binary colour,
no light, texture, skybox or shadow: 1 table, 2 eraser, 3 pads, 4 holders, 5 tags, 6 rig parts, 7 other, 0 sky) and
the table-plane point (O x, y) of every pixel's ray through the copy's 640 camera model. Out:
``sim_evals/_pipeline/W2/camera/renders/<look>/<key>.npz`` + ``meta.json``.

``fit``: region statistics real vs sim (labels from the sim, eroded so that a few px of misregistration do not mix
regions; the IMX219 table only within ``UW_TABLE_MAX_M`` of the fingertips on the table, the lab beyond the real
table edge is step 1's real_bg question), per camera:

* edges: an erf fit across the label boundaries (eraser|table, tag black|white, pad|table, holder|table) in real and
  sim -> 10-90 % width; ``blur_sigma_px`` = sqrt(median(sigma_real^2 - sigma_sim^2));
* noise: Immerkaer's estimator in 8x8 blocks of flat regions vs block level (real), lag-1 autocorrelation and the
  channel correlation of the high-pass residual; fitted s0, s1 (camera_look's sqrt-domain noise) so the same
  estimator on the sim + model matches the real curve;
* colour + shading: quantiles (10/25/50/75/90 %) per channel of each region (and of the table in radial bands
  around the lens centre) of the sim pixels through the model vs the real pixels, least squares over the colour
  matrix, offset, tone gammas and the shading map (IMX519: diagonal gains + mild radial shading only, so the scene
  is not fitted into the camera; IMX219: full matrix + radial + tilt shading);
* before / after: means, mean abs CDF difference per channel and region, noise and edge widths, timing.

``--write-yaml`` writes ``sim_deploy/look/camera_look_real.yaml``; ``camera_look.json`` (results) always.

``images``: real | sim (no camera model) | sim + camera model, per camera, at 640 and at 224 (JPEG q90 + the copy's
ImagePrep), for ep14_first and two others: ``sim_evals/_pipeline/W2/camera/compare_<key>.png``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sim_evals" / "_pipeline" / "W2" / "camera"
B_REF = REPO / "sim_evals" / "_camera_study" / "B" / "reference"
STEP3_PLACEMENT = REPO / "sim_evals" / "_pipeline" / "W2" / "sets" / "placement.json"
ROLES = ("main", "ultrawide")
SHORT = {"main": "main", "ultrawide": "uw"}
CLASSES = {0: "sky", 1: "table", 2: "eraser", 3: "pads", 4: "holders", 5: "tags", 6: "rig", 7: "other"}


def geom_class(name: str, body: str) -> int:
    if name in ("table", "table_top") or name.startswith("table_vis"):   # the colliding table / a look's visual slab
        return 1
    if name == "eraser":
        return 2
    if name in ("pad_l", "pad_r"):
        return 3
    if name in ("finger_holder_l", "finger_holder_r"):
        return 4
    if name in ("tag_l", "tag_r"):
        return 5
    if name.startswith("rig_") or body == "rig":
        return 6
    return 7


# ------------------------------------------------------------------------------------------------ render stage
def stage_render(station: Path, out_root: Path, only: set | None) -> int:
    import sim_deploy.sim_pi.render  # noqa: F401  (EGL + NVIDIA before mujoco)
    import cv2
    import mujoco
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.sim_pi.cameras import CameraChain, _grid, fisheye_rays
    from sim_deploy.sim_pi.render import GLRenderer, Lighting
    from sim_deploy.sim_pi._deploy import PinholeRadtan
    cv2.setNumThreads(4)
    t0 = time.perf_counter()
    st = RS.Study(station)
    look_id = st.look_id
    out = out_root / "renders" / look_id
    out.mkdir(parents=True, exist_ok=True)
    if STEP3_PLACEMENT.is_file():
        placement = json.loads(STEP3_PLACEMENT.read_text())
        pl_src = str(STEP3_PLACEMENT.relative_to(REPO))
    else:
        placement = st.fit_placement()
        pl_src = "Study.fit_placement() (in memory; step 3's placement.json did not exist)"
    (out / "placement_used.json").write_text(json.dumps({"source": pl_src, "placement": placement}, indent=1))
    scenes = RS.plan_scenes(st, placement)
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    lighting = Lighting(shadows=st.s.shadows)
    rm, info = RS.load_handheld_render_model(st.s.model, lighting, W, H, st.s.keep_model_lights, st.tcp_T_cam["E7"])
    for r, c in chains.items():
        c.set_hide([rm.own_board[r]])
    gl = GLRenderer(rm.model, W, H)
    gl.set_shadows(st.s.shadows)
    m = rm.model
    gclass = np.array([geom_class(m.geom(i).name, m.body(m.geom_bodyid[i]).name) for i in range(m.ngeom)], int)
    codes = {k: np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1], float) for k in range(8)}
    # 640 rays per pixel (camera frame, OpenCV) through the copy's 640 model, for the table-plane points
    uv = _grid(640, 480)
    rays = {}
    for role, c in chains.items():
        mod = c.model_out
        if isinstance(mod, PinholeRadtan):
            rr, ok = mod.unproject(uv)
        else:
            rr, ok, _ = fisheye_rays(mod, uv)
        rr = np.asarray(rr, float).reshape(480, 640, 3)
        rr[~np.asarray(ok).reshape(480, 640)] = np.nan
        rays[role] = rr
    table_z = float(st.doc["table"]["top_z_m"])
    meta = {"station": str(station), "look": look_id, "model": str(st.s.model), "model_sha256": rm.sha256,
            "placement_source": pl_src, "handheld_model_edits": info, "shadows": bool(st.s.shadows),
            "keep_model_lights": bool(st.s.keep_model_lights), "density": float(st.s.density),
            "classes": CLASSES, "refs": {}}
    try:
        for key, sc in scenes.items():
            if only and key not in only:
                continue
            arrs = {}
            for role in ROLES:
                for j, row in enumerate(sc["rows"][role]):
                    pr = sc["per_row"][row]
                    qpos = RS.qpos_for(st.kin, pr["q"], pr["gap"], sc["eraser_pose"])
                    rm.set_qpos(qpos)
                    gl.update(rm.data)
                    Rw, pw = rm.camera_pose(role)
                    rgb = chains[role].frame(gl, Rw, pw)
                    scn = gl.scn
                    for i in range(scn.ngeom):
                        g = scn.geoms[i]
                        k = int(gclass[g.objid]) if g.objtype == mujoco.mjtObj.mjOBJ_GEOM else 7
                        a = float(g.rgba[3])
                        g.rgba[:] = [*codes[k], a]
                        g.emission = 1.0
                        g.specular = 0.0
                        g.reflectance = 0.0
                        g.matid = -1
                    fl = scn.flags
                    sky, shadow = fl[mujoco.mjtRndFlag.mjRND_SKYBOX], fl[mujoco.mjtRndFlag.mjRND_SHADOW]
                    fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
                    fl[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    try:
                        code = chains[role].frame(gl, Rw, pw)
                    finally:
                        fl[mujoco.mjtRndFlag.mjRND_SKYBOX] = sky
                        fl[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
                    bits = (code >= 128).astype(np.uint8)
                    lab = bits[..., 0] | (bits[..., 1] << 1) | (bits[..., 2] << 2)
                    pure = np.all((code <= 40) | (code >= 215), axis=2)
                    lab = np.where(pure, lab, 255).astype(np.uint8)         # 255 = mixed (edge) pixel
                    d = rays[role] @ Rw.T
                    t = (table_z - pw[2]) / d[..., 2]
                    hit = pw[None, None, :2] + t[..., None] * d[..., :2]
                    hit[~(t > 0)] = np.nan
                    nm = f"{SHORT[role]}_{j}"
                    arrs[f"rgb_{nm}"] = rgb
                    arrs[f"label_{nm}"] = lab
                    arrs[f"table_xy_{nm}"] = hit.astype(np.float32)
                    arrs[f"O_T_cam_{nm}"] = RS._T(Rw, pw)
                    arrs[f"O_T_tcp_{nm}"] = pr["O_T_tcp"]
                    arrs[f"gap_{nm}"] = np.float64(pr["gap"])
            np.savez_compressed(out / f"{key}.npz", **arrs)
            meta["refs"][key] = {"rows": sc["rows"], "eraser_pose_O": [float(v) for v in sc["eraser_pose"]],
                                 "dz_m": float(sc["dz_m"])}
            print(f"[camfit] rendered {key}", flush=True)
        meta["vram"] = RS._vram()
    finally:
        gl.close()
    meta["render_s"] = round(time.perf_counter() - t0, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[camfit] {look_id}: {len(meta['refs'])} references in {meta['render_s']} s, VRAM {meta.get('vram')}",
          flush=True)
    return 0


def stage_mount_labels(station: Path, out_root: Path) -> int:
    """Label images of the MOUNT view at the start pose (arm at q_start, robot fingertip gap 0.040 m, the eraser moved
    out of view), at 640 and at 224 through the copy's ImagePrep: where the fingers / holders / tags / arm are in the
    real mount frames (used by the ``mount`` check to keep only the table)."""
    import sim_deploy.sim_pi.render  # noqa: F401
    import cv2
    import mujoco
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.sim_pi.cameras import CameraChain
    from sim_deploy.sim_pi.model_file import load_render_model
    from sim_deploy.sim_pi.render import GLRenderer, Lighting
    st = RS.Study(station)
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    rm = load_render_model(st.s.model, Lighting(shadows=False), W, H, keep_model_lights=st.s.keep_model_lights)
    for r, c in chains.items():
        c.set_hide([rm.own_board[r]])
    gl = GLRenderer(rm.model, W, H)
    m = rm.model
    gclass = np.array([geom_class(m.geom(i).name, m.body(m.geom_bodyid[i]).name) for i in range(m.ngeom)], int)
    codes = {k: np.array([(k >> 0) & 1, (k >> 1) & 1, (k >> 2) & 1], float) for k in range(8)}
    qpos = RS.qpos_for(st.kin, st.q_start, 0.040, st.kin.eraser_pose(2.0, 2.0, 0.0))
    arrs = {}
    try:
        rm.set_qpos(qpos)
        gl.update(rm.data)
        scn = gl.scn
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            k = int(gclass[g.objid]) if g.objtype == mujoco.mjtObj.mjOBJ_GEOM else 7
            g.rgba[:] = [*codes[k], float(g.rgba[3])]
            g.emission, g.specular, g.reflectance, g.matid = 1.0, 0.0, 0.0, -1
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        for role in ROLES:
            Rw, pw = rm.camera_pose(role)
            code = chains[role].frame(gl, Rw, pw)
            c224 = st.prep.prep_main(code) if role == "main" else st.prep.prep_ultrawide(code)
            for nm, c in (("640", code), ("224", c224)):
                bits = (c >= 128).astype(np.uint8)
                lab = bits[..., 0] | (bits[..., 1] << 1) | (bits[..., 2] << 2)
                pure = np.all((c <= 40) | (c >= 215), axis=2)
                arrs[f"label{nm}_{SHORT[role]}"] = np.where(pure, lab, 255).astype(np.uint8)
            arrs[f"code224_{SHORT[role]}"] = c224
    finally:
        gl.close()
    out = out_root / "renders" / "mount_start_labels.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrs, model_sha256=np.array(rm.sha256), q_start=st.q_start)
    print(f"[camfit] wrote {out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.look.fit_camera_look", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--station", type=Path, required=True)
    r.add_argument("--only", nargs="*", default=None)
    r.add_argument("--out", type=Path, default=OUT)
    f = sub.add_parser("fit")
    f.add_argument("--look", required=True)
    f.add_argument("--write-yaml", action="store_true")
    f.add_argument("--out", type=Path, default=OUT)
    i = sub.add_parser("images")
    i.add_argument("--look", required=True)
    i.add_argument("--yaml", type=Path, default=REPO / "sim_deploy" / "look" / "camera_look_real.yaml")
    i.add_argument("--out", type=Path, default=OUT)
    ml = sub.add_parser("mount-labels")
    ml.add_argument("--station", type=Path, default=REPO / "sim_deploy" / "stations" / "sim_rig.yaml")
    ml.add_argument("--out", type=Path, default=OUT)
    mo = sub.add_parser("mount")
    mo.add_argument("--yaml", type=Path, default=REPO / "sim_deploy" / "look" / "camera_look_real.yaml")
    mo.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if a.cmd == "mount-labels":
        return stage_mount_labels(a.station.resolve(), a.out)
    if a.cmd == "mount":
        from sim_deploy.look import _camera_mount224 as CM
        return CM.stage_mount(a.yaml, a.out)
    if a.cmd == "render":
        return stage_render(a.station.resolve(), a.out, set(a.only) if a.only else None)
    if a.cmd == "fit":
        from sim_deploy.look import _camera_fit as CF
        return CF.stage_fit(a.look, a.out, a.write_yaml)
    if a.cmd == "images":
        from sim_deploy.look import _camera_fit as CF
        return CF.stage_images(a.look, a.out, a.yaml)
    return 2


if __name__ == "__main__":
    sys.exit(main())
