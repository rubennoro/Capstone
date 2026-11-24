"""W2 step 1 "look", the ``robot`` look: assets of the REAL ROBOT CELL from the real-arm mount frames.

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.make_robot --work DIR [--out sim_deploy/look/assets]

Input (read-only): the policy's 224x224 inputs saved on the real Franka (``~/Capstone/data/*/policy/*/deploy/
*_probe_live/NNNNNN_{main,ultrawide}_1.png``, after the real Pi's JPEG + the deployment's ImagePrep), the first
``N_STATIC`` sets of each trial (the arm still at the start pose; their per-pixel median). Geometry: the MOUNT cameras
of ``build/sim_rig`` at start_pose.yaml q_start with the robot's fingertip gap 0.040, and pure projection through the
copy's camera models into the 224 images (``render_sets.Projector``).

Measured, all in the 224 frames:
* the table: plain white; its right edge (O x) and far edge (O y) from the strongest brightness step in both
  cameras' frames rectified onto O z 0 (per row / column, median over the trials) -> ``TABLE`` extent;
* the table colour (IMX519 frames rectified onto z 0, robot parts and the object masked): a per-channel plane fit
  in O inside the IMX519's footprint, held constant beyond it -> ``table_robot.png`` (the visual table, 5 mm texels);
* the eraser / object colour (yellow: HSV hue 20-40, S > 120) and the holders' dark grey (inside the robot's own
  silhouette, V 40-150, S < 60), medians over the trials (trial4, a black object, excluded);
* the IMX219's colour shading on the robot: its table pixels vs the IMX519 table colour at the same O point, a
  per-channel gain in the fisheye radius (``backdrop.fit_radial_gain``);
* the backdrop: proxy planes beyond the table (floor at the sim floor z, a wall behind the far edge, a vertical
  plane through the person beside the table, a left wall) painted with the source trial's frames
  (``backdrop.bake``: IMX519 and IMX219 (shading divided out) median, the robot's own parts, the object and
  everything behind the table masked) -> ``backdrop_robot.obj`` / ``.png``.
The robot's own silhouette in each camera: a render of ``build/sim_rig`` at the start pose with only the arm / rig /
fingers drawn (one EGL context). Writes ``robot_look.json`` (all numbers) and check images in ``DIR``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import sim_deploy.sim_pi.render as R_  # noqa: F401  (EGL + NVIDIA before mujoco)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_deploy.look import backdrop as BD  # noqa: E402
from sim_deploy.sim_pi import _deploy as D  # noqa: E402

cv2.setNumThreads(4)
REPO = D.REPO
ASSETS = REPO / "sim_deploy" / "look" / "assets"
TRIALS_GLOB = "~/Capstone/data/*/policy/*/deploy/*_probe_live"
SOURCE_TRIAL = "2026-11-20_17-41_trial1_probe_live"        # PICK_DOUBLE run_002 (the deployment candidate)
EXCLUDE_COLOUR = ("2026-11-20_17-28_trial4_probe_live",)   # a black object, not the eraser
N_STATIC = 40
GAP_M = 0.040
FLOOR_Z = -0.70                                            # the sim's floor (sim_rig.yaml table.floor_z_m)
ROLES = ("main", "ultrawide")
SHORT = {"main": "main", "ultrawide": "ultrawide"}


def trials() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(os.path.expanduser(TRIALS_GLOB)))]


def static_median(d: Path, role: str) -> np.ndarray:
    st = np.stack([cv2.imread(str(d / f"{i:06d}_{SHORT[role]}_1.png")) for i in range(N_STATIC)])
    return cv2.cvtColor(np.median(st, 0).astype(np.uint8), cv2.COLOR_BGR2RGB)


class Geo:
    def __init__(self):
        from sim_deploy.camera_study import render_sets as RS
        scene = REPO / "sim_deploy" / "build" / "sim_rig" / "scene.xml"
        doc = json.loads(scene.with_name("model.json").read_text())
        self.kin = RS.Kin(scene, doc)
        import yaml
        sp = yaml.safe_load((REPO / "sim_deploy" / "stations" / "start_pose.yaml").read_text())
        from sim_deploy.look.make_assets import _find_key
        self.q_start = np.asarray(_find_key(sp, "q_start_rad"), float)
        self.kin.set(self.q_start, GAP_M, self.kin.eraser_pose(0.55, 0.0, 0.0))
        self.cam = {r: self.kin.cam_T(r) for r in ROLES}
        self.rig = D.load_live_rig(D.DEFAULT_RIG_FILE)
        self.prep = D.ImagePrep(self.rig)
        self.proj = RS.Projector(self.rig, self.prep)
        self.fish = D.physical_model(self.rig, "ultrawide")

    def project(self, role):
        T = self.cam[role]

        def f(P):
            pc = (P - T[:3, 3]) @ T[:3, :3]
            uv = self.proj(role, T, P)
            return uv, pc[:, 2] > 0.02
        return f

    def fish_radius(self, P):
        T = self.cam["ultrawide"]
        pc = (P - T[:3, 3]) @ T[:3, :3]
        uv = self.fish.project(pc)[0]
        cx, cy = self.fish.cx, self.fish.cy
        return np.hypot(uv[:, 0] - cx, uv[:, 1] - cy)

    def rectify(self, role, img, z, box, mm):
        (x0, x1), (y0, y1) = box
        xs = np.arange(x0, x1, mm / 1000)
        ys = np.arange(y1, y0, -mm / 1000)
        X, Y = np.meshgrid(xs, ys)
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
        uv, ok = self.project(role)(P)
        mx = np.where(ok, uv[:, 0], -1e4).reshape(X.shape).astype(np.float32)
        my = np.where(ok, uv[:, 1], -1e4).reshape(X.shape).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), xs, ys


def robot_silhouette(geo: Geo) -> dict:
    """{role: bool (224, 224)}: the arm / hand / rig / fingers at the start pose (render of build/sim_rig, scene
    geoms hidden, robot geoms emissive white)."""
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.sim_pi import settings as SPS
    from sim_deploy.sim_pi.cameras import CameraChain
    from sim_deploy.sim_pi.model_file import load_render_model
    from sim_deploy.sim_pi.render import Lighting

    s = SPS.load(REPO / "sim_deploy" / "stations" / "sim_rig.yaml", {})
    chains = {r: CameraChain(r, geo.rig, density=s.density) for r in ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    rm = load_render_model(s.model, Lighting(shadows=False), W, H, keep_model_lights=True)
    rend = RS.Renderer(rm, chains, s, geo.prep, None)
    out = {}
    try:
        q = rm.home_qpos.copy()
        q[:7] = geo.q_start
        q[7:9] = GAP_M / 2
        rm.set_qpos(q)
        rend.gl.update(rm.data)
        scn = rend.gl.scn
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and rm.model.geom_group[g.objid] == 0:
                g.rgba[3] = 0.0
            else:
                g.rgba[:3] = 1.0
                g.emission = 1.0
                g.matid = -1
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        for role in ROLES:
            Rw, pw = rm.camera_pose(role)
            img = rend._chain_frame(role, Rw, pw) if hasattr(rend, "_chain_frame") else chains[role].frame(rend.gl, Rw, pw)
            p = geo.prep.prep_main(img) if role == "main" else geo.prep.prep_ultrawide(img)
            out[role] = p.max(-1) > 100
    finally:
        rend.close()
    return out


def yellow_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] >= 20) & (hsv[..., 0] <= 40) & (hsv[..., 1] > 120) & (hsv[..., 2] > 90)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(m.shape, bool)
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == j


def edge_steps(geo, imgs) -> dict:
    """Right edge x (rows y -0.1..0.3, x 0.55..0.75) and far edge y (columns x 0..0.6, y 0.2..0.6): the strongest
    brightness step in the frames rectified onto z 0 (2 mm), per camera and trial."""
    res = {"x_edge": {r: [] for r in ROLES}, "y_far": {"ultrawide": []}}
    for name, (m, u) in imgs.items():
        for role, img in (("main", m), ("ultrawide", u)):
            im, xs, ys = geo.rectify(role, img, 0.0, ((0.4, 0.9), (-0.35, 0.6)), 2)
            g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(float), (0, 0), 2)
            gx = np.diff(g, axis=1)
            for yy in (-0.1, 0.0, 0.1, 0.2, 0.3):
                i = int(round((0.6 - yy) / 0.002))
                valid = (im[i, 1:].max(-1) > 0) & (im[i, :-1].max(-1) > 0)
                sel = (xs[:-1] > 0.58) & (xs[:-1] < 0.70) & valid
                if sel.sum() > 10:
                    res["x_edge"][role].append(float(xs[np.argmin(np.where(sel, gx[i], 1e9))]))
        im, xs, ys = geo.rectify("ultrawide", u, 0.0, ((-0.4, 0.7), (0.0, 0.8)), 2)
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(float), (0, 0), 2)
        gy = np.diff(g, axis=0)
        for xx in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            j = int(round((xx + 0.4) / 0.002))
            sel = (ys[:-1] > 0.25) & (ys[:-1] < 0.5) & (im[1:, j].max(-1) > 0)
            if sel.sum() > 10:
                res["y_far"]["ultrawide"].append(float(ys[np.argmax(np.where(sel, np.abs(gy[:, j]), -1))]))
    out = {}
    for k, per in res.items():
        for role, v in per.items():
            v = np.array(v)
            out[f"{k}_{role}"] = {"median": round(float(np.median(v)), 4), "p10_p90": np.round(np.percentile(v, [10, 90]), 4).tolist(),
                                  "n": int(len(v))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=str(ASSETS))
    a = ap.parse_args(argv)
    work = Path(a.work) / "robot"
    work.mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    geo = Geo()
    tr = trials()
    imgs = {t.name: (static_median(t, "main"), static_median(t, "ultrawide")) for t in tr}
    for n, (m, u) in imgs.items():
        cv2.imwrite(str(work / f"static_{n}_main.png"), m[:, :, ::-1])
        cv2.imwrite(str(work / f"static_{n}_uw.png"), u[:, :, ::-1])
    rep = {"trials": [str(t) for t in tr], "n_static_sets": N_STATIC, "source_trial": SOURCE_TRIAL,
           "mount_cameras_O": {r: np.round(geo.cam[r], 5).tolist() for r in ROLES}, "gap_m": GAP_M}
    sil = robot_silhouette(geo)
    for r in ROLES:
        cv2.imwrite(str(work / f"silhouette_{r}.png"), sil[r].astype(np.uint8) * 255)
    sil_d = {r: cv2.dilate(sil[r].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for r in ROLES}

    # -- table extent
    edges = edge_steps(geo, imgs)
    x_edge = round(float(np.median([edges["x_edge_main"]["median"], edges["x_edge_ultrawide"]["median"]])), 3)
    y_far = round(edges["y_far_ultrawide"]["median"], 3)
    table = {"x_m": [-0.40, x_edge], "y_m": [-0.80, y_far], "edges": edges,
             "note": "left / near sides: the colliding box's (not seen by the cameras at the start pose)"}
    rep["table"] = table

    # -- colours: object, holders, table (IMX519)
    obj, hold = [], []
    for n, (m, u) in imgs.items():
        if n in EXCLUDE_COLOUR:
            continue
        ym = yellow_mask(m)
        core = cv2.erode(ym.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if core.sum() > 30:
            obj.append(np.median(m[core].astype(float), 0))
        hsv = cv2.cvtColor(m, cv2.COLOR_RGB2HSV)
        hm = sil[ "main"] & (hsv[..., 2] > 40) & (hsv[..., 2] < 150) & (hsv[..., 1] < 60)
        hm[:120] = False                                         # holders: the lower part of the image
        if hm.sum() > 50:
            hold.append(np.median(m[hm].astype(float), 0))
    rep["object_rgb_imx519_224"] = np.round(np.median(obj, 0), 1).tolist()
    rep["holder_rgb_imx519_224"] = np.round(np.median(hold, 0), 1).tolist()

    # table colour in O from the IMX519 (all trials' medians, object + robot masked), plane fit per channel
    (tx0, tx1), (ty0, ty1) = table["x_m"], table["y_m"]
    pts, cols = [], []
    for n, (m, u) in imgs.items():
        bad = sil_d["main"] | cv2.dilate(yellow_mask(m).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) \
            | cv2.dilate((cv2.cvtColor(m, cv2.COLOR_RGB2HSV)[..., 1] > 60).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        bad[:30] = True
        bad[194:] = True
        X, Y = np.meshgrid(np.arange(tx0 + 0.6, tx1 - 0.01, 0.005), np.arange(ty0 + 0.5, ty1 - 0.01, 0.005))
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        v = BD.View("m", geo.project("main"), m, ~bad)
        c, ok = BD.sample(v, P)
        pts.append(P[ok])
        cols.append(c[ok])
    P = np.concatenate(pts)
    C = np.concatenate(cols)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    coef = np.stack([np.linalg.lstsq(A, C[:, k], rcond=None)[0] for k in range(3)], 1)       # (3, 3)
    resid = C - A @ coef
    rep["table_colour_fit"] = {"n_samples": int(len(P)), "rgb_at_O_0.53_-0.05": np.round(np.array([0.53, -0.05, 1]) @ coef, 1).tolist(),
                               "coef_x_y_c_per_channel": np.round(coef, 3).tolist(),
                               "resid_std_rgb": np.round(resid.std(0), 2).tolist(),
                               "seen_x_m": [round(float(P[:, 0].min()), 3), round(float(P[:, 0].max()), 3)],
                               "seen_y_m": [round(float(P[:, 1].min()), 3), round(float(P[:, 1].max()), 3)]}
    cmin, cmax = np.percentile(C, 2, axis=0), np.percentile(C, 98, axis=0)

    sx = (float(P[:, 0].min()), float(P[:, 0].max()))
    sy = (float(P[:, 1].min()), float(P[:, 1].max()))

    def table_rgb(Pq):
        """The plane fit inside the IMX519's footprint, held constant beyond it (no extrapolated gradient)."""
        x = np.clip(Pq[:, 0], *sx)
        y = np.clip(Pq[:, 1], *sy)
        return (np.c_[x, y, np.ones(len(Pq))] @ coef).clip(cmin, cmax)
    mm = 5.0
    w = int(round((tx1 - tx0) * 1000 / mm))
    h = int(round((ty1 - ty0) * 1000 / mm))
    X, Y = np.meshgrid(tx0 + (np.arange(w) + 0.5) * mm / 1000, ty1 - (np.arange(h) + 0.5) * mm / 1000)
    tex = table_rgb(np.stack([X.ravel(), Y.ravel()], 1)).reshape(h, w, 3)
    rng = np.random.default_rng(0)                               # faint paper-like grain (the real table is matte)
    tex = tex + cv2.GaussianBlur(rng.normal(0, 1.2, (h, w)).astype(np.float32), (0, 0), 0.7)[..., None]
    t8 = tex.clip(0, 255).round().astype(np.uint8)
    cv2.imwrite(str(out / "table_robot.png"), t8[:, :, ::-1])
    rep["table_texture"] = {"file": "table_robot.png", "size_m": [w * mm / 1000, h * mm / 1000],
                            "origin_O_m": [tx0, ty1 - h * mm / 1000], "mm_per_px": mm,
                            "mean_rgb": np.round(t8.reshape(-1, 3).mean(0), 1).tolist()}

    # -- IMX219 shading on the robot (fisheye radius), from its table pixels vs the IMX519 table colour
    rr, meas, expct = [], [], []
    for n, (m, u) in imgs.items():
        bad = sil_d["ultrawide"] | cv2.dilate(yellow_mask(u).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        bad |= cv2.dilate((u.max(-1) < 8).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        X, Y = np.meshgrid(np.arange(tx0 + 0.3, tx1 - 0.015, 0.006), np.arange(ty0 + 0.5, ty1 - 0.015, 0.006))
        P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        c, ok = BD.sample(BD.View("u", geo.project("ultrawide"), u, ~bad), P)
        # LED glow / cables on the table: drop pixels far from the white table (saturation)
        hsv = cv2.cvtColor(c.clip(0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        ok &= hsv[:, 1] < 90
        rr.append(geo.fish_radius(P[ok]))
        meas.append(c[ok])
        expct.append(table_rgb(P[ok][:, :2]))
    fit = BD.fit_radial_gain(np.concatenate(rr), np.concatenate(meas), np.concatenate(expct),
                             r_max=float(np.hypot(geo.fish.cx, geo.fish.cy)))
    rep["imx219_gain_robot"] = fit

    # -- backdrop proxies, painted from the source trial
    m, u = imgs[SOURCE_TRIAL]

    def occluded_by_table(cam):
        c0 = cam[:3, 3]

        def f(P):
            d = P - c0
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (0.0 - c0[2]) / d[:, 2]
            hit = c0[None, :2] + t[:, None] * d[:, :2]
            inside = (hit[:, 0] > tx0) & (hit[:, 0] < tx1) & (hit[:, 1] > ty0) & (hit[:, 1] < ty1)
            return (P[:, 2] < -0.001) & (t > 0) & (t < 1) & inside
        return f

    def correct_uw(rgb, uv, _geo=geo):
        return rgb                                              # replaced below (needs the 3D points)

    quads = [
        BD.Quad("floor", np.array([tx1, 1.6, FLOOR_Z]), np.array([1.6 - tx1, 0, 0]) * 0 + np.array([1.4, 0, 0]),
                np.array([0, -2.6, 0]), 5.0),
        BD.Quad("wall_far", np.array([-1.2, y_far + 0.05, 1.6]), np.array([2.2, 0, 0]), np.array([0, 0, -(1.6 - FLOOR_Z)]), 5.0),
        BD.Quad("person_plane", np.array([1.05, 1.6, 1.6]), np.array([0, -2.6, 0]), np.array([0, 0, -(1.6 - FLOOR_Z)]), 5.0),
        BD.Quad("wall_left", np.array([-0.9, -1.0, 1.6]), np.array([0, 2.6, 0]), np.array([0, 0, -(1.6 - FLOOR_Z)]), 6.0),
    ]
    # floor only beyond the table edge and up to the person plane; it starts at the table's right edge
    quads[0] = BD.Quad("floor", np.array([tx1, 1.6, FLOOR_Z]), np.array([1.05 - tx1, 0, 0]), np.array([0, -2.6, 0]), 5.0)
    bad_m = sil_d["main"] | cv2.dilate(yellow_mask(m).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    bad_m[:29] = True
    bad_m[195:] = True
    bad_u = sil_d["ultrawide"] | cv2.dilate(yellow_mask(u).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    bad_u |= cv2.dilate((u.max(-1) < 8).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    class UwView(BD.View):
        pass
    vm = BD.View("main", geo.project("main"), m, ~bad_m, occluded=occluded_by_table(geo.cam["main"]), weight=2.0)
    vu = BD.View("ultrawide", geo.project("ultrawide"), u, ~bad_u, occluded=occluded_by_table(geo.cam["ultrawide"]))
    # the IMX219 correction needs each sample's 3D point: bake per quad with a closure over its points
    texs = {}
    for q in quads:
        Pq = q.points().reshape(-1, 3)
        g = BD.radial_gain(fit, geo.fish_radius(Pq))
        cm, okm = BD.sample(vm, Pq)
        cu, oku = BD.sample(vu, Pq)
        cu = cu / g
        hgt, wid = q.shape
        col = np.where(okm[:, None], cm, np.where(oku[:, None], cu, np.nan))     # IMX519 first, then IMX219
        seen = (okm | oku).reshape(hgt, wid)
        base = np.nanmedian(col[seen.ravel()], 0) if seen.any() else np.array([128, 128, 128.0])
        img = np.where(seen[..., None], np.nan_to_num(col.reshape(hgt, wid, 3)), base).clip(0, 255).astype(np.uint8)
        if seen.any() and not seen.all():
            near = cv2.dilate(seen.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~seen
            img = cv2.inpaint(img, near.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        texs[q.name] = (img, float(seen.mean()))
        cv2.imwrite(str(work / f"quad_{q.name}.png"), img[:, :, ::-1])
    rep["backdrop"] = BD.write_obj(out / "backdrop_robot.obj", out / "backdrop_robot.png", quads, texs,
                                   header="sim_deploy W2 robot look: projective backdrop of the robot cell (O frame)")
    rep["backdrop"]["colour"] = ("IMX519 pixels as they are; IMX219 pixels divided by imx219_gain_robot (fisheye "
                                 "radius): IMX519 colours")
    from sim_deploy.look.make_assets import sha256
    rep["files"] = {f: sha256(out / f) for f in ("table_robot.png", "backdrop_robot.obj", "backdrop_robot.png")}
    (out / "robot_look.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("imx219_gain_robot", "mount_cameras_O")}, indent=1)[:5000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
