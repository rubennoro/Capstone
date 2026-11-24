"""W2 step 1 "look": checks of the look variants (physics equivalence, renders vs the real frames, GL timing, mesh).

    unset PYTHONPATH; cd ~/Capstone-main-sim
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look physics --a sim_deploy/build/sim_rig \
        --b sim_deploy/build/sim_rig_look_real
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look render --station sim_deploy/stations/<id>.yaml \
        [--look NAME] [--only ep14_first ...]        # one look per process (one GL context at a time)
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look compare [--looks current real real_bg]
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look timing --station sim_deploy/stations/<id>.yaml
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look robot_render --station ... --look NAME
    ~/envs/sim_deploy/bin/python -m sim_deploy.look.check_look robot_compare --work DIR [--looks current real robot]

Results: ``sim_evals/_pipeline/W2/look/`` (``render/<look>/<ref>/...``, ``physics_<b>.json``, ``timing_<look>.json``,
``colour_<look>.json``, side-by-side PNGs).

``physics``: both scene.xml compiled with MuJoCo; compared exactly: sizes (nq, nv, nu, nbody, njnt, neq, ntendon,
npair, nexclude, nkey, ...), every body's mass / inertia / pos / quat / ipos / iquat / parent, every joint and dof
field, every COLLIDING geom (contype or conaffinity non-zero, matched by order) type / size / pos / quat / friction /
contype / conaffinity / condim / solref / solimp / solmix / margin / gap / priority / body / mesh vertex data,
contact pairs and excludes (by geom / body names), actuators, tendons, equalities, keyframes, options; then the
same 2000 steps from the home key with the same random controls in both: qpos / qvel bit-identical?

``render``: the sim Pi's own path (``render_sets``: ``Study`` scenes = IK of the recorded TCP, recorded gap, the
eraser where it lay; ``load_handheld_render_model`` / ``load_render_model`` + ``CameraChain`` + ``GLRenderer``, JPEG
q90, the copy's ImagePrep) for the HANDHELD pose of the references; the 640 decoded frames, the 224 images and the
eraser coverage are stored. For a look with a background mesh also: the IMX219 background-only renders (edge
overlay on the real frame, best shift in px) and the MOUNT view at the start pose (q_start, gap 0.040).

``compare``: per look and reference, the colour statistics (mean, std per channel) of the table region and the
eraser core in the real 640 frames (``raw640_*_1.png``) and in the sim's (decoded JPEG), and the side-by-side sheets
real | current | real | real_bg at 224 (real = the deploy path ``deploy_*_1.png``).

``timing``: the sim Pi's per-tick GL (``CameraChain.render_into`` of both cameras, ``GLRenderer.update`` once) at
the home pose with the joints swinging, 60 ticks after 10 warm-up; the CPU part (remap + INTER_AREA) separately;
VRAM of this process (nvidia-smi).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sim_deploy.sim_pi.render as R_  # noqa: F401  (EGL + NVIDIA before mujoco)

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_deploy.sim_pi import _deploy as D  # noqa: E402

cv2.setNumThreads(4)
REPO = D.REPO
OUT = REPO / "sim_evals" / "_pipeline" / "W2" / "look"
REF = REPO / "sim_evals" / "_camera_study" / "B" / "reference"
CHECK_REFS = ("ep14_first", "ep25_approach_2of3", "ep02_first", "ep09_approach_1of3", "ep19_first")
FINGER_LINE = 0.52


def _vram() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    for ln in out.splitlines():
        p = [s.strip() for s in ln.split(",")]
        if p and p[0] == str(os.getpid()):
            return p[1]
    try:                                  # EGL contexts are graphics ("G") processes: only the plain table lists them
        txt = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return "not listed"
    for ln in txt.splitlines():
        f = ln.strip("| ").split()
        if len(f) >= 6 and f[3] == str(os.getpid()):
            return f[-1]
    return "not listed"


# --------------------------------------------------------------------------------------------------- physics


def _arr(x):
    return np.array(x).copy()


def cmd_physics(a) -> dict:
    A = mujoco.MjModel.from_xml_path(str(Path(a.a) / "scene.xml"))
    B = mujoco.MjModel.from_xml_path(str(Path(a.b) / "scene.xml"))
    diffs, same = [], []

    def cmp(name, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            diffs.append(f"{name}: shape {x.shape} vs {y.shape}")
        elif x.dtype.kind in "fc" and not np.array_equal(x, y):
            diffs.append(f"{name}: max |diff| {float(np.max(np.abs(x - y))):.3e}")
        elif x.dtype.kind not in "fc" and not np.array_equal(x, y):
            diffs.append(f"{name}: differs")
        else:
            same.append(name)

    sizes = ("nq", "nv", "nu", "na", "nbody", "njnt", "neq", "ntendon", "nwrap", "npair", "nexclude", "nkey",
             "nsensor", "nsite", "ncam")
    for k in sizes:
        cmp(k, getattr(A, k), getattr(B, k))
    for k in ("body_parentid", "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
              "body_jntnum", "body_jntadr", "body_dofnum", "body_gravcomp", "body_subtreemass", "body_contype",
              "body_conaffinity"):
        cmp(k, getattr(A, k), getattr(B, k))
    for k in [n for n in dir(A) if n.startswith(("jnt_", "dof_"))]:
        cmp(k, getattr(A, k), getattr(B, k))
    ca = [g for g in range(A.ngeom) if A.geom_contype[g] or A.geom_conaffinity[g]]
    cb = [g for g in range(B.ngeom) if B.geom_contype[g] or B.geom_conaffinity[g]]
    cmp("colliding geom count", len(ca), len(cb))
    if len(ca) == len(cb):
        cmp("colliding geom names", [A.geom(g).name for g in ca], [B.geom(g).name for g in cb])
        cmp("colliding geom bodies", [A.body(A.geom_bodyid[g]).name for g in ca], [B.body(B.geom_bodyid[g]).name for g in cb])
        for k in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_contype",
                  "geom_conaffinity", "geom_condim", "geom_solref", "geom_solimp", "geom_solmix", "geom_margin",
                  "geom_gap", "geom_priority", "geom_rbound", "geom_aabb"):
            cmp(k + " (colliding)", getattr(A, k)[ca], getattr(B, k)[cb])
        va, vb = [], []
        for M, ids, out in ((A, ca, va), (B, cb, vb)):
            for g in ids:
                mid = M.geom_dataid[g]
                if M.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0:
                    s, n = M.mesh_vertadr[mid], M.mesh_vertnum[mid]
                    out.append(M.mesh_vert[s:s + n].copy())
        cmp("colliding mesh vertices", np.concatenate(va) if va else [], np.concatenate(vb) if vb else [])
    gn = lambda M, i: M.geom(i).name or f"#{M.body(M.geom_bodyid[i]).name}"  # noqa: E731
    cmp("pairs (geom names)", [(gn(A, A.pair_geom1[i]), gn(A, A.pair_geom2[i])) for i in range(A.npair)],
        [(gn(B, B.pair_geom1[i]), gn(B, B.pair_geom2[i])) for i in range(B.npair)])
    for k in [n for n in dir(A) if n.startswith("pair_") and n not in ("pair_geom1", "pair_geom2", "pair_signature")]:
        cmp(k, getattr(A, k), getattr(B, k))
    cmp("excludes (body names)", [(A.body(s >> 16).name, A.body(s & 0xFFFF).name) for s in A.exclude_signature],
        [(B.body(s >> 16).name, B.body(s & 0xFFFF).name) for s in B.exclude_signature])
    for pre in ("actuator_", "tendon_", "eq_", "key_", "wrap_"):
        for k in [n for n in dir(A) if n.startswith(pre)]:
            v = getattr(A, k)
            if callable(v) or not isinstance(v, np.ndarray):
                continue
            cmp(k, v, getattr(B, k))
    for k in ("timestep", "gravity", "wind", "magnetic", "density", "viscosity", "impratio", "tolerance",
              "ls_tolerance", "noslip_tolerance", "o_margin", "o_solref", "o_solimp", "o_friction", "integrator",
              "cone", "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations", "disableflags",
              "enableflags"):
        cmp("opt." + k, getattr(A.opt, k), getattr(B.opt, k))
    # dynamics: the same 2000 steps from the home key with the same random controls
    rng = np.random.default_rng(0)
    da, db = mujoco.MjData(A), mujoco.MjData(B)
    mujoco.mj_resetDataKeyframe(A, da, 0)
    mujoco.mj_resetDataKeyframe(B, db, 0)
    lo, hi = A.actuator_ctrlrange[:, 0], A.actuator_ctrlrange[:, 1]
    q0 = da.ctrl.copy()
    ctrl0 = np.where(A.actuator_ctrllimited.astype(bool), np.clip(da.qpos[A.actuator_trnid[:, 0].clip(0, A.nq - 1)], lo, hi), 0)
    ncon = []
    for t in range(2000):
        if t % 50 == 0:
            c = ctrl0 + rng.normal(0, 0.05, A.nu)
            c = np.where(A.actuator_ctrllimited.astype(bool), np.clip(c, lo, hi), c)
        da.ctrl[:] = c
        db.ctrl[:] = c
        mujoco.mj_step(A, da)
        mujoco.mj_step(B, db)
        ncon.append((da.ncon, db.ncon))
    del q0
    dyn = {"steps": 2000, "qpos_bit_identical": bool(np.array_equal(da.qpos, db.qpos)),
           "qvel_bit_identical": bool(np.array_equal(da.qvel, db.qvel)),
           "qpos_max_abs_diff": float(np.max(np.abs(da.qpos - db.qpos))),
           "contacts_same_every_step": all(x == y for x, y in ncon), "contacts_last": ncon[-1]}
    vis = {k: [int(getattr(A, k)), int(getattr(B, k))] for k in ("ngeom", "nmesh", "ntex", "nmat", "nlight")}
    extra = [B.geom(g).name for g in range(B.ngeom) if not (B.geom_contype[g] or B.geom_conaffinity[g])
             and B.geom(g).name not in {A.geom(i).name for i in range(A.ngeom)}]
    rep = {"a": str(a.a), "b": str(a.b), "identical": not diffs and dyn["qpos_bit_identical"], "differences": diffs,
           "n_fields_compared_equal": len(same), "dynamics": dyn,
           "visual_only_counts_a_b": vis, "extra_visual_geoms_in_b": extra,
           "a_sha": json.loads((Path(a.a) / "model.json").read_text())["model_sha256"],
           "b_sha": json.loads((Path(a.b) / "model.json").read_text())["model_sha256"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"physics_{Path(a.b).name}.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------------------------------------------- render


def _placement(st) -> dict:
    for p in (REPO / "sim_evals" / "_pipeline" / "W2" / "sets" / "placement.json",
              REPO / "sim_evals" / "_camera_study" / "B" / "sets" / "placement.json", OUT / "placement.json"):
        if p.is_file():
            doc = json.loads(p.read_text())
            doc = doc.get("placement", doc)
            if "episodes" in doc:
                doc["_from"] = str(p)
                return doc
    print("[look] no placement.json: fitting it (render_sets.Study.fit_placement)", flush=True)
    doc = st.fit_placement()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "placement.json").write_text(json.dumps(doc, indent=1))
    doc["_from"] = str(OUT / "placement.json")
    return doc


def cmd_render(a) -> dict:
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.sim_pi.cameras import CameraChain
    from sim_deploy.sim_pi.model_file import load_render_model
    from sim_deploy.sim_pi.render import Lighting

    st = RS.Study(Path(a.station))
    look = a.look or st.look_id
    placement = _placement(st)
    scenes = RS.plan_scenes(st, placement)
    only = set(a.only) if a.only else set(CHECK_REFS)
    chains = {r: CameraChain(r, st.rig, density=st.s.density) for r in RS.ROLES}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    lighting = Lighting(shadows=st.s.shadows)
    out = OUT / "render" / look
    out.mkdir(parents=True, exist_ok=True)
    rep = {"look": look, "station": str(st.station), "model": str(st.s.model), "model_sha256": st.doc["model_sha256"],
           "placement_from": placement.get("_from"), "shadows": st.s.shadows, "refs": {}}
    rm, info = RS.load_handheld_render_model(st.s.model, lighting, W, H, st.s.keep_model_lights, st.tcp_T_cam["E7"])
    rend = RS.Renderer(rm, chains, st.s, st.prep, None)
    rep["vram_handheld"] = _vram()
    bg_id = mujoco.mj_name2id(rm.model, mujoco.mjtObj.mjOBJ_GEOM, "background")
    try:
        for key, sc in scenes.items():
            if key not in only:
                continue
            d = out / key
            d.mkdir(parents=True, exist_ok=True)
            for role in RS.ROLES:
                row = sc["rows"][role][1]
                pr = sc["per_row"][row]
                qpos = RS.qpos_for(st.kin, pr["q"], pr["gap"], sc["eraser_pose"])
                rng = np.random.default_rng(0)
                img, jpg, dec, alpha = rend.frame(role, qpos, rng)
                s = RS.SHORT[role]
                cv2.imwrite(str(d / f"{s}_1.png"), img[:, :, ::-1])
                cv2.imwrite(str(d / f"dec640_{s}_1.png"), dec[:, :, ::-1])
                np.save(d / f"alpha224_{s}_1.npy", alpha.astype(np.float32))
                if bg_id >= 0 and role == "ultrawide":
                    _bg_only(rend, role, qpos, bg_id, d)
            rep["refs"][key] = {"rows": sc["rows"], "eraser_pose": sc["eraser_pose"]}
            print(f"[look] {look} {key}", flush=True)
    finally:
        rend.close()
    if bg_id >= 0 or a.mount_start:
        rm = load_render_model(st.s.model, lighting, W, H, keep_model_lights=st.s.keep_model_lights)
        rend = RS.Renderer(rm, chains, st.s, st.prep, None)
        try:
            d = out / "mount_start"
            d.mkdir(parents=True, exist_ok=True)
            q = rm.home_qpos.copy()
            q[0:7] = st.q_start
            q[7:9] = 0.020
            for role in RS.ROLES:
                img, _, dec, _ = rend.frame(role, q, np.random.default_rng(0))
                cv2.imwrite(str(d / f"{RS.SHORT[role]}.png"), img[:, :, ::-1])
                cv2.imwrite(str(d / f"dec640_{RS.SHORT[role]}.png"), dec[:, :, ::-1])
            _external(rend, q, d / "external.png")
            rep["vram_mount"] = _vram()
        finally:
            rend.close()
    (out / "render.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _bg_only(rend, role, qpos, bg_id, d: Path) -> None:
    """The background mesh alone in this camera (everything else and the skybox off): 640 RGB + coverage."""
    rm, gl = rend.rm, rend.gl
    rm.set_qpos(qpos)
    gl.update(rm.data)
    scn = gl.scn
    saved = []
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        if not (g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == bg_id):
            saved.append((i, float(g.rgba[3])))
            g.rgba[3] = 0.0
    sky = scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]
    scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    try:
        Rw, pw = rm.camera_pose(role)
        rgb = rend._chain_frame(role, Rw, pw) if hasattr(rend, "_chain_frame") else rend.chains[role].frame(gl, Rw, pw)
    finally:
        scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = sky
        for i, al in saved:
            scn.geoms[i].rgba[3] = al
    cv2.imwrite(str(d / f"bgonly640_{RS_short(role)}_1.png"), rgb[:, :, ::-1])


def RS_short(role):
    return {"main": "main", "ultrawide": "uw"}[role]


def _external(rend, q, path: Path) -> None:
    from sim_deploy.sim_pi.render import View
    rm, gl = rend.rm, rend.gl
    rm.set_qpos(q)
    gl.update(rm.data)
    c = rm.model.camera("cam_external").id
    R = rm.data.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    p = rm.data.cam_xpos[c]
    w, h = min(800, gl.max_w), min(600, gl.max_h)
    f = 0.5 * h / np.tan(np.radians(float(rm.model.cam_fovy[c])) / 2)
    img = gl.render(View("ext", w, h, f, (w - 1) / 2, (h - 1) / 2), R, p)
    cv2.imwrite(str(path), img[:, :, ::-1])


# --------------------------------------------------------------------------------------------------- compare


def _eraser_mask(rgb):
    from sim_deploy.look.make_assets import eraser_mask
    return eraser_mask(rgb)


def regions(rgb: np.ndarray, role: str) -> dict:
    """Masks on a 640x480 frame: 'eraser' (colour core, eroded 5 px) and 'table' (IMX519: above the finger line,
    minus the eraser dilated 25 px; IMX219: the central band between the fingers and the horizon, x 200-440,
    y 230-330, minus the eraser)."""
    h, w = rgb.shape[:2]
    em = _eraser_mask(rgb)
    core = cv2.erode(em.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    big = cv2.dilate(em.astype(np.uint8), np.ones((51, 51), np.uint8)).astype(bool)
    t = np.zeros((h, w), bool)
    if role == "main":
        t[:int(FINGER_LINE * h)] = True
    else:
        t[230:330, 200:440] = True
    return {"eraser": core, "table": t & ~big}


def _stats(rgb, m):
    if m.sum() < 30:
        return None
    v = rgb[m].astype(np.float64)
    return {"n": int(m.sum()), "mean": np.round(v.mean(0), 1).tolist(), "std": np.round(v.std(0), 1).tolist()}


def cmd_compare(a) -> dict:
    looks = a.looks
    rep = {}
    rows = []
    for key in CHECK_REFS:
        rref = REF / key
        if not (OUT / "render" / looks[0] / key).is_dir():
            continue
        cells = {}
        for role, s in (("main", "main"), ("ultrawide", "uw")):
            real = cv2.cvtColor(cv2.imread(str(rref / f"raw640_{s}_1.png")), cv2.COLOR_BGR2RGB)
            M = regions(real, role)
            ent = {"_real": {k: _stats(real, m) for k, m in M.items()}}
            for lk in looks:
                f = OUT / "render" / lk / key / f"dec640_{s}_1.png"
                if not f.is_file():
                    continue
                sim = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                Ms = regions(sim, role)
                ent[lk] = {k: _stats(sim, m) for k, m in Ms.items()}
            cells[role] = ent
        rep[key] = cells
        # side-by-side at 224: real (deploy path) | looks
        tiles = []
        for s in ("main", "uw"):
            row = [cv2.imread(str(rref / f"deploy_{s}_1.png"))]
            for lk in looks:
                f = OUT / "render" / lk / key / f"{s}_1.png"
                row.append(cv2.imread(str(f)) if f.is_file() else np.zeros((224, 224, 3), np.uint8))
            tiles.append(np.concatenate(row, 1))
        sheet = np.concatenate(tiles, 0)
        lab = np.zeros((22, sheet.shape[1], 3), np.uint8)
        for j, name in enumerate(["real"] + list(looks)):
            cv2.putText(lab, name, (j * 224 + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        sheet = np.concatenate([lab, sheet], 0)
        cv2.imwrite(str(OUT / f"side_by_side_{key}.png"), sheet)
        rows.append(sheet)
    if rows:
        cv2.imwrite(str(OUT / "side_by_side_all.png"), np.concatenate(rows, 0))
    # summary: per look, mean over refs of (sim - real) mean and std ratio
    summ = {}
    for lk in looks:
        for role in ("main", "ultrawide"):
            for reg in ("table", "eraser"):
                dm, ds = [], []
                for key, cells in rep.items():
                    r, s = cells[role]["_real"].get(reg), cells[role].get(lk, {}).get(reg)
                    if r and s:
                        dm.append(np.subtract(s["mean"], r["mean"]))
                        ds.append(np.subtract(s["std"], r["std"]))
                if dm:
                    summ.setdefault(lk, {}).setdefault(role, {})[reg] = {
                        "n_refs": len(dm), "mean_sim_minus_real": np.round(np.mean(dm, 0), 1).tolist(),
                        "std_sim_minus_real": np.round(np.mean(ds, 0), 1).tolist(),
                        "abs_mean_diff_per_ref_max": round(float(np.max(np.abs(dm))), 1)}
    doc = {"note": "640x480: real = raw640_*_1.png (the frames the Pi would stream, pre-JPEG); sim = the sim Pi's "
                   "decoded JPEG q90 at the handheld pose; regions per check_look.regions", "per_ref": rep,
           "summary": summ}
    (OUT / "colour_compare.json").write_text(json.dumps(doc, indent=1))
    return summ


# ------------------------------------------------------------------------------------------------- mesh edges


def cmd_edges(a) -> dict:
    """Edge overlay of the background-only render on the real IMX219 640 frame; best integer shift (px) of the
    rendered edges onto the real edges (mean truncated distance), in the region the mesh covers above the table."""
    look = a.look
    rep = {}
    for key in CHECK_REFS:
        f = OUT / "render" / look / key / "bgonly640_uw_1.png"
        if not f.is_file():
            continue
        bg = cv2.imread(str(f))
        real = cv2.imread(str(REF / key / "raw640_uw_1.png"))
        cov = cv2.erode((bg.max(-1) > 0).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        sim_full = cv2.imread(str(OUT / "render" / look / key / "dec640_uw_1.png"))
        eg = lambda im: cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.5), 40, 100) > 0  # noqa: E731
        e_sim = eg(bg) & cov
        e_real = eg(real)
        # exclude the fingers / holders area (bottom centre) from the real edges
        dist = cv2.distanceTransform((~e_real).astype(np.uint8), cv2.DIST_L2, 5)
        dist = np.minimum(dist, 15.0)
        ys, xs = np.nonzero(e_sim)
        best, zero = None, None
        for dy in range(-20, 21):
            for dx in range(-20, 21):
                yy, xx = ys + dy, xs + dx
                ok = (yy >= 0) & (yy < 480) & (xx >= 0) & (xx < 640)
                sc = float(dist[yy[ok], xx[ok]].mean()) if ok.any() else 99
                if dx == 0 and dy == 0:
                    zero = sc
                if best is None or sc < best[0]:
                    best = (sc, dx, dy)
        ov = real.copy()
        ov[e_sim] = (255, 0, 255)
        ov2 = real.copy()
        ov2[cov] = (0.5 * real[cov] + 0.5 * bg[cov]).astype(np.uint8)
        cv2.imwrite(str(OUT / f"bg_edges_{key}.png"), np.concatenate([ov, ov2, sim_full], 1))
        rep[key] = {"n_edge_px": int(len(ys)), "mean_dist_px_at_zero_shift": round(zero, 2),
                    "best_shift_dxdy_px": [best[1], best[2]], "mean_dist_px_at_best": round(best[0], 2),
                    "coverage_frac": round(float(cov.mean()), 3)}
    (OUT / f"bg_edges_{look}.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------------------------------------------- timing


def cmd_timing(a) -> dict:
    from sim_deploy.sim_pi import settings as SPS
    from sim_deploy.sim_pi.cameras import CameraChain
    from sim_deploy.sim_pi.model_file import load_render_model
    from sim_deploy.sim_pi.render import GLRenderer, Lighting

    s = SPS.load(Path(a.station), {})
    rig = D.load_live_rig(s.rig_file)
    chains = {r: CameraChain(r, rig, density=s.density) for r in ("main", "ultrawide")}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    lighting = Lighting(shadows=s.shadows)
    t0 = time.perf_counter()
    rm = load_render_model(s.model, lighting, W, H, keep_model_lights=s.keep_model_lights)
    t_load = time.perf_counter() - t0
    for r, c in chains.items():
        c.set_hide([rm.own_board[r]])
    gl = GLRenderer(rm.model, W, H)
    gl.set_shadows(s.shadows)
    bufs = {r: c.new_buffer() for r, c in chains.items()}
    gl_ms, cpu_ms, per = [], [], {"main": [], "ultrawide": []}
    from OpenGL import GL
    try:
        q0 = rm.home_qpos.copy()
        for k in range(70):
            q = q0.copy()
            q[0] += 0.3 * np.sin(k / 10)
            q[4] += 0.3 * np.sin(k / 7)
            rm.set_qpos(q)
            t1 = time.perf_counter()
            gl.update(rm.data)
            tt = {}
            for r, c in chains.items():
                ta = time.perf_counter()
                Rw, pw = rm.camera_pose(r)
                c.render_into(gl, Rw, pw, bufs[r])
                GL.glFinish()
                tt[r] = time.perf_counter() - ta
            t2 = time.perf_counter()
            for r, c in chains.items():
                c.stream_from(bufs[r])
            t3 = time.perf_counter()
            if k >= 10:
                gl_ms.append((t2 - t1) * 1e3)
                cpu_ms.append((t3 - t2) * 1e3)
                for r in tt:
                    per[r].append(tt[r] * 1e3)
        vram = _vram()
    finally:
        gl.close()
    rep = {"station": str(a.station), "model": str(s.model), "model_sha256": rm.sha256, "shadows": s.shadows,
           "load_render_model_s": round(t_load, 2), "gl_renderer": gl.gl_renderer,
           "gl_per_tick_ms": {"mean": round(float(np.mean(gl_ms)), 2), "p95": round(float(np.percentile(gl_ms, 95)), 2)},
           "gl_main_ms": round(float(np.mean(per["main"])), 2), "gl_ultrawide_ms": round(float(np.mean(per["ultrawide"])), 2),
           "cpu_remap_area_ms_both": round(float(np.mean(cpu_ms)), 2), "vram": vram, "ticks": len(gl_ms),
           "note": "GL = GLRenderer.update + render_into (both cameras) with glFinish; the live sim Pi overlaps CPU "
                   "encode with GL (DESIGN E5: ~23 ms GL per tick for sim_rig)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"timing_{Path(s.model).parent.name}.json").write_text(json.dumps(rep, indent=1))
    return rep


# ------------------------------------------------------------------------------------- robot cell (mount)

ROBOT_FIT = REPO / "sim_evals" / "_pipeline" / "W2" / "sets" / "robot_trials_fit.json"


def cmd_robot_render(a) -> dict:
    """The MOUNT view at the start pose (q_start, gap 0.040) with the eraser where step 3 fitted it in each real
    trial (``robot_trials_fit.json`` eraser_pose_O), through the sim Pi path at 224: ``robot/<look>/<trial>_{main,
    uw}.png``."""
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.sim_pi import settings as SPS
    from sim_deploy.sim_pi.cameras import CameraChain
    from sim_deploy.sim_pi.model_file import load_render_model
    from sim_deploy.sim_pi.render import Lighting

    fit = json.loads(ROBOT_FIT.read_text())
    s = SPS.load(Path(a.station), {})
    look = a.look or Path(s.model).parent.name
    rig = D.load_live_rig(s.rig_file)
    prep = D.ImagePrep(rig)
    chains = {r: CameraChain(r, rig, density=s.density) for r in ("main", "ultrawide")}
    W = max(c.max_view[0] for c in chains.values())
    H = max(c.max_view[1] for c in chains.values())
    rm = load_render_model(s.model, Lighting(shadows=s.shadows), W, H, keep_model_lights=s.keep_model_lights)
    rend = RS.Renderer(rm, chains, s, prep, None)
    out = OUT / "robot" / look
    out.mkdir(parents=True, exist_ok=True)
    rep = {"look": look, "model_sha256": rm.sha256, "vram": None, "trials": {}}
    lay = rm.doc["qpos_layout"]
    try:
        for tid, t in fit["trials"].items():
            q = rm.home_qpos.copy()
            q[slice(*lay["arm"])] = fit["q_start_rad"]
            q[slice(*lay["fingers"])] = fit["gap_m"] / 2
            q[slice(*lay["eraser"])] = t["eraser_pose_O"]
            for role in ("main", "ultrawide"):
                img, _, _, _ = rend.frame(role, q, np.random.default_rng(0))
                cv2.imwrite(str(out / f"{tid}_{RS.SHORT[role]}.png"), img[:, :, ::-1])
            rep["trials"][tid] = {"folder": t["folder"], "eraser_pose_O": t["eraser_pose_O"]}
        rep["vram"] = _vram()
    finally:
        rend.close()
    (out / "render.json").write_text(json.dumps(rep, indent=1))
    return rep


def _robot_masks():
    """224 masks per camera from the geometry: 'robot' (the sim silhouette at the start pose, make_robot), 'table'
    (the real table's top, robot_look.json extent, projected), 'background' (the rest of the image content)."""
    from sim_deploy.camera_study import render_sets as RS
    from sim_deploy.look.make_robot import Geo
    rl = json.loads((REPO / "sim_deploy" / "look" / "assets" / "robot_look.json").read_text())
    (x0, x1), (y0, y1) = rl["table"]["x_m"], rl["table"]["y_m"]
    geo = Geo()
    sil_dir = Path(rl.get("_work", "")) if rl.get("_work") else None
    out = {}
    X, Y = np.meshgrid(np.arange(x0, x1, 0.002), np.arange(y0, y1, 0.002))
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    for role, s in (("main", "main"), ("ultrawide", "uw")):
        uv, ok = geo.project(role)(P)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < 224) & (uv[:, 1] >= 0) & (uv[:, 1] < 224)
        t = np.zeros((224, 224), np.uint8)
        t[np.round(uv[ok, 1]).astype(int).clip(0, 223), np.round(uv[ok, 0]).astype(int).clip(0, 223)] = 1
        t = cv2.morphologyEx(t, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        out[role] = {"table": t}
    del sil_dir, RS
    return out


def cmd_robot_compare(a) -> dict:
    fit = json.loads(ROBOT_FIT.read_text())
    sil_dir = Path(a.work) / "robot"
    masks = _robot_masks()
    for role, s in (("main", "main"), ("ultrawide", "uw")):
        sil = cv2.imread(str(sil_dir / f"silhouette_{role}.png"), 0) > 0
        sil = cv2.dilate(sil.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        masks[role]["robot"] = sil
    rep, sheets = {}, []
    for tid, t in fit["trials"].items():
        folder = Path(t["folder"])
        cells = {}
        rows = []
        for role, s, fs in (("main", "main", "main"), ("ultrawide", "uw", "ultrawide")):
            st = np.stack([cv2.imread(str(folder / f"{i:06d}_{fs}_1.png")) for i in range(40)])
            real = cv2.cvtColor(np.median(st, 0).astype(np.uint8), cv2.COLOR_BGR2RGB)
            content = cv2.erode((real.max(-1) > 6).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            from sim_deploy.look.make_robot import yellow_mask
            er = yellow_mask(real)
            er_core = cv2.erode(er.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            er_big = cv2.dilate(er.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            M = masks[role]
            regs = {"table": M["table"] & ~M["robot"] & ~er_big & content,
                    "background": ~M["table"] & ~M["robot"] & ~er_big & content,
                    "eraser": er_core, "image": content}
            ent = {"_real": {k: _stats(real, m) for k, m in regs.items()}}
            dbg = np.zeros((224, 224, 3), np.uint8)
            dbg[regs["table"]] = (0, 255, 0)
            dbg[regs["background"]] = (255, 0, 0)
            dbg[regs["eraser"]] = (0, 255, 255)
            cv2.imwrite(str(OUT / "robot" / f"regions_{s}.png"), dbg)
            row = [real[:, :, ::-1]]
            for lk in a.looks:
                f = OUT / "robot" / lk / f"{tid}_{s}.png"
                if not f.is_file():
                    row.append(np.zeros((224, 224, 3), np.uint8))
                    continue
                sim = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
                e = {k: _stats(sim, m) for k, m in regs.items() if k != "eraser"}
                es = cv2.erode(yellow_mask(sim).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) \
                    if yellow_mask(sim).any() else regs["eraser"]
                e["eraser"] = _stats(sim, es) if es.sum() >= 30 else _stats(sim, regs["eraser"])
                e["mad_image"] = round(float(np.abs(sim.astype(float) - real.astype(float))[content].mean()), 2)
                e["mad_background"] = round(float(np.abs(sim.astype(float) - real.astype(float))[regs["background"]].mean()), 2)
                ent[lk] = e
                row.append(sim[:, :, ::-1])
            cells[role] = ent
            rows.append(np.concatenate(row, 1))
        rep[tid] = cells
        sheet = np.concatenate(rows, 0)
        lab = np.zeros((20, sheet.shape[1], 3), np.uint8)
        for j, name in enumerate(["real " + tid[-15:]] + list(a.looks)):
            cv2.putText(lab, name, (j * 224 + 3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        sheets.append(np.concatenate([lab, sheet], 0))
    cv2.imwrite(str(OUT / "robot_side_by_side.png"), np.concatenate(sheets, 0))
    summ = {}
    for lk in a.looks:
        for role in ("main", "ultrawide"):
            for reg in ("table", "background", "eraser"):
                dm = [np.subtract(c[role][lk][reg]["mean"], c[role]["_real"][reg]["mean"]) for c in rep.values()
                      if lk in c[role] and c[role][lk].get(reg) and c[role]["_real"].get(reg)]
                ds = [np.subtract(c[role][lk][reg]["std"], c[role]["_real"][reg]["std"]) for c in rep.values()
                      if lk in c[role] and c[role][lk].get(reg) and c[role]["_real"].get(reg)]
                if dm:
                    summ.setdefault(lk, {}).setdefault(role, {})[reg] = {
                        "n": len(dm), "mean_sim_minus_real": np.round(np.mean(dm, 0), 1).tolist(),
                        "std_sim_minus_real": np.round(np.mean(ds, 0), 1).tolist()}
            mads = [c[role][lk]["mad_image"] for c in rep.values() if lk in c[role]]
            madb = [c[role][lk]["mad_background"] for c in rep.values() if lk in c[role]]
            if mads:
                summ[lk][role]["mad_image_mean"] = round(float(np.mean(mads)), 2)
                summ[lk][role]["mad_background_mean"] = round(float(np.mean(madb)), 2)
    doc = {"note": "224: real = median of the first 40 sets of each trial (static, arm at the start pose); sim = "
                   "the look's mount render at q_start, gap 0.040, eraser at step 3's fitted pose (the look station's box: 127 x 30 x 12 since 2026-11-21, 127 x 51 before); "
                   "regions from the geometry (robot_look.json table extent, sim silhouette)", "per_trial": rep,
           "summary": summ}
    (OUT / "robot_compare.json").write_text(json.dumps(doc, indent=1))
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("physics")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p = sub.add_parser("render")
    p.add_argument("--station", required=True)
    p.add_argument("--look")
    p.add_argument("--only", nargs="*")
    p.add_argument("--mount-start", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("--looks", nargs="+", default=["current", "real", "real_bg"])
    p = sub.add_parser("edges")
    p.add_argument("--look", default="real_bg")
    p = sub.add_parser("timing")
    p.add_argument("--station", required=True)
    p = sub.add_parser("robot_render")
    p.add_argument("--station", required=True)
    p.add_argument("--look")
    p = sub.add_parser("robot_compare")
    p.add_argument("--work", required=True, help="make_robot's --work (silhouettes)")
    p.add_argument("--looks", nargs="+", default=["current", "real", "robot"])
    a = ap.parse_args(argv)
    rep = {"physics": cmd_physics, "render": cmd_render, "compare": cmd_compare, "edges": cmd_edges,
           "timing": cmd_timing, "robot_render": cmd_robot_render, "robot_compare": cmd_robot_compare}[a.cmd](a)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
