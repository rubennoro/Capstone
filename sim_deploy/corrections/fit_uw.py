"""Camera study B4 ``corr_uw`` (sim only): the IMX219 view handheld vs Franka mount, and a rig-file correction of the
virtual ultrawide camera (rotation + intrinsics) fitted so that the MOUNT view through the corrected virtual camera
matches the HANDHELD training view through the original one.

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_deploy/bin/python -m sim_deploy.corrections.fit_uw [--write] [--no-png]

CPU only (projection math, MuJoCo kinematics, OpenCV on synthetic images; no rendering, no GL, no ports). Steps:

1. camera poses in the fingertip TCP frame, two methods for the mount (sim model vs DESIGN.md E2's formulas);
2. point sets (``geometry.point_sets``): tags, fingertip pads, holders (handheld kit vs the model's fingers at
   gap = width, their difference in mm, cross-checked with ``handheld_check.HandheldScene`` / ``tag_corners_3d``),
   table near/mid/far and the eraser per reference of ``layout.json``; evidence for a table offset from tag z 0;
3. projection to the policy's 224 image; handheld vs mount offsets per class, scale ratios; a third check of the
   handheld projection against the real training images (ArUco corners detected in the packaged 224 frames, when the
   exports ``ref_ep<i>.npz`` from ``sim_pi/handheld_reference.py`` are in the scratch folder);
4. least-squares fits (several parameterisations x point weightings), residuals per class before/after, the black
   border each one introduces; ``--write`` writes ``rig_pi_v3_640_corr.yaml`` (+ ``_corr_ft.yaml``, the
   fingers+tags fit) next to this file and the variant station ``stations/policy_sim_uwcorr.yaml``;
5. checks of the written files with the copy's own code: ``load_live_rig`` + ``ImagePrep`` use the keys, an
   end-to-end synthetic fisheye (dots at known rays) through ``prep_ultrawide`` vs the projection math, the
   launcher's rig-file check and the frames ``Settings.load`` of the variant station (subprocesses in the laptop env).

Outputs: ``sim_evals/_camera_study/B/corr_uw/fit.json`` and ``overlay_<ref>.png``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from sim_deploy.corrections import geometry as G
from sim_deploy.sim_pi import _deploy as D

cv2.setNumThreads(4)
REPO = G.REPO
OUT = REPO / "sim_evals" / "_camera_study" / "B" / "corr_uw"
SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad/campB/corr_uw")
HERE = Path(__file__).resolve().parent
CORR_RIG = HERE / "rig_pi_v3_640_corr.yaml"
CORR_RIG_FT = HERE / "rig_pi_v3_640_corr_ft.yaml"
STATION_SRC = REPO / "sim_deploy" / "stations" / "policy_sim.yaml"
STATION_OUT = REPO / "sim_deploy" / "stations" / "policy_sim_uwcorr.yaml"
PD_DIR = REPO / "deployment" / "policy_deploy"
PD_PY = Path("~/envs/sim_policy_deploy/bin/python").expanduser()
CROP_KEEP = int(224 * 0.95)                       # 212: training's CenterCrop / RandomCrop size
CROP_MARGIN = (224 - CROP_KEEP) / 2               # 6 px per side
CONTENT_ROWS = (28, 196)                          # 640x480 letterboxed into 224: rows 28..195 are the image
PARAMS = {"f": ("s",), "f+c": ("s", "cx", "cy"), "rot": ("rx", "ry", "rz"), "rot+f": ("rx", "ry", "rz", "s"),
          "rot+f+c": ("rx", "ry", "rz", "s", "cx", "cy"), "rot+fxfy+c": ("rx", "ry", "rz", "sx", "sy", "cx", "cy")}
X_SCALE = {"rx": 1.0, "ry": 1.0, "rz": 1.0, "s": 0.05, "sx": 0.05, "sy": 0.05, "cx": 5.0, "cy": 5.0}
FITSETS = {"all": dict.fromkeys(("tags", "fingertips", "holders", "table_near", "table_mid", "table_far"), 1.0),
           "fingers_tags": dict.fromkeys(("tags", "fingertips", "holders"), 1.0),
           "table": dict.fromkeys(("table_near", "table_mid", "table_far"), 1.0)}
PRIMARY = ("all", "rot+f")                        # written to rig_pi_v3_640_corr.yaml
SECONDARY = ("fingers_tags", "rot+f")             # written to rig_pi_v3_640_corr_ft.yaml
EVIDENCE_Z_NOTE = "table z from the grasp rows (fingertip TCP at the table when pinching the 12 mm eraser)"


def r3(x, n=3):
    return np.round(np.asarray(x, float), n).tolist()


def stats(d: np.ndarray) -> dict:
    d = np.asarray(d, float).reshape(-1, 2)
    if not len(d):
        return {"n": 0}
    e = np.linalg.norm(d, axis=1)
    return {"n": int(len(e)), "mean_px": round(float(e.mean()), 2), "p90_px": round(float(np.quantile(e, 0.9)), 2),
            "max_px": round(float(e.max()), 2), "mean_dxdy_px": r3(d.mean(0), 2)}


# ------------------------------------------------------------------------------------------- 1. camera poses

def camera_section(rig) -> dict:
    M = G.model()
    cams = {k: G.camera_poses_tcp(k, rig) for k in ("handheld", "handheld_nominal", "mount")}
    T_hand_tcp = G.inv(M.body_world("hand")) @ M.site_world("tcp")
    z_tip = float(T_hand_tcp[2, 3] * 1e3)
    out = {"frame_note": "T_tcp_cam: camera pose (OpenCV axes) in the fingertip TCP frame (x closing axis = image "
                         "right, y = away from the camera side, z = toward the fingertips); offsets 'in_a_axes' are "
                         "in the handheld camera's own axes (x right, y down, z optical axis)",
           "E7_camera_dt_mm_imx219_axes": r3(G.E7_CAMERA_DT_M * 1e3, 2), "hand_T_tcp_z_tip_mm": round(z_tip, 3),
           "hand_T_tcp_xy_mm": r3(T_hand_tcp[:2, 3] * 1e3, 4)}
    for role in G.ROLES:
        e2 = G.inv(G.e2_formula_T_cam_tcp(role, z_tip))
        sec = {k: {"T_tcp_cam_t_mm": r3(v[role][:3, 3] * 1e3, 2), "T_cam_tcp_t_mm": r3(G.inv(v[role])[:3, 3] * 1e3, 2),
                   "T_cam_tcp_euler_xyz_deg": r3(Rotation.from_matrix(G.inv(v[role])[:3, :3]).as_euler(
                       "xyz", degrees=True), 4)} for k, v in cams.items()}
        sec["mount_method2_E2_formula"] = {"T_cam_tcp_t_mm": r3(G.inv(e2)[:3, 3] * 1e3, 2),
                                          "vs_model": {k: (round(v, 4) if isinstance(v, float) else r3(v, 4))
                                                       for k, v in G.pose_diff(cams["mount"][role], e2).items()}}
        for ref in ("handheld", "handheld_nominal"):
            d = G.pose_diff(cams[ref][role], cams["mount"][role])
            sec[f"mount_vs_{ref}"] = {k: (round(v, 4) if isinstance(v, float) else r3(v, 3)) for k, v in d.items()}
        out[role] = sec
    return out


# ------------------------------------------------------------------------------------------- 2. point sets

def finger_model_section(rig) -> dict:
    """Handheld kit vs the sim model's fingers (TCP frame), cross-checks with handheld_check and the compiled mesh."""
    from scipy.spatial import cKDTree
    out = {"gap_mapping": G.model_gap_fn()[2]}
    diffs = {}
    for w in (0.0, 0.052, 0.087):
        h, m = G.finger_points("handheld", w), G.finger_points("mount", w)
        diffs[f"width_{w * 1e3:.0f}mm"] = {k: {"mean_mm": round(float(np.linalg.norm(h[k] - m[k], axis=1).mean() * 1e3), 3),
                                               "max_mm": round(float(np.linalg.norm(h[k] - m[k], axis=1).max() * 1e3), 3),
                                               "mean_vec_mount_minus_handheld_mm": r3((m[k] - h[k]).mean(0) * 1e3, 3)}
                                           for k in h}
        diffs[f"width_{w * 1e3:.0f}mm"]["tip_gap_mm"] = {"handheld": round(float(np.ptp(h["tips"][:, 0]) * 1e3), 3),
                                                         "mount": round(float(np.ptp(m["tips"][:, 0]) * 1e3), 3)}
    out["handheld_kit_vs_model"] = diffs
    # cross-check 1: handheld_check's HandheldScene (the G3 scene, E7 variant) and tag_corners_3d
    import sim_deploy.sim_pi.handheld_check as HC
    chains = {"ultrawide": SimpleNamespace(max_view=(640, 480))}
    sc = HC.HandheldScene(rig, chains, SCRATCH / "hc_textures", tag_outward_m=HC.TAG_OUTWARD_M)
    T = G.T_tcp_asm()
    cc = {}
    for w in (0.0, 0.087):
        sc.set_tag_distance(HC.E7_TAG_DIST(w))
        mine = G.finger_points("handheld", w)
        tips = G.apply(T, sc.tips_world())
        corners = []
        for tid in (0, 1):
            g = sc.d.geom(f"tag_{tid}")
            R = g.xmat.reshape(3, 3)
            h = G.MARKER_M / 2
            c = [g.xpos + R @ [sx * h, sy * h, 0] for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))]
            corners.append(G.apply(T, np.array(c)))
        corners = np.concatenate(corners)
        tags_hc = {t.id: t for s in sc.kit.sides.values() for t in s.tags}
        c3d = np.concatenate([G.apply(T, HC.tag_corners_3d(tags_hc[i], HC.E7_TAG_DIST(w), G.MARKER_M)) for i in (0, 1)])
        cc[f"width_{w * 1e3:.0f}mm"] = {
            "tips_vs_HandheldScene_mm": round(float(np.abs(tips - mine["tips"]).max() * 1e3), 4),
            "tag_corners_vs_HandheldScene_plane_mm": round(float(np.abs(corners - mine["tags"]).max() * 1e3), 4),
            "tag_corners_vs_tag_corners_3d_mm": round(float(np.abs(c3d - mine["tags"]).max() * 1e3), 4)}
    out["crosscheck_handheld_check"] = cc
    out["crosscheck_note"] = ("tag_corners_3d is the holder surface (no lift); HandheldScene's tag plane and geometry's "
                              "handheld tags are 0.2 mm above it; the model's tag slab face is 0.4 mm above")
    # cross-check 2: the compiled model's pad / holder meshes vs geometry's placement of the kit's asm vertices
    M = G.model()
    _, q_of_gap, _ = G.model_gap_fn()
    q = q_of_gap(0.052)
    M.set_fingers(q, q)
    T_tcp_O = G.inv(M.site_world("tcp"))
    K = G.kit()
    mesh = {}
    for body, (kit_side, pad, holder, _a, _t) in G._MODEL_SIDES.items():
        for gname, part in ((pad, K.sides[kit_side].parts[0]), (holder, K.sides[kit_side].parts[1])):
            gid = M.m.geom(gname).id
            mid = M.m.geom_dataid[gid]
            v = M.m.mesh_vert[M.m.mesh_vertadr[mid]:M.m.mesh_vertadr[mid] + M.m.mesh_vertnum[mid]]
            g = M.d.geom(gname)
            Pw = v @ g.xmat.reshape(3, 3).T + g.xpos
            P_model = G.apply(T_tcp_O, Pw)
            P_mine = G.apply(T_tcp_O @ M.body_world(body) @ M.geom_local(gname), K.parts[part].vertices)
            dist, _ = cKDTree(P_model).query(P_mine)
            mesh[gname] = {"vertices": int(len(P_mine)), "max_nn_mm": round(float(dist.max() * 1e3), 5)}
    out["crosscheck_compiled_mesh"] = mesh
    return out


def table_evidence() -> dict:
    """Fingertip TCP z (tag frame) at the hold row of every episode and its episode minimum."""
    from sim_deploy.scene.zarr_lowdim import n_episodes
    n = n_episodes(G.layout()["dataset"])
    zg, zmin = [], []
    for i in range(n):
        ep = G._episode(i)
        zg.append(ep["poses"][ep["grasp"]["end"], 2, 3])
        zmin.append(ep["poses"][:, 2, 3].min())
    zg, zmin = np.array(zg) * 1e3, np.array(zmin) * 1e3
    return {"episodes": n, "tcp_z_at_hold_row_mm": {"median": round(float(np.median(zg)), 2), "min": round(float(zg.min()), 2),
                                                    "max": round(float(zg.max()), 2)},
            "tcp_z_episode_min_mm": {"median": round(float(np.median(zmin)), 2), "min": round(float(zmin.min()), 2),
                                     "max": round(float(zmin.max()), 2)},
            "note": "the fingertip TCP is the distal end of the soft finger on its gripping face; pinching a 12 mm "
                    "eraser lying flat puts it at or just above the table. A hold-row z below 0 means the table top is "
                    "below tag z 0 (or the poses carry a z offset): evidence for a table at about the median hold z."}


# ------------------------------------------------------------------------------------------- 3. projections

def collect(rig, target_kind="handheld", table_z_m=0.0, shared_fingers=False) -> list[dict]:
    """Per reference and class: the target 224 points (handheld view through the original virtual camera) and the
    same physical points in the MOUNT camera frame. Kept: inside the target 224 view and in the mount's fisheye
    coverage."""
    cams = {k: G.camera_poses_tcp(k, rig)["ultrawide"] for k in ("handheld", "handheld_nominal", "mount")}
    rows = []
    for ref in G.load_refs():
        ph = G.point_sets(ref, "handheld", rig, table_z_m)
        pm = ph if shared_fingers else G.point_sets(ref, "mount", rig, table_z_m)
        for c in G.CLASSES:
            uv_t, ok_t = G.project("ultrawide", cams[target_kind], ph[c], rig)
            _, ok_m, P_cam = G.project_native("ultrawide", cams["mount"], pm[c], rig)
            ok = ok_t & ok_m
            rows.append({"ref": ref["name"], "cls": c, "target": uv_t[ok], "P_cam": P_cam[ok]})
    return rows


def unpack(names, x, rig) -> tuple[np.ndarray, np.ndarray, list]:
    p = {"rx": 0.0, "ry": 0.0, "rz": 0.0, "s": 1.0, "sx": None, "sy": None, "cx": 0.0, "cy": 0.0}
    p.update(dict(zip(names, map(float, x))))
    fx0, fy0, cx0, cy0 = map(float, rig.virtual_ultrawide.intrinsics)
    euler = [p["rx"], p["ry"], p["rz"]]
    R = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    K = np.array([fx0 * (p["sx"] if p["sx"] is not None else p["s"]), fy0 * (p["sy"] if p["sy"] is not None else p["s"]),
                  cx0 + p["cx"], cy0 + p["cy"]])
    return R, K, euler


def proj_corr(P_cam: np.ndarray, R: np.ndarray, K) -> np.ndarray:
    """Physical-camera points -> 224 px through a virtual pinhole (R, K): UltrawidePrep's math, no bounds check."""
    d = P_cam @ R                                         # d_virt = R^T d_phys
    uv = np.stack([K[0] * d[:, 0] / d[:, 2] + K[2], K[1] * d[:, 1] / d[:, 2] + K[3]], axis=1)
    return D.letterbox_point(uv, (640, 480))


def fit(rows, rig, param: str, weights: dict) -> dict:
    names = PARAMS[param]
    x0 = np.array([1.0 if n in ("s", "sx", "sy") else 0.0 for n in names])
    sel = [r for r in rows if r["cls"] in weights and len(r["target"])]
    ncls = {c: sum(len(r["target"]) for r in sel if r["cls"] == c) for c in weights}
    P = np.concatenate([r["P_cam"] for r in sel])
    T = np.concatenate([r["target"] for r in sel])
    W = np.concatenate([np.full(len(r["target"]), np.sqrt(weights[r["cls"]] / ncls[r["cls"]])) for r in sel])

    def res(x):
        R, K, _ = unpack(names, x, rig)
        return ((proj_corr(P, R, K) - T) * W[:, None]).ravel()

    sol = least_squares(res, x0, x_scale=np.array([X_SCALE[n] for n in names]), method="lm")
    R, K, euler = unpack(names, sol.x, rig)
    J = sol.jac
    try:
        cond = float(np.linalg.cond(J / np.array([1.0 / X_SCALE[n] for n in names])[None, :]))
    except np.linalg.LinAlgError:
        cond = float("inf")
    return {"param": param, "weights": weights, "points_per_class": ncls, "x": dict(zip(names, r3(sol.x, 5))),
            "euler_xyz_deg": r3(euler, 4), "intrinsics": r3(K, 4), "R": R, "K": K,
            "zoom_vs_original": round(float(K[0] / float(rig.virtual_ultrawide.intrinsics[0])), 5),
            "weighted_rms_px": round(float(np.sqrt(np.sum(sol.fun ** 2) / sum(weights.values()) / 2)), 3),
            "jacobian_condition": round(cond, 1)}


def residuals(rows, R=None, K=None) -> dict:
    """Per class over all references: mount through (R, K) minus target. None = the original virtual camera."""
    R = np.eye(3) if R is None else R
    out = {}
    for c in G.CLASSES:
        d = [proj_corr(r["P_cam"], R, K) - r["target"] for r in rows if r["cls"] == c and len(r["target"])]
        out[c] = stats(np.concatenate(d)) if d else {"n": 0}
    return out


def residuals_per_ref(rows, R, K) -> dict:
    out = {}
    for r in rows:
        if len(r["target"]):
            out.setdefault(r["ref"], {})[r["cls"]] = stats(proj_corr(r["P_cam"], R, K) - r["target"])
    return out


def scale_ratios(rig, R=None, K=None) -> dict:
    """Apparent sizes in the 224 image, mount (through R, K) / handheld: tag side, tag-centre and tip spacing."""
    cams = {k: G.camera_poses_tcp(k, rig)["ultrawide"] for k in ("handheld", "mount")}
    R = np.eye(3) if R is None else R
    K = np.asarray(rig.virtual_ultrawide.intrinsics, float) if K is None else K
    side = lambda u: np.mean([np.linalg.norm(u[i] - u[(i + 1) % 4]) for i in range(4)])  # noqa: E731
    out = {"tag_side": [], "tag_centre_spacing": [], "tip_spacing": [], "handheld_tag_side_px": [],
           "handheld_tip_spacing_px": []}
    for ref in G.load_refs():
        f = {"handheld": G.finger_points("handheld", ref["width_m"]), "mount": G.finger_points("mount", ref["width_m"])}
        uh = {k: G.project("ultrawide", cams["handheld"], f["handheld"][k], rig)[0] for k in ("tags", "tips")}
        um = {k: proj_corr(G.apply(G.inv(cams["mount"]), f["mount"][k]), R, K) for k in ("tags", "tips")}
        out["tag_side"].append(np.mean([side(um["tags"][4 * i:4 * i + 4]) / side(uh["tags"][4 * i:4 * i + 4]) for i in (0, 1)]))
        cs = lambda u: np.linalg.norm(u[:4].mean(0) - u[4:].mean(0))  # noqa: E731
        out["tag_centre_spacing"].append(cs(um["tags"]) / cs(uh["tags"]))
        out["tip_spacing"].append(np.linalg.norm(um["tips"][0] - um["tips"][1]) / np.linalg.norm(uh["tips"][0] - uh["tips"][1]))
        out["handheld_tag_side_px"].append(np.mean([side(uh["tags"][4 * i:4 * i + 4]) for i in (0, 1)]))
        out["handheld_tip_spacing_px"].append(np.linalg.norm(uh["tips"][0] - uh["tips"][1]))
    return {k: {"mean": round(float(np.mean(v)), 4), "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}
            for k, v in out.items()}


def border(rig, R=None, K=None) -> dict:
    """Black (uncovered) part of the 224 image the corrected virtual camera introduces."""
    cov = G.coverage_mask_224(rig, R, K)
    content = np.zeros((224, 224), bool)
    content[CONTENT_ROWS[0]:CONTENT_ROWS[1], :] = True
    crop = np.zeros((224, 224), bool)
    a, b = int(CROP_MARGIN), int(CROP_MARGIN) + CROP_KEEP
    crop[a:b, a:b] = True
    black = cov < 0.5
    out = {"black_px_in_image_content": int((black & content).sum()),
           "black_frac_of_content": round(float((black & content).sum() / content.sum()), 5),
           "black_frac_inside_centre_crop_212": round(float((black & content & crop).sum() / (content & crop).sum()), 5)}
    if (black & content).any():
        ys, xs = np.nonzero(black & content)
        out["black_extent_px"] = {"left": int(max(0, xs[xs < 112].max() + 1)) if (xs < 112).any() else 0,
                                  "right": int(224 - xs[xs >= 112].min()) if (xs >= 112).any() else 0,
                                  "top_from_row28": int(ys[ys < 112].max() + 1 - 28) if (ys < 112).any() else 0,
                                  "bottom_from_row196": int(196 - ys[ys >= 112].min()) if (ys >= 112).any() else 0}
    return out


# ------------------------------------------------------------------------------------------- real-image check

def _ref_image(ref) -> tuple[np.ndarray | None, int | None]:
    f = SCRATCH / f"ref_ep{ref['index']}.npz"
    if not f.is_file():
        return None, None
    z = np.load(f)
    s = G._episode(ref["index"])["rows"][0]
    k = np.flatnonzero(z["rows"] <= s + ref["row"])
    if not len(k):
        return None, None
    j = int(k[-1])
    return z["uw"][j], int(z["rows"][j] - s)


def real_tag_check(rig) -> dict:
    """Training's 224 frames at the reference rows: ArUco corners vs the handheld projections (E7 and nominal)."""
    import sim_deploy.sim_pi.handheld_check as HC
    det = HC.TagDetector()
    cams = {k: G.camera_poses_tcp(k, rig)["ultrawide"] for k in ("handheld", "handheld_nominal", "mount")}
    out, allv = {}, {k: [] for k in cams}
    for ref in G.load_refs():
        img, row = _ref_image(ref)
        if img is None:
            continue
        found = det(img)
        rec = {"image_row_in_episode": row, "tags_found": sorted(found)}
        for kind, T in cams.items():
            P = G.finger_points("mount" if kind == "mount" else "handheld", ref["width_m"])["tags"]
            uv = G.project("ultrawide", T, P, rig)[0]
            ds = []
            for i, tid in enumerate((0, 1)):
                if tid in found:
                    dv, _ = HC.match_corners(uv[4 * i:4 * i + 4], found[tid])
                    ds.append(dv)
            if ds:
                d = np.concatenate(ds)
                rec[kind] = stats(d)
                allv[kind].append(d)
        out[ref["name"]] = rec
    summ = {k: stats(np.concatenate(v)) for k, v in allv.items() if v}
    return {"per_ref": out, "all": summ,
            "note": "projection minus the detected corners (224 px); the reference image is the newest IMX219 frame "
                    "at the row; rows are 60 Hz so the TCP pose and the frame can differ by < 1/60 s"}


# ------------------------------------------------------------------------------------------- 4/5. files

def rig_text(rig_id: str, euler, K, header: list[str]) -> str:
    src = D.DEFAULT_RIG_FILE.read_text().splitlines()
    body = [ln for ln in src if not ln.startswith("#")]
    text = "\n".join(body) + "\n"
    old_block = ("virtual_ultrawide:\n  resolution: [640, 480]\n  intrinsics: [246.64919, 246.64919, 320.16495, 239.53679]\n"
                 "  rotation:\n    euler_xyz_deg: [0.0, 0.0, 0.0]\n")
    assert text.count(old_block) == 1 and text.count("rig_id: rig_pi_v3_640\n") == 1
    new_block = ("virtual_ultrawide:\n  resolution: [640, 480]\n  intrinsics: [%.5f, %.5f, %.5f, %.5f]\n"
                 "  rotation:\n    euler_xyz_deg: [%.5f, %.5f, %.5f]\n") % (*K, *euler)
    text = text.replace(old_block, new_block).replace("rig_id: rig_pi_v3_640\n", f"rig_id: {rig_id}\n")
    return "\n".join("# " + h for h in header) + "\n" + text


def write_files(res: dict, rig) -> dict:
    import hashlib
    sha = hashlib.sha256(D.DEFAULT_RIG_FILE.read_bytes()).hexdigest()
    written = {}
    for path, key, what in ((CORR_RIG, PRIMARY, "fingers + tags + table (6 classes, equal weight per class)"),
                            (CORR_RIG_FT, SECONDARY, "fingers + tags only (tags, fingertip pads, holders)")):
        f = res["fits"][key[0]][key[1]]
        rr = f["residuals"]
        hdr = [f"SIM ONLY - camera study B (sim_deploy/PLAN.md section 10), sub-step B4 corr_uw. Not for the robot or training.",
               f"Copy of deployment/contracts/rigs/rig_pi_v3_640.yaml (sha256 {sha[:16]}...) with two changes:",
               f"  rig_id {path.stem} (= the file stem, load_rig checks it) and virtual_ultrawide.intrinsics + "
               f"rotation.euler_xyz_deg",
               f"  (was [246.64919, 246.64919, 320.16495, 239.53679] / [0, 0, 0]). Everything else is unchanged.",
               f"Fitted by sim_deploy/corrections/fit_uw.py ({key[1]}: rotation + one focal scale fx = fy, principal point",
               f"  fixed): least squares in 224 px so that the Franka MOUNT IMX219 (sim model cam_imx219) seen through this",
               f"  virtual camera matches the HANDHELD training view (T_cam_tcp + E7's camera fix) through the original one,",
               f"  over {what}, 15 reference frames of sim_evals/_camera_study/B/layout.json.",
               f"  Residual mean / p90 px after: " + ", ".join(f"{c} {rr[c]['mean_px']}/{rr[c]['p90_px']}"
                                                          for c in G.CLASSES if rr[c].get("n")),
               f"  zoom x{f['zoom_vs_original']}; black border: {f['border']['black_frac_of_content']:.4f} of the image. "
               f"Numbers: sim_evals/_camera_study/B/corr_uw/fit.json.",
               "euler_xyz_deg: extrinsic x-y-z (config.py _rotation: scipy from_euler('xyz')), R_phys_virt, "
               "d_phys = R_phys_virt d_virt."]
        euler = Rotation.from_matrix(f["R"]).as_euler("xyz", degrees=True)
        path.write_text(rig_text(path.stem, euler, f["K"], hdr))
        written[str(path.relative_to(REPO))] = {"fit": list(key)}
    # the variant station: policy_sim.yaml with frames.rig_file only (absolute: the eval driver freezes copies elsewhere)
    st = STATION_SRC.read_text()
    old = "frames: {}                        # overrides of config/frames_process.yaml"
    assert st.count(old) == 1
    rig_path = "~/" + str(CORR_RIG.relative_to(Path.home()))
    st = st.replace(old, f"frames: {{rig_file: {rig_path}}}  # camera study B: the corrected IMX219 virtual camera"
                         f"\n                                  # (was: frames: {{}}, overrides of config/frames_process.yaml")
    hdr = ("# camera study B (sim only, B4 corr_uw): a copy of sim_deploy/stations/policy_sim.yaml with one change,\n"
           f"# frames.rig_file = {rig_path} (the corrected virtual ultrawide).\n"
           "# The path is absolute (~) so that frozen copies of this file in other folders still find it.\n")
    STATION_OUT.write_text(hdr + st)
    written[str(STATION_OUT.relative_to(REPO))] = {"frames.rig_file": rig_path}
    return written


def check_files(res: dict) -> dict:
    """The written rig through the copy's loader + ImagePrep, end to end on a synthetic fisheye, and the laptop-env
    checks (rig_file --json, frames Settings.load of the variant station). No process is started."""
    out = {}
    for path, key in ((CORR_RIG, PRIMARY), (CORR_RIG_FT, SECONDARY)):
        f = res["fits"][key[0]][key[1]]
        rig_c = D.load_live_rig(path)
        prep = D.ImagePrep(rig_c)
        R_l = np.asarray(rig_c.virtual_ultrawide.R_phys_virt)
        out[path.name] = {
            "rig_id": rig_c.rig_id,
            "R_phys_virt_vs_fit_max_abs": float(np.abs(R_l - f["R"]).max()),
            "UltrawidePrep_R_vs_fit_max_abs": float(np.abs(prep.ultrawide.R_phys_virt - f["R"]).max()),
            "UltrawidePrep_K_vs_fit_max_abs": float(np.abs(np.array([prep.ultrawide.virt.fx, prep.ultrawide.virt.fy,
                                                                     prep.ultrawide.virt.cx, prep.ultrawide.virt.cy])
                                                           - f["K"]).max()),
            "remap_coverage": round(float(prep.ultrawide.remap.coverage), 5),
            "e2e_synthetic": e2e_check(rig_c, prep)}
    out["default_rig_e2e_synthetic"] = e2e_check(G.load_rig(), D.ImagePrep(G.load_rig()))
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    r = subprocess.run([str(PD_PY), "-m", "policy_deploy.frames.rig_file", str(CORR_RIG), "--json"], cwd=PD_DIR,
                       capture_output=True, text=True, env=env, timeout=60)
    out["launcher_rig_file_check"] = {"rc": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[-400:]}
    code = ("import json,sys; from pathlib import Path; from policy_deploy.frames.settings import Settings; "
            "s=Settings.load(Path(sys.argv[1]), mock=True); "
            "from policy_deploy.frames.rig_file import load_live_rig; from policy_deploy.frames._dataproc.imageprep "
            "import ImagePrep; rig=load_live_rig(s.rig_file); p=ImagePrep(rig); "
            "print(json.dumps({'rig_file': str(s.rig_file), 'rig_id': rig.rig_id, 'fake_pi': s.fake_pi, "
            "'K_virt': [p.ultrawide.virt.fx, p.ultrawide.virt.fy, p.ultrawide.virt.cx, p.ultrawide.virt.cy], "
            "'R_phys_virt': p.ultrawide.R_phys_virt.tolist()}))")
    for st in (STATION_OUT, STATION_SRC):
        r = subprocess.run([str(PD_PY), "-c", code, str(st)], cwd=PD_DIR, capture_output=True, text=True, env=env,
                           timeout=60)
        out[f"frames_settings_load_{st.stem}"] = {"rc": r.returncode, "stdout": r.stdout.strip()[-800:],
                                                  "stderr": r.stderr.strip()[-600:]}
    return out


def e2e_check(rig_c, prep, sigma_px: float = 3.0) -> dict:
    """Dots at the physical-fisheye projections of known rays (the mount camera's points of three references) ->
    prep_ultrawide -> centroids in the 224 image vs the projection math through rig_c's virtual camera."""
    cam = G.camera_poses_tcp("mount", rig_c)["ultrawide"]
    R = np.asarray(rig_c.virtual_ultrawide.R_phys_virt, float)
    K = np.array(rig_c.virtual_ultrawide.intrinsics, float)
    P = []
    for ref in G.load_refs()[::5]:
        ps = G.point_sets(ref, "mount", rig_c)
        P += [ps["tags"], ps["fingertips"][::8], ps["holders"][::6], ps["table_near"][::3], ps["table_mid"][::3],
              ps["table_far"][::3], ps["eraser"][:8]]
    uv_phys, ok, P_cam = G.project_native("ultrawide", cam, np.concatenate(P), rig_c)
    pred, okv = G.project("ultrawide", cam, np.concatenate(P), rig_c, R, K)
    ok &= okv
    ok &= (pred[:, 0] > 4) & (pred[:, 0] < 219) & (pred[:, 1] > 32) & (pred[:, 1] < 191)
    keep = []
    for i in np.flatnonzero(ok):                                  # >= 9 px apart in 224, >= 20 px in the fisheye
        if all(np.linalg.norm(pred[i] - pred[j]) >= 9 and np.linalg.norm(uv_phys[i] - uv_phys[j]) >= 20 for j in keep):
            keep.append(i)
    keep = np.array(keep)
    H, W = 480, 640
    img = np.zeros((H, W), np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    for i in keep:
        u, v = uv_phys[i]
        x0, x1, y0, y1 = int(max(0, u - 15)), int(min(W, u + 16)), int(max(0, v - 15)), int(min(H, v + 16))
        img[y0:y1, x0:x1] += np.exp(-((xx[y0:y1, x0:x1] - u) ** 2 + (yy[y0:y1, x0:x1] - v) ** 2) / (2 * sigma_px ** 2))
    rgb = np.repeat(np.clip(img * 255.0, 0, 255).round().astype(np.uint8)[:, :, None], 3, axis=2)
    out224 = prep.prep_ultrawide(np.ascontiguousarray(rgb))[:, :, 0].astype(np.float64)
    n, lab = cv2.connectedComponents((out224 > 8).astype(np.uint8))
    cents = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        w = out224[ys, xs]
        cents.append([np.sum(xs * w) / w.sum(), np.sum(ys * w) / w.sum()])
    cents = np.array(cents)
    d = []
    for i in keep:
        j = int(np.argmin(np.linalg.norm(cents - pred[i], axis=1)))
        d.append(cents[j] - pred[i])
    d = np.array(d)
    return {"dots": int(len(keep)), "blobs_found": int(n - 1), "sigma_phys_px": sigma_px,
            "detected_minus_projection": stats(d)}


# ------------------------------------------------------------------------------------------- pictures

COL = {"handheld": (60, 220, 60), "mount": (40, 40, 235), "corrected": (235, 160, 30)}   # BGR


def overlay(ref, rig, fitres, path: Path) -> None:
    S = 3
    img, row = _ref_image(ref)
    bg = (np.full((224, 224, 3), 40, np.uint8) if img is None else (0.55 * img[:, :, ::-1]).astype(np.uint8))
    bg = cv2.resize(bg, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)
    cams = {k: G.camera_poses_tcp(k, rig)["ultrawide"] for k in ("handheld", "mount")}
    sets = {"handheld": G.point_sets(ref, "handheld", rig), "mount": G.point_sets(ref, "mount", rig)}
    proj = {"handheld": {c: G.project("ultrawide", cams["handheld"], sets["handheld"][c], rig) for c in G.CLASSES},
            "mount": {c: G.project("ultrawide", cams["mount"], sets["mount"][c], rig) for c in G.CLASSES}}
    panels = []
    for title, other, RK in (("before: mount (red) vs handheld (green)", "mount", None),
                             (f"corrected mount (orange, fit {PRIMARY[0]}/{PRIMARY[1]}) vs handheld (green)", "corrected",
                              (fitres["R"], fitres["K"]))):
        im = bg.copy()
        a, b = int(CROP_MARGIN * S), int((CROP_MARGIN + CROP_KEEP) * S)
        cv2.rectangle(im, (a, a), (b, b), (200, 200, 200), 1)
        for kind in ("handheld", other):
            col = COL[kind]
            for c in G.CLASSES:
                if kind == "corrected":
                    _, ok, Pc = G.project_native("ultrawide", cams["mount"], sets["mount"][c], rig)
                    uv = proj_corr(Pc, *RK)
                    ok = ok & (uv[:, 0] > -0.5) & (uv[:, 0] < 223.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < 223.5)
                else:
                    uv, ok = proj[kind][c]
                q = ((uv + 0.5) * S - 0.5)
                if c == "tags":
                    for i in (0, 1):
                        if ok[4 * i:4 * i + 4].all():
                            cv2.polylines(im, [q[4 * i:4 * i + 4].round().astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
                elif c == "eraser":
                    top = q[4:8][[0, 1, 3, 2]]
                    if ok[4:8].all():
                        cv2.polylines(im, [top.round().astype(np.int32).reshape(-1, 1, 2)], True, col, 2)
                else:
                    step = {"fingertips": 3, "holders": 2}.get(c, 1)
                    r = {"fingertips": 1, "holders": 1}.get(c, 2)
                    for k in np.flatnonzero(ok)[::step]:
                        cv2.circle(im, tuple(int(x) for x in q[k].round()), r, col, -1)
        cv2.putText(im, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        panels.append(im)
    lab = np.zeros((24, 2 * 224 * S + 6, 3), np.uint8)
    cv2.putText(lab, f"{ref['name']} (row {ref['row']}, width {ref['width_m'] * 1e3:.1f} mm; background: training "
                     f"IMX219 224 frame row {row}, dimmed; box = 212 px centre crop). Dots: fingertip pads, holders, "
                     f"table grid; polygons: tags, eraser top", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    sep = np.full((224 * S, 6, 3), 255, np.uint8)
    cv2.imwrite(str(path), np.concatenate([lab, np.concatenate([panels[0], sep, panels[1]], axis=1)], axis=0))


# ------------------------------------------------------------------------------------------- main

def strip(f: dict) -> dict:
    return {k: v for k, v in f.items() if k not in ("R", "K")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim_deploy.corrections.fit_uw", description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the rig files and the variant station, then check them")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rig = G.load_rig()
    res = {"what": "camera study B4 corr_uw: IMX219 handheld vs mount geometry and a sim-only virtual-camera correction",
           "rig": str(D.DEFAULT_RIG_FILE.relative_to(REPO)), "model": str(G.MODEL_XML.relative_to(REPO)),
           "conventions": {"224px": "policy ultrawide image, OpenCV pixel centres; rows 28..195 = the 640x480 "
                                    "virtual image letterboxed", "offset": "mount minus handheld (dx right, dy down)",
                           "training_crop": f"CenterCrop/RandomCrop {CROP_KEEP} of 224 then resize to 224: the eval "
                                            f"crop removes {CROP_MARGIN:g} px per side; train-time RandomCrop shifts the "
                                            f"crop by up to +-{CROP_MARGIN:g} px (x and y independently, uniform), no "
                                            f"scale or rotation augmentation"}}
    print("[corr_uw] 1. camera poses", flush=True)
    res["cameras"] = camera_section(rig)
    print("[corr_uw] 2. finger models, table evidence", flush=True)
    res["finger_models"] = finger_model_section(rig)
    res["table_evidence"] = table_evidence()
    refs = G.load_refs()
    res["references"] = {r["name"]: {"row": r["row"], "width_mm": round(r["width_m"] * 1e3, 2),
                                     "tcp_tag_mm": r3(r["T_tag_tcp"][:3, 3] * 1e3, 1),
                                     "camera_height_handheld_mm": round(float((r["T_tag_tcp"] @ G.camera_poses_tcp(
                                         "handheld", rig)["ultrawide"])[2, 3] * 1e3), 1),
                                     "grasp_row": r["grasp_row"],
                                     "grasp_tcp_tag_mm": r3(r["T_tag_tcp_grasp"][:3, 3] * 1e3, 1),
                                     "table_dist_edges_m": r3(G.scene_points(r, rig)["_table_dist_edges_m"], 3)}
                         for r in refs}
    print("[corr_uw] 3. projections", flush=True)
    rows = collect(rig, "handheld")
    rows_nom = collect(rig, "handheld_nominal")
    res["before"] = {"mount_vs_handheld_E7": residuals(rows), "mount_vs_handheld_nominal": residuals(rows_nom),
                     "mount_vs_handheld_E7_per_ref": residuals_per_ref(rows, np.eye(3),
                                                                       np.asarray(rig.virtual_ultrawide.intrinsics, float)),
                     "scale_ratio_mount_over_handheld": scale_ratios(rig)}
    rows_shared = collect(rig, "handheld", shared_fingers=True)
    res["before"]["mount_vs_handheld_E7_same_finger_points"] = residuals(rows_shared)
    print("[corr_uw] real-image tag check", flush=True)
    res["real_image_tag_check"] = real_tag_check(rig)
    print("[corr_uw] 4. fits", flush=True)
    fits = {}
    for fs, w in FITSETS.items():
        fits[fs] = {}
        for p in PARAMS:
            f = fit(rows, rig, p, w)
            f["residuals"] = residuals(rows, f["R"], f["K"])
            f["border"] = border(rig, f["R"], f["K"])
            fits[fs][p] = f
            rr = f["residuals"]
            print(f"   {fs:12s} {p:11s} zoom {f['zoom_vs_original']:.4f} euler {f['euler_xyz_deg']} K {f['intrinsics']} "
                  f"| " + " ".join(f"{c[:6]} {rr[c]['mean_px']:.1f}" for c in G.CLASSES), flush=True)
    res["fits"] = fits
    pf = fits[PRIMARY[0]][PRIMARY[1]]
    res["primary"] = {"fit": list(PRIMARY), "scale_ratio_corrected_over_handheld": scale_ratios(rig, pf["R"], pf["K"]),
                      "per_ref": residuals_per_ref(rows, pf["R"], pf["K"])}
    res["secondary"] = {"fit": list(SECONDARY), "scale_ratio_corrected_over_handheld": scale_ratios(
        rig, fits[SECONDARY[0]][SECONDARY[1]]["R"], fits[SECONDARY[0]][SECONDARY[1]]["K"])}
    # sensitivity: the fit target = training's nominal camera; the table at the hold-row evidence z
    zt = res["table_evidence"]["tcp_z_at_hold_row_mm"]["median"] * 1e-3
    sens = {}
    for name, rws in (("target_handheld_nominal", rows_nom), (f"table_at_z_{zt * 1e3:.1f}mm", collect(rig, "handheld", zt))):
        f = fit(rws, rig, PRIMARY[1], FITSETS[PRIMARY[0]])
        sens[name] = {**strip(f), "before": residuals(rws), "residuals": residuals(rws, f["R"], f["K"])}
    res["sensitivity"] = sens
    res["original_border"] = border(rig)
    if a.write:
        print("[corr_uw] 5. writing + checking the rig files and the variant station", flush=True)
        res["written"] = write_files(res, rig)
        res["checks"] = check_files(res)
    if not a.no_png:
        for name in ("ep14_first", "ep02_approach_2of3", "ep25_first"):
            overlay(next(r for r in refs if r["name"] == name), rig, pf, OUT / f"overlay_{name}.png")
    for fs in fits.values():
        for p in list(fs):
            fs[p] = strip(fs[p])
    (OUT / "fit.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"[corr_uw] wrote {OUT / 'fit.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
