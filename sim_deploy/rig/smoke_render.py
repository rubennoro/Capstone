"""Smoke run for add_rig: franka_sim's sim-rig model (Panda + Franka Hand + finger kit + table) with the CAD rig added,
compiled, then one pinhole image from each rig camera and one external view of the hand + rig.

    MUJOCO_GL=egl ~/envs/env_sim/bin/python sim_deploy/rig/smoke_render.py --out <dir>

The old iPhone placeholder geoms on franka_sim's ``tool`` body (housing, cradle) are removed so they do not hide the
rig. Prints the camera poses read back from the compiled model, contacts touching the rig at home, and where the two
fingertips land in each image (they should be in the lower half).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FRANKA_SIM = HERE.parents[1] / "controlSW" / "franka_sim"      # this repo's franka_sim (not pip-installed in env_sim)
if str(FRANKA_SIM) not in sys.path:
    sys.path.insert(1, str(FRANKA_SIM))
from add_rig import add_rig, cam_T_tip, hand_T_cam, load_rig  # noqa: E402

from franka_sim.model import build_model  # noqa: E402  (also puts controlSW/franka_deploy on sys.path)
from franka_deploy.station import PROFILES, STATION_ROOT, load_station  # noqa: E402

W, H = 640, 480


def K_of(fx):
    return np.array([[fx, 0, (W - 1) / 2], [0, fx, (H - 1) / 2], [0, 0, 1.0]])


def project(K, T_world_cam_cv, p_world):
    p = np.linalg.inv(T_world_cam_cv) @ np.r_[p_world, 1.0]
    return (K @ (p[:3] / p[2]))[:2], p[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile", default="sim-rig")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    station = load_station({k: dict(v) for k, v in PROFILES[a.profile].items()}, root=STATION_ROOT)
    sm = build_model(station)
    spec = sm.spec
    for g in list(spec.body("tool").geoms):
        g.delete()
    hand = sm.names["bodies"]["hand"]
    Ks = {"cam_imx219": K_of(250.0), "cam_imx519": K_of(400.0)}
    info = add_rig(spec, hand, HERE, intrinsics={n: (K, W, H) for n, K in Ks.items()})
    # a second, very wide pinhole at the IMX219 pose (142 deg across): shows which rig parts enter the fisheye's view
    T_w = hand_T_cam(load_rig(HERE), "cam_imx219")
    cw = spec.body(hand).add_camera()
    from add_rig import CV_TO_MJ, _quat, _set_intrinsics
    cw.name, cw.pos, cw.quat = "cam_imx219_wide", T_w[:3, 3], _quat(T_w[:3, :3] @ CV_TO_MJ)
    Ks["cam_imx219_wide"] = K_of(110.0)
    _set_intrinsics(cw, Ks["cam_imx219_wide"], W, H)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, W)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, H)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    print(f"[smoke] compiled: nbody {m.nbody}, ngeom {m.ngeom}, rig body mass {m.body('rig').mass[0]:.4f} kg "
          f"(rig.yaml {info['mass_kg']:.4f})")

    rig = load_rig(HERE)
    hb = m.body(hand).id
    W_T_hand = np.eye(4)
    W_T_hand[:3, :3] = d.xmat[hb].reshape(3, 3)
    W_T_hand[:3, 3] = d.xpos[hb]
    W_T_fl = np.eye(4)
    sid = m.site("flange").id
    W_T_fl[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    W_T_fl[:3, 3] = d.site_xpos[sid]
    fl_T_hand = np.linalg.inv(W_T_fl) @ W_T_hand
    print("[smoke] flange_T_hand (model):", np.round(fl_T_hand, 6).tolist())
    print("[smoke] flange_T_hand (rig.yaml):", np.round(np.array(rig["frames"]["flange_T_hand"]), 6).tolist())

    # contacts touching the rig at home
    rig_geoms = {m.geom(n).id for n in info["collision_geoms"]}
    hits = [(m.geom(c.geom1).name, m.geom(c.geom2).name, c.dist) for c in d.contact[: d.ncon]
            if c.geom1 in rig_geoms or c.geom2 in rig_geoms]
    print(f"[smoke] contacts with rig collision geoms at home: {hits or 'none'}")

    # fingertip positions: the max-z point of each finger body's geoms, in the world
    tips = []
    for fb in (sm.names["bodies"]["finger_l"], sm.names["bodies"]["finger_r"]):
        bid = m.body(fb).id
        best = None
        for g in np.flatnonzero(m.geom_bodyid == bid):
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            h = (np.linalg.inv(W_T_hand) @ np.c_[w, np.ones(len(w))].T).T[:, :3]
            i = int(np.argmax(h[:, 2]))
            if best is None or h[i, 2] > best[2]:
                best = h[i]
        tips.append(best)
    tips = np.array(tips)
    z_tip = float(tips[:, 2].mean())
    print(f"[smoke] fingertips (hand frame, mm): {np.round(tips * 1e3, 1).tolist()}")

    r = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    for gi in range(6):
        opt.geomgroup[gi] = 1 if gi in (0, 1, 2, 3) else 0
    for name in ("cam_imx219", "cam_imx519", "cam_imx219_wide"):
        cid = m.camera(name).id
        T_mj = np.eye(4)
        T_mj[:3, :3] = d.cam_xmat[cid].reshape(3, 3)
        T_mj[:3, 3] = d.cam_xpos[cid]
        T_cv = T_mj.copy()
        T_cv[:3, :3] = T_mj[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        exp = W_T_hand @ hand_T_cam(rig, name.replace("_wide", ""))
        err_t = np.linalg.norm(T_cv[:3, 3] - exp[:3, 3]) * 1e3
        err_r = np.degrees(np.arccos(np.clip((np.trace(T_cv[:3, :3].T @ exp[:3, :3]) - 1) / 2, -1, 1)))
        r.update_scene(d, camera=name, scene_option=opt)
        img = r.render()
        px = [project(Ks[name], T_cv, W_T_hand[:3, :3] @ t + W_T_hand[:3, 3]) for t in tips]
        tip_pt = W_T_hand @ np.r_[0, 0, z_tip, 1]
        pc, _ = project(Ks[name], T_cv, tip_pt[:3])
        T_ct = cam_T_tip(rig, name.replace("_wide", ""), z_tip)
        print(f"[smoke] {name}: model vs rig.yaml {err_t:.4f} mm / {err_r:.4f} deg; fingertips at px "
              f"{[np.round(p[0], 1).tolist() for p in px]} (image {W}x{H}; v > {H / 2} = lower half); fingertip point "
              f"({z_tip * 1e3:.1f} mm) at {np.round(pc, 1).tolist()}, cam_T_tip t = "
              f"{np.round(T_ct[:3, 3] * 1e3, 1).tolist()} mm")
        _save(a.out / f"{name}.png", img, marks=[p[0] for p in px] + [pc])

    # external views (world azimuth/elevation; at home the hand's +x, the camera side, is about world +x)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    look = W_T_hand @ np.r_[0.07, 0.0, 0.09, 1]
    cam.lookat[:] = look[:3]
    cam.distance = 0.55
    for nm, az, el in (("external_profile", 90.0, -10.0), ("external_above", 150.0, -30.0),
                       ("external_below", 200.0, 25.0)):
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, camera=cam, scene_option=opt)
        _save(a.out / f"{nm}.png", r.render())
    r.close()
    print(f"[smoke] wrote {sorted(p.name for p in a.out.glob('*.png'))} to {a.out}")
    return 0


def _save(path: Path, img: np.ndarray, marks=()) -> None:
    img = img.copy()
    for u, v in marks:
        u, v = int(round(u)), int(round(v))
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            img[max(v - 4, 0): v + 5, max(u - 1, 0): u + 2] = (255, 0, 255)
            img[max(v - 1, 0): v + 2, max(u - 4, 0): u + 5] = (255, 0, 255)
    try:
        import cv2

        cv2.imwrite(str(path), img[:, :, ::-1])
    except ImportError:
        from PIL import Image

        Image.fromarray(img).save(path)


if __name__ == "__main__":
    sys.exit(main())
