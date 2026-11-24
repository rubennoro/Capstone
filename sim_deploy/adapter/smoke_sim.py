"""Smoke test (E3): mount the kit_franka_v1 finger kit (soft fingers + UMI holders + tags + this adapter) on
franka_sim's Panda + Franka Hand, compile, check the contact geometry and render the hand at gap 0 and 0.08.

Checks: the model compiles with the kit (finger_kit placement, no placeholder block); F_T_TCP z = adapter.yaml's D7
value; the soft fingertips meet at gap 0 (gripping faces at hand y = 0, pad hulls touching); the tag-to-tag distance;
over the whole stroke no adapter vertex is inside or within the clearance of the Franka Hand housing (mj_ray against
the hand's visual meshes, which are the real shape; hand-finger contacts are excluded in hand.xml, so MuJoCo itself
would not report it).

Usage: MUJOCO_GL=egl ~/envs/env_sim/bin/python sim_deploy/adapter/smoke_sim.py [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "controlSW/franka_sim"))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

from franka_sim import finger_kit as FK  # noqa: E402
from franka_sim.model import build_model, forward_at, xform_of  # noqa: E402
from franka_deploy.station import PROFILES, load_station  # noqa: E402

KIT = HERE / "kit_franka_v1"


def station(camera_side: str = "+y"):
    sha = FK.kit_digest(KIT)
    sets = [f"tooling.finger_adapter.kit={KIT}", f"tooling.finger_adapter.kit_sha256={sha}",
            "tooling.finger_adapter.mode=kit", "tooling.finger_adapter.kit_variant=full",
            "tooling.finger_adapter.adapter_length_m=0.0", "tooling.finger_adapter.lateral_offset_m=0.0",
            "tooling.finger_adapter.adapter_yaw_deg=0.0", f"tooling.tcp_x_along_hand={camera_side}"]
    spec = {k: dict(v) for k, v in PROFILES["sim-rig"].items()}
    return load_station(spec, sets=sets)


def gid(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise KeyError(name)
    return i


def geom_verts_world(m, d, g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/"
                                     "scratchpad/e3")
    ap.add_argument("--camera-side", default="+y", help="tooling.tcp_x_along_hand: +y = camera on hand +x (PLAN 2)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ay = yaml.safe_load((HERE / "adapter.yaml").read_text())
    D7 = ay["D7_tcp_z_offset_from_flange_m"]

    sm = build_model(station(a.camera_side), with_scene=False)
    m = sm.model
    fk = sm.info["finger_kit"]
    print(f"compiled: nq={m.nq} ngeom={m.ngeom}; kit {fk['id']} variant {fk['variant']} L={fk['L']} dy={fk['dy']}")
    print(f"  sides on fingers: {fk['side_for_finger']}; tag ids {fk['tag_ids']}")
    print(f"  masses (kg): " + ", ".join(f"{k} {v:.4f}" for k, v in fk["masses_kg"].items()))
    print(f"  finger body masses (kg): {fk['finger_body_mass_kg']}")
    tz = sm.F_T_TCP[2, 3]
    res = {"F_T_TCP_xyz_mm": (sm.F_T_TCP[:3, 3] * 1e3).round(3).tolist(), "D7_m": D7,
           "tcp_z_err_mm": float((tz - D7) * 1e3)}
    print(f"  F_T_TCP xyz mm {res['F_T_TCP_xyz_mm']} (adapter.yaml D7 {D7 * 1e3:.2f}: err {res['tcp_z_err_mm']:+.3f})")

    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, sm.names["bodies"]["hand"])
    hand_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hand and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                  and m.geom_contype[g] == 0]                          # visual meshes = the real housing shape
    ad = {s: gid(m, f"finger_holder_{s}_1") for s in ("l", "r")}       # 2nd structure part = the adapter
    fbody = {s: int(m.geom_bodyid[ad[s]]) for s in ("l", "r")}
    pads = {s: gid(m, f"pad_{s}") for s in ("l", "r")}
    gmask = np.zeros(6, np.uint8)
    gmask[[0, 1, 2]] = 1
    worst = {"down": np.inf, "side": np.inf, "end": np.inf}
    for gap in np.linspace(0.0, 0.08, 9):
        d = forward_at(sm, gap_m=gap)
        W_H = xform_of(d, "body", hand)
        R = W_H[:3, :3]
        for s in ("l", "r"):
            V = geom_verts_world(m, d, ad[s])
            Vh = (V - W_H[:3, 3]) @ R                                   # hand frame
            # rays from every adapter vertex along -z, +-x, +-y (hand frame): distance to the housing
            for key, dirs in (("down", [(0, 0, -1)]), ("side", [(1, 0, 0), (-1, 0, 0)]),
                              ("end", [(0, 1, 0), (0, -1, 0)])):
                sel = Vh[:, 2] < (0.066 if key != "down" else 0.080)
                for dv in dirs:
                    dw = R @ np.array(dv, float)
                    for p in V[sel]:
                        gi = np.array([-1], np.int32)
                        dist = mujoco.mj_ray(m, d, p, dw, gmask, 1, fbody[s], gi)
                        if gi[0] in hand_geoms and dist >= 0:
                            worst[key] = min(worst[key], dist)
        if gap == 0.0:
            P = {s: (geom_verts_world(m, d, pads[s]) - W_H[:3, 3]) @ R for s in ("l", "r")}
            inner = {s: float(np.abs(P[s][:, 1]).min()) for s in P}
            fromto = np.zeros(6)
            hull_d = mujoco.mj_geomDistance(m, d, pads["l"], pads["r"], 0.01, fromto)
            tags = {s: (d.geom_xpos[gid(m, f"tag_{s}")] - W_H[:3, 3]) @ R for s in ("l", "r")}
            res["gap0"] = {"pad_inner_abs_y_mm": {k: v * 1e3 for k, v in inner.items()},
                           "pad_hull_distance_mm": hull_d * 1e3,
                           "tips_hand_mm": {k: (np.array(v) * 1e3).round(3).tolist() for k, v in fk["tips_hand_m"].items()},
                           "tag_l_hand_mm": (tags["l"] * 1e3).round(2).tolist(),
                           "tag_r_hand_mm": (tags["r"] * 1e3).round(2).tolist(),
                           "tag_distance_y_mm": float(abs(tags["l"][1] - tags["r"][1]) * 1e3)}
            mujoco.mj_collision(m, d)
            res["gap0"]["pad_pad_contacts"] = int(sum(1 for c in d.contact[:d.ncon]
                                                     if {c.geom1, c.geom2} == {pads["l"], pads["r"]}))
    res["housing_clearance_mm"] = {k: float(v * 1e3) for k, v in worst.items()}
    for k, v in res.items():
        print(f"  {k}: {v}")
    ok = (abs(res["tcp_z_err_mm"]) < 0.05 and max(res["gap0"]["pad_inner_abs_y_mm"].values()) < 0.05
          and res["gap0"]["pad_hull_distance_mm"] < 0.05 and res["housing_clearance_mm"]["down"] > 0.9
          and res["housing_clearance_mm"]["side"] > 0.9 and res["housing_clearance_mm"]["end"] > 2.0)
    print("SMOKE", "OK" if ok else "FAILED")

    # renders: external views of the hand, gap 0 and 0.08
    r = mujoco.Renderer(m, 720, 960)
    opt = mujoco.MjvOption()
    for gap in (0.0, 0.08):
        d = forward_at(sm, gap_m=gap)
        W_H = xform_of(d, "body", hand)
        imgs = []
        # view directions in the hand frame (camera -> target): from the camera side (+x, tags visible), from the
        # side (+y, along the closing axis), and an oblique close-up of the adapters
        for dv, dist, zc in (((-1, 0, 0), 0.45, 0.15), ((0, -1, 0), 0.45, 0.15), ((-1, -0.7, -0.25), 0.22, 0.085)):
            v = W_H[:3, :3] @ (np.array(dv, float) / np.linalg.norm(dv))
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = W_H[:3, :3] @ np.array([0, 0, zc]) + W_H[:3, 3]
            cam.distance = dist
            cam.azimuth, cam.elevation = np.degrees(np.arctan2(v[1], v[0])), np.degrees(np.arcsin(v[2]))
            r.update_scene(d, cam, opt)
            imgs.append(r.render())
        from PIL import Image

        im = Image.fromarray(np.concatenate(imgs, axis=1))
        p = out / f"sim_hand_gap{int(round(gap * 1000)):03d}.png"
        im.save(p)
        print(f"wrote {p}")
    r.close()
    (out / "smoke_sim.yaml").write_text(yaml.safe_dump(
        json_safe(res), sort_keys=False, default_flow_style=None, width=120))
    return 0 if ok else 1


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


if __name__ == "__main__":
    sys.exit(main())
