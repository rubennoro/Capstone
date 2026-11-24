"""The `load` block for franka_control (libfranka setLoad): what the arm carries beyond Desk's Franka Hand.

    ~/envs/sim_deploy/bin/python sim_deploy/sim_pc/payload.py [--model sim_deploy/build/sim_rig/scene.xml] [--gap 0.04]

Desk's end effector is the stock Franka Hand (0.73 kg, its own stock fingers included). The extra load is the rig
(body ``rig``, E2's inertial from rig.yaml) plus, per finger, the kit (adapter + UMI holder + soft finger, E3) minus the
stock finger it replaces (menagerie: 15 g at the finger body origin). Mass, centre of mass and inertia about that
centre are given in the flange frame (site ``flange``), as config.yaml ``load`` wants them. The fingers move, so the
inertia is taken at ``--gap`` (the centre of mass does not move: the two fingers are symmetric).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
STOCK_FINGER_KG = 0.015          # menagerie hand.xml finger inertial (pos 0 0 0, diaginertia 2.375e-6 2.375e-6 7.5e-7)
STOCK_FINGER_I = np.diag([2.375e-6, 2.375e-6, 7.5e-7])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", type=Path, default=ROOT / "build" / "sim_rig" / "scene.xml")
    ap.add_argument("--gap", type=float, default=0.04)
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(str(a.model))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    for j in ("finger_joint1", "finger_joint2"):
        d.qpos[m.jnt_qposadr[m.joint(j).id]] = a.gap / 2
    mujoco.mj_forward(m, d)
    s = m.site("flange").id
    R_F, p_F = d.site_xmat[s].reshape(3, 3), d.site_xpos[s]

    parts = []   # (name, mass, com in F, inertia about its com in F)
    def body_part(name, sign=1.0):
        b = m.body(name).id
        R = R_F.T @ d.ximat[b].reshape(3, 3)
        parts.append((name, sign * m.body_mass[b], R_F.T @ (d.xipos[b] - p_F), sign * R @ np.diag(m.body_inertia[b]) @ R.T))
    body_part("rig")
    for fb in ("left_finger", "right_finger"):
        body_part(fb)
        b = m.body(fb).id
        R = R_F.T @ d.xmat[b].reshape(3, 3)
        parts.append((fb + " stock (minus)", -STOCK_FINGER_KG, R_F.T @ (d.xpos[b] - p_F), -R @ STOCK_FINGER_I @ R.T))
    M = sum(p[1] for p in parts)
    c = sum(p[1] * p[2] for p in parts) / M
    I = np.zeros((3, 3))
    for _, mi, ci, Ii in parts:
        r = ci - c
        I += Ii + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    for name, mi, ci, _ in parts:
        print(f"  {name:28s} {mi:+.4f} kg  com_F {np.round(ci, 4).tolist()}")
    rig = yaml.safe_load((ROOT / "rig" / "rig.yaml").read_text())["mass"]
    print(f"rig.yaml: {rig['total_kg']:.5f} kg, com_flange {rig['com_flange_m']}")
    print("load:                     # rig + finger kits - stock fingers, flange frame (sim_deploy/sim_pc/payload.py)")
    print(f"  mass_kg: {M:.4f}")
    print(f"  com_m: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
    print("  inertia_kg_m2:")
    for r in I:
        print(f"    - [{r[0]:.6e}, {r[1]:.6e}, {r[2]:.6e}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
