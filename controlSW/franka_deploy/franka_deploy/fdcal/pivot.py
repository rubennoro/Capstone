"""Pivot test of the TCP: rotate +-20 deg about each TCP axis around a fixed point (fingertip centre on a marked
point). With a correct F_T_TCP the TCP position stays put; this tool reports the TCP excursion and estimates the
actual pivot point in the flange frame by least squares (R_i p_F + t_i = p_O), i.e. the F_T_TCP translation error.
Accept: excursion <= 2 mm AND estimated TCP translation error <= 2 mm.

    python -m franka_deploy fdcal pivot --poses poses.json --F_T_TCP franka_cartesian.yaml|4x4.json
poses.json = list of 4x4 O_T_F recorded at the rotation extremes (bridge get_state).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def pivot_commands(center_O_T_TCP: np.ndarray, angle_deg: float = 20.0) -> list[np.ndarray]:
    """TCP target poses: +-angle about the TCP's own x, y, z axes (same TCP position)."""
    out = []
    for axis in np.eye(3):
        for s in (+1, -1):
            T = center_O_T_TCP.copy()
            T[:3, :3] = center_O_T_TCP[:3, :3] @ Rotation.from_rotvec(axis * np.radians(s * angle_deg)).as_matrix()
            out.append(T)
    return out


def analyze(O_T_F_list, F_T_TCP) -> dict:
    tcp = np.array([(T @ F_T_TCP)[:3, 3] for T in O_T_F_list])
    exc = float(np.linalg.norm(tcp - tcp.mean(0), axis=1).max())
    # least squares: R_i p_F - p_O = -t_i
    A = np.vstack([np.hstack([T[:3, :3], -np.eye(3)]) for T in O_T_F_list])
    b = np.concatenate([-T[:3, 3] for T in O_T_F_list])
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    p_F = x[:3]
    err = float(np.linalg.norm(p_F - F_T_TCP[:3, 3]))
    # with +-20 deg the excursion is only ~0.35x the TCP error, so also gate on the least-squares pivot estimate
    return {"tcp_excursion_mm": exc * 1000, "pivot_in_flange_m": p_F.tolist(), "F_T_TCP_translation_error_mm": err * 1000,
            "accept": exc <= 0.002 and err <= 0.002}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--F_T_TCP", required=True)
    a = ap.parse_args(argv)
    poses = [np.array(T) for T in json.load(open(a.poses))]
    src = a.F_T_TCP
    F_T_TCP = np.array(yaml.safe_load(open(src))["F_T_TCP"] if src.endswith((".yaml", ".yml")) else json.load(open(src)))
    out = analyze(poses, F_T_TCP)
    print(json.dumps(out, indent=1))
    return 0 if out["accept"] else 2
