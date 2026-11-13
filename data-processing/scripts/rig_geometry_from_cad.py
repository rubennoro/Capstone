"""Handheld rig geometry from CAD -> the rig-file blocks dataproc needs (tcp.T_cam_tcp, cameras.*.T_imu_cam).

The handheld training gripper is iPhUMI's UMI gripper with our top cover (IMX519 on the 30 deg face, IMX219 and the
LSM6DSOX IMU on the 15 deg face). Every number below was read from two CAD exports (2026-11-12):

  ~/Downloads/vis_assembly.{x_t,stl}                    the full UMI gripper, incl. the soft fingers (assembly frame)
  ~/Downloads/gripper - top_cover_modified.step.STL     our top cover; same 172 x 91.4 mm base plate as UMI's top_cover,
                                                        so it sits in the assembly frame shifted by (-86, -24, 0) mm

Assembly frame (mm): x across the gripper (the fingers close along x), y forward toward the fingertips, z up toward the
top cover. Camera frames are OpenCV optical (x right in the image, y down, z out of the lens); T_a_b maps frame b to
frame a, as in dataproc (docs/GEOMETRY.md).

Check: iPhUMI's own calibration (TCP 211.4 mm ahead and 64.41 mm below the iPhone main camera, 15 deg mount) puts the
iPhone camera at (0, 13.8, 114.7), right at the front-top of the iphone_mount body in the same assembly.

Usage:
  python scripts/rig_geometry_from_cad.py                  # print the YAML blocks
  python scripts/rig_geometry_from_cad.py --check configs/rigs/rig_pi_v3.yaml   # compare with a rig file
Edit the constants below after re-measuring (e.g. the lens optical-centre offset), rerun, and paste the blocks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# --- CAD measurements (assembly frame, mm) ------------------------------------------------------------------------
TOP_COVER_SHIFT = np.array([-86.0, -24.0, 0.0])        # our STL -> assembly frame (base-plate footprint alignment)

# Fingertip point (TCP), iPhUMI convention: centred between the soft fingertips, at their very end.
TCP_POS = np.array([0.0, 201.35, -2.2])                # tips at y = 201.35; vertical middle of the tips (-15.1..10.7)
TCP_AXES = np.column_stack([[1, 0, 0],                 # x: across, the finger-closing direction ("right")
                            [0, 0, -1],                # y: down (away from the top cover)
                            [0, 1, 0]]).astype(float)  # z: forward, toward the fingertips

# Camera boards: centre of the 21 x 12.5 mm hole pattern on the tips of the 4 mm bosses (our STL coordinates).
# Re-measured independently on 2026-11-13 (all within 0.04 mm / 0.04 deg, faces exactly 15.00 and 30.00 deg).
IMX219_BOARD_STL = np.array([77.87, 40.84, 110.49])     # 15 deg face
IMX519_BOARD_STL = np.array([77.82, 52.165, 135.12])    # 30 deg face
IMX219_TILT_DEG = 15.0
IMX519_TILT_DEG = 30.0
# Lens optical centre relative to the hole pattern (user, 2026-11-13): each lens is in line with the LOWER hole pair
# (the pair further down the face), centred between the two holes; the boards sit directly on the 4 mm bosses (no
# spacers). Hole pattern 21 x 12.5 mm, so the lens is 12.5 / 2 down the face from the pattern centre.
LENS_DOWN_MM = 6.25
LENS_OUT_MM = 2.5             # optical centre out from the boss tips: ~1 mm PCB + ~1.5 mm lens. Nominal: +-2 mm; it
                              # only moves the camera along its optical axis (1:1 into "ahead" of T_cam_tcp).

# IMU (Modulino Movement, LSM6DSOX): centre of the 16 x 32 mm hole pattern on the 15 deg face (our STL coordinates).
IMU_STL = np.array([48.38, 31.57, 91.35])
IMU_FACE_TILT_DEG = 15.0      # the IMU is on the same face as the IMX219
# IMU axes as mounted, as seen by someone facing the camera lenses: x down the face, y to their right, z = x cross y
# (toward that viewer, i.e. out of the face toward the fingers, parallel to the IMX219 axis). Their right is -x.


def face_axes(tilt_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a face tilted tilt_deg down toward the fingers: (right, down-the-face, out-of-the-face toward the fingers)."""
    t = np.radians(tilt_deg)
    out = np.array([0.0, np.cos(t), -np.sin(t)])
    down = np.array([0.0, -np.sin(t), -np.cos(t)])
    right = np.cross(down, out)                        # = +x for an upright board
    return right, down, out


def pose(R: np.ndarray, p_mm: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p_mm / 1000.0                           # metres in the rig file
    return T


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def camera_pose(board_stl: np.ndarray, tilt_deg: float) -> np.ndarray:
    right, down, out = face_axes(tilt_deg)
    lens = board_stl + TOP_COVER_SHIFT + LENS_DOWN_MM * down + LENS_OUT_MM * out
    return pose(np.column_stack([right, down, out]), lens)   # OpenCV: x right, y down, z out of the lens


def imu_pose() -> np.ndarray:
    right, down, out = face_axes(IMU_FACE_TILT_DEG)
    x, y = down, -right                                # -right: right as seen facing the lenses (right is +x)
    return pose(np.column_stack([x, y, np.cross(x, y)]), IMU_STL + TOP_COVER_SHIFT)


def compute() -> dict[str, np.ndarray]:
    A_uw = camera_pose(IMX219_BOARD_STL, IMX219_TILT_DEG)
    A_main = camera_pose(IMX519_BOARD_STL, IMX519_TILT_DEG)
    A_imu = imu_pose()
    A_tcp = pose(TCP_AXES, TCP_POS)
    return {
        "T_cam_tcp": inv(A_uw) @ A_tcp,                 # TCP in the physical ultrawide optical frame
        "T_imu_cam_ultrawide": inv(A_imu) @ A_uw,
        "T_imu_cam_main": inv(A_imu) @ A_main,
    }


def as_yaml_matrix(T: np.ndarray, indent: str) -> str:
    rows = ",\n".join(indent + "  [" + ", ".join(f"{v: .9f}" for v in row) + "]" for row in T)
    return f"{indent}matrix: [\n{rows}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=Path, help="compare these values with a rig file (needs dataproc importable)")
    args = ap.parse_args()
    out = compute()
    t = out["T_cam_tcp"][:3, 3] * 1000
    print(f"# camera -> fingertips: {np.linalg.norm(t):.1f} mm "
          f"({t[2]:.1f} ahead, {t[1]:.1f} below, {t[0]:.1f} right of the IMX219 optical axis)")
    for key in ("T_cam_tcp", "T_imu_cam_main", "T_imu_cam_ultrawide"):
        print(f"# {key}:")
        print(as_yaml_matrix(out[key], "    "))
    if args.check:
        from dataproc.config import load_rig
        rig = load_rig(args.check)
        got = {"T_cam_tcp": rig.T_cam_tcp,
               "T_imu_cam_main": rig.cameras["main"].T_imu_cam,
               "T_imu_cam_ultrawide": rig.cameras["ultrawide"].T_imu_cam}
        bad = 0
        for key, T in out.items():
            dt = np.linalg.norm(got[key][:3, 3] - T[:3, 3]) * 1000
            dR = got[key][:3, :3].T @ T[:3, :3]
            # rotation angle from the skew part (robust near 0, unlike arccos of the trace)
            da = np.degrees(np.arcsin(min(1.0, np.linalg.norm([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                                                                 dR[1, 0] - dR[0, 1]]) / 2)))
            ok = dt < 0.01 and da < 0.001
            bad += not ok
            print(f"{'OK  ' if ok else 'DIFF'} {key}: {dt:.3f} mm, {da:.4f} deg")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
