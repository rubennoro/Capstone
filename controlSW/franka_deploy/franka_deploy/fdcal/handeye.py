"""Hand-eye calibration of the rig's (virtual) main camera on the Franka flange: F_T_C.

Data: N arm poses with a fixed ChArUco board in view. Per pose: O_T_F from the bridge (`get_state`, flange) and one
RigCamera virtual-main image (rectified pinhole, K known, no distortion). Board pose by ChArUco detection + PnP;
F_T_C by cv2.calibrateHandEye (Tsai and Park; both reported, Park used). Then F_T_TCP = F_T_C @ C_T_TCP (the
training TCP in the OpenCV camera frame) and the agreement with the CAD F_T_C (accept <= 3 mm / 1 deg).

    python -m franka_deploy fdcal handeye --data DIR [--cad-F_T_C cad.json] [--out result.json]
DIR/poses.json = [{"image": "000.png", "O_T_F": 4x4}, ...]; DIR/board.json = {"squares_x", "squares_y",
"square_m", "marker_m", "dict"}; DIR/K.json = 3x3 (default: the virtual main camera at 960x720).
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# training TCP in the OpenCV frame of the (virtual) main camera (PLAN_B §2.7, gripper_calibration iPhone17,1)
C_T_TCP = np.array([[1, 0, 0, 0], [0, 0.966, -0.259, 0.0644], [0, 0.259, 0.966, 0.2114], [0, 0, 0, 1.0]])
K_MAIN_960 = np.array([[703.1714, 0, 476.3752], [0, 703.1714, 361.3600], [0, 0, 1.0]])


def orthonormalize(T):
    T = np.array(T, float)
    u, _, vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = u @ vt
    return T


def make_board(cfg: dict):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.get("dict", "DICT_5X5_100")))
    return cv2.aruco.CharucoBoard((cfg["squares_x"], cfg["squares_y"]), cfg["square_m"], cfg["marker_m"], d)


def board_pose_from_image(img, board, K, min_corners: int = 6):
    det = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = det.detectBoard(img)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None
    obj = board.getChessboardCorners()[ch_ids.reshape(-1)]
    return pnp(obj, ch_corners.reshape(-1, 2), K)


def pnp(obj_pts, img_pts, K):
    ok, rvec, tvec = cv2.solvePnP(np.asarray(obj_pts, np.float64), np.asarray(img_pts, np.float64), K, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.reshape(3)
    return T


def solve_hand_eye(O_T_F_list, C_T_B_list, method="park") -> np.ndarray:
    m = {"tsai": cv2.CALIB_HAND_EYE_TSAI, "park": cv2.CALIB_HAND_EYE_PARK}[method]
    R_g2b = [T[:3, :3] for T in O_T_F_list]
    t_g2b = [T[:3, 3] for T in O_T_F_list]
    R_t2c = [T[:3, :3] for T in C_T_B_list]
    t_t2c = [T[:3, 3] for T in C_T_B_list]
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=m)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t.reshape(3)
    return T


def pose_diff(A, B):
    dt = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
    da = float(np.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))
    return dt, da


def calibrate(O_T_F_list, C_T_B_list, cad_F_T_C=None) -> dict:
    res = {}
    for m in ("tsai", "park"):
        res[m] = solve_hand_eye(O_T_F_list, C_T_B_list, m)
    F_T_C = res["park"]
    out = {"F_T_C": F_T_C.tolist(), "F_T_C_tsai": res["tsai"].tolist(), "tsai_vs_park_mm_deg": [pose_diff(res["tsai"], F_T_C)[0] * 1000, pose_diff(res["tsai"], F_T_C)[1]],
           "F_T_TCP": (F_T_C @ C_T_TCP).tolist(), "n_poses": len(O_T_F_list)}
    # consistency: board pose in base from every view should agree
    O_T_B = [F @ F_T_C @ C for F, C in zip(O_T_F_list, C_T_B_list)]
    p = np.array([T[:3, 3] for T in O_T_B])
    out["board_in_base_spread_mm"] = float(np.linalg.norm(p - p.mean(0), axis=1).max() * 1000)
    if cad_F_T_C is not None:
        dt, da = pose_diff(F_T_C, np.asarray(cad_F_T_C))
        out["cad_agreement"] = {"mm": dt * 1000, "deg": da, "accept": dt <= 0.003 and da <= 1.0}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cad-F_T_C", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    poses = json.load(open(os.path.join(a.data, "poses.json")))
    board = make_board(json.load(open(os.path.join(a.data, "board.json"))))
    kp = os.path.join(a.data, "K.json")
    K = np.array(json.load(open(kp))) if os.path.exists(kp) else K_MAIN_960
    F, C = [], []
    for p in poses:
        img = cv2.imread(os.path.join(a.data, p["image"]), cv2.IMREAD_GRAYSCALE)
        T = board_pose_from_image(img, board, K)
        if T is not None:
            F.append(orthonormalize(p["O_T_F"]))
            C.append(T)
    if len(F) < 5:
        print(f"only {len(F)} usable views (need >= 5)")
        return 1
    cad = json.load(open(a.cad_F_T_C)) if a.cad_F_T_C else None
    out = calibrate(F, C, cad)
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0 if (cad is None or out["cad_agreement"]["accept"]) else 2
