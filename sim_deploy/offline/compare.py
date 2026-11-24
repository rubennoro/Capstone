"""W1 metrics: deploy-path vs direct-path inputs, chunk vs chunk / ground truth per horizon step, and the montage PNG.
numpy + cv2 only (both envs). Library module; used by build_obs.py and policy_runner.py.

Chunks are (16, 10) arrays in the model's action layout [pos (3) m, rot6d (6), width (1) m], relative to the anchor
(the pose the observation was made relative to). Per step: position |dp| mm, rotation geodesic angle deg (rot6d ->
matrix with the copy's pose_math.matrix_from_rot6d; the angle of R_a^T R_b is the same in the row or column rot6d
convention), width |dw| mm.
"""

from __future__ import annotations

import numpy as np

LOWDIM = ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m")
CAMS = ("main", "ultrawide")


def image_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """a vs b (H, W, 3) uint8: max |diff|, mean abs diff (0-255), per-channel MAD, PSNR dB."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(a64 - b64)
    mse = float(np.mean((a64 - b64) ** 2))
    return {"max_abs": int(d.max()), "mad": round(float(d.mean()), 4),
            "mad_rgb": [round(float(x), 3) for x in d.reshape(-1, 3).mean(0)],
            "psnr_db": round(float(10 * np.log10(255.0 ** 2 / mse)), 2) if mse > 0 else 99.0}


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    from policy_deploy.manager.pose_math import matrix_from_rot6d
    return matrix_from_rot6d(np.asarray(d6, dtype=np.float64))


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between (..., 3, 3) rotation stacks, degrees."""
    cos = (np.einsum("...ij,...ij->...", Ra, Rb) - 1.0) / 2.0          # trace(Ra^T Rb)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def chunk_errors(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Per horizon step (16): pos mm, rot deg, width mm; plus max / mean over the steps."""
    pred, ref = np.asarray(pred, np.float64), np.asarray(ref, np.float64)
    pos = np.linalg.norm(pred[:, :3] - ref[:, :3], axis=-1) * 1000.0
    rot = rot_angle_deg(_rot6d_to_mat(pred[:, 3:9]), _rot6d_to_mat(ref[:, 3:9]))
    width = np.abs(pred[:, 9] - ref[:, 9]) * 1000.0
    return {"pos_mm": np.round(pos, 4).tolist(), "rot_deg": np.round(rot, 4).tolist(),
            "width_mm": np.round(width, 4).tolist(),
            "max": {"pos_mm": float(pos.max()), "rot_deg": float(rot.max()), "width_mm": float(width.max())},
            "mean": {"pos_mm": float(pos.mean()), "rot_deg": float(rot.mean()), "width_mm": float(width.mean())},
            "at_k": {f"k{k}": {"pos_mm": float(pos[k]), "rot_deg": float(rot[k]), "width_mm": float(width[k])}
                     for k in (0, 4, 8, 15)}}


def chunk_array(eef_pos, eef_rot6d, width) -> np.ndarray:
    return np.concatenate([np.asarray(eef_pos, np.float32).reshape(-1, 3), np.asarray(eef_rot6d, np.float32)
                           .reshape(-1, 6), np.asarray(width, np.float32).reshape(-1, 1)], axis=-1)


def pose_errors(Ta: np.ndarray, Tb: np.ndarray) -> dict:
    """(n, 4, 4) vs (n, 4, 4): per pose position mm and rotation deg."""
    pos = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=-1) * 1000.0
    rot = rot_angle_deg(Ta[:, :3, :3], Tb[:, :3, :3])
    return {"pos_mm": np.round(pos, 4).tolist(), "rot_deg": np.round(rot, 4).tolist(),
            "max_pos_mm": float(pos.max()), "max_rot_deg": float(rot.max())}


def compare_inputs(z: dict, meta: dict, used_fields) -> dict:
    """Deploy vs direct for one cached sample (``z``: the merged npz arrays, ``meta``: its json)."""
    dep, tf = meta["deploy"], meta["deploy"]["train_frames"]
    df = dep.get("deploy_frames", {})
    out: dict = {"used_fields": list(used_fields), "images": {}, "lowdim": {}, "times_ms": {}}
    for cam in CAMS:
        key = "main" if cam == "main" else "uw"
        for k in range(2):
            m = image_metrics(z[f"deploy_{cam}"][k], z[f"direct_{cam}"][k])
            m["deploy_frame_idx"] = df.get(f"{key}_frame_idx", [None, None])[k]
            m["train_frame_idx"] = tf[f"{key}_frame_idx"][k]
            m["same_frame"] = m["deploy_frame_idx"] == m["train_frame_idx"]
            out["images"][f"{cam}_{k}"] = m
    for k in LOWDIM:
        d = np.abs(z[f"deploy_{k}"].astype(np.float64) - z[f"direct_{k}"].astype(np.float64))
        out["lowdim"][k] = {"max_abs": float(d.max()), "per_step_max_abs": d.max(axis=-1).tolist()}
    d = np.abs(z["deploy_eef_rot6d_wrt_start"].astype(np.float64) - z["direct_wrt_start_clean"].astype(np.float64))
    out["lowdim"]["eef_rot6d_wrt_start_vs_noiseless"] = {"max_abs": float(d.max())}
    if "deploy_gripper_width_m_B" in z:
        out["lowdim"]["gripper_width_m_B_robot"] = z["deploy_gripper_width_m_B"].ravel().tolist()
    # eef_pos[0]: the older pose relative to the anchor; its length tells the time span the two poses cover
    n_dep, n_dir = np.linalg.norm(z["deploy_eef_pos"][0]), np.linalg.norm(z["direct_eef_pos"][0])
    out["lowdim"]["eef_pos0_norm_mm"] = {"deploy": float(n_dep * 1000), "direct": float(n_dir * 1000),
                                         "ratio": float(n_dep / n_dir) if n_dir > 0 else None}
    row_t = tf["row_t_ns"]
    if "main_t_ns" in df:
        out["times_ms"]["deploy_main_minus_rows"] = [(a - b) / 1e6 for a, b in zip(df["main_t_ns"], row_t)]
        out["times_ms"]["deploy_main_spacing"] = (df["main_t_ns"][1] - df["main_t_ns"][0]) / 1e6
        out["times_ms"]["deploy_uw_minus_row_T"] = [(a - row_t[1]) / 1e6 for a in df["uw_frame_t_ns"]]
    out["times_ms"]["train_main_minus_rows"] = [(a - b) / 1e6 for a, b in zip(tf["main_frame_t_ns"], row_t)]
    out["times_ms"]["train_lowdim_spacing"] = (row_t[1] - row_t[0]) / 1e6
    out["times_ms"]["train_uw_minus_row_T"] = [(a - row_t[1]) / 1e6 for a in tf["uw_frame_t_ns"]]
    imgs = out["images"]
    same = [v for v in imgs.values() if v["same_frame"]]
    out["summary"] = {"frames_same": f"{len(same)}/4",
                      "same_frame_max_abs": max((v["max_abs"] for v in same), default=None),
                      "same_frame_mad_max": max((v["mad"] for v in same), default=None),
                      "lowdim_max_abs": {k: out["lowdim"][k]["max_abs"] for k in LOWDIM}}
    return out


def montage(z: dict, path, title: str = "") -> None:
    """2 rows (deploy, direct) x 4 images (main T-1, main T, uw T-1, uw T) + |diff| x4 row, labelled PNG."""
    import cv2
    rows = []
    for name in ("deploy", "direct"):
        imgs = [z[f"{name}_main"][0], z[f"{name}_main"][1], z[f"{name}_ultrawide"][0], z[f"{name}_ultrawide"][1]]
        rows.append(np.concatenate(imgs, axis=1))
    diff = np.clip(np.abs(rows[0].astype(np.int16) - rows[1].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
    rows.append(diff)
    labels = ["deploy path (Pi 640 JPEG -> C1)", "direct path (training dataset)", "|deploy - direct| x4"]
    out = []
    for r, lab in zip(rows, labels):
        bar = np.full((22, r.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        out += [bar, r]
    img = np.concatenate(out, axis=0)
    if title:
        bar = np.full((24, img.shape[1], 3), 230, np.uint8)
        cv2.putText(bar, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = np.concatenate([bar, img], axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
