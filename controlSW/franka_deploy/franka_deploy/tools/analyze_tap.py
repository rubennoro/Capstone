"""Analyse a FrankaPromptTask tap directory (one npz per policy call).

    python -m franka_deploy.tools.analyze_tap --tap-dir T [--replay-npz R --start-now-ns N --zarr Z] [--out J]

Reports: contract check of every `obs_to_send` (keys, shapes, dtypes) and reply; main/ultrawide sample spacing and
ultrawide age (newest main capture - newest UW packet time) histograms; inference RPC latency and capture->reply
latency histograms; history-selection check (proprio history == the expected entries of the previous chunk);
with --replay-npz: teacher-forcing per-step errors of the predicted RELATIVE chunk against the demo's relative future
from the image time (k = 4, 8, 15), and drift of the FK anchor; with --zarr (env_bp): MAE at 224 between tapped
images and the zarr frames they were rendered from.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

CONTRACT_OBS = {
    "robot0_eef_pos": ((2, 3), "float32"), "robot0_eef_rot_axis_angle": ((2, 6), "float32"),
    "robot0_gripper_width": ((2, 1), "float32"), "robot0_eef_rot_axis_angle_wrt_start": ((2, 6), "float32"),
    "robot0_main_camera": ((2, 720, 960, 3), "uint8"), "robot0_ultrawide_camera": ((2, 720, 960, 3), "uint8"),
    "episode_idx": ((), "int64"), "task_name": ((), "<U11"),
}
CONTRACT_REPLY = {"action0_eef_pos": (16, 3), "action0_eef_rot_axis_angle": (16, 6), "action0_gripper_width": (16, 1)}


def hist(x, bins):
    h, e = np.histogram(np.asarray(x), bins=bins)
    return {"edges": [round(float(v), 4) for v in e], "counts": h.tolist()}


def stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return {}
    return {"n": int(x.size), "min": float(x.min()), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "max": float(x.max()), "mean": float(x.mean())}


def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def load(tap_dir):
    files = sorted(glob.glob(os.path.join(tap_dir, "call_*.npz")))
    return files, [dict(np.load(f, allow_pickle=False)) for f in files]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tap-dir", required=True)
    ap.add_argument("--replay-npz", default=None)
    ap.add_argument("--start-now-ns", type=int, default=None)
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--exclude-loop-edge-s", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    files, calls = load(a.tap_dir)
    res: dict = {"n_calls": len(calls)}
    bad = []
    for f, c in zip(files, calls):
        for k, (shape, dtype) in CONTRACT_OBS.items():
            if f"meta_shape_{k}" not in c:
                bad.append((os.path.basename(f), k, "missing"))
                continue
            if tuple(c[f"meta_shape_{k}"].tolist()) != shape or str(c[f"meta_dtype_{k}"]) != dtype:
                bad.append((os.path.basename(f), k, tuple(c[f"meta_shape_{k}"].tolist()), str(c[f"meta_dtype_{k}"])))
        extra = sorted(k[len("meta_shape_"):] for k in c if k.startswith("meta_shape_") and k[len("meta_shape_"):] not in CONTRACT_OBS)
        if extra:
            bad.append((os.path.basename(f), "extra keys", extra))
        for k, shape in CONTRACT_REPLY.items():
            if c[f"reply_{k}"].shape != shape:
                bad.append((os.path.basename(f), k, c[f"reply_{k}"].shape))
        if not np.allclose(c["obs_robot0_eef_pos"][-1], 0, atol=1e-7) or not np.allclose(
                c["obs_robot0_eef_rot_axis_angle"][-1], [1, 0, 0, 0, 1, 0], atol=1e-6):
            bad.append((os.path.basename(f), "last proprio not identity"))
    res["contract_violations"] = bad[:20]
    res["contract_ok"] = not bad
    main_dt = [float(np.diff(c["info_main_ts"])[0]) for c in calls]
    uw_dt = [float(np.diff(c["info_uw_ts"])[0]) for c in calls]
    uw_age = [float(c["info_main_ts"][-1] - c["info_uw_ts"][-1]) for c in calls]
    inf = [float(c["info_inference_s"]) for c in calls]
    cap_to_reply = [float(c["info_t_reply"] - c["info_main_ts"][-1]) for c in calls]
    res["main_spacing_s"] = stats(main_dt)
    res["uw_spacing_s"] = stats(uw_dt)
    res["main_spacing_ok"] = bool(np.all(np.abs(np.array(main_dt) - 0.05) <= 0.017))
    res["uw_spacing_ok"] = bool(np.all(np.abs(np.array(uw_dt) - 0.10) <= 0.017))
    res["uw_age_hist_s"] = hist(uw_age, [-0.05, 0, 0.017, 0.033, 0.05, 0.067, 0.083, 0.1, 0.15, 0.2])
    res["uw_age_s"] = stats(uw_age)
    res["inference_rpc_s"] = {**stats(inf), "hist": hist(inf, [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.75, 1, 2])}
    res["capture_to_reply_s"] = stats(cap_to_reply)
    res["state_age_s"] = stats([float(c["info_state_age_s"]) for c in calls])
    res["anchor_used"] = {k: int(sum(str(c["info_anchor_used"]) == k for c in calls)) for k in ("measured", "target")}
    if "info_controller_counters" in calls[-1]:
        res["controller_counters_last"] = json.loads(str(calls[-1]["info_controller_counters"]))
    res["n_with_images"] = sum("img224_robot0_main_camera" in c for c in calls)

    if a.replay_npz:
        from scipy.spatial.transform import Rotation as R

        z = np.load(a.replay_npz)
        t_ns, P, origin, span = z["t_ns"], z["xyz_wxyz"], int(z["origin_ns"]), int(z["span_ns"])
        rot = R.from_quat(P[:, [4, 5, 6, 3]])

        def tau_of(t):
            return (int(t * 1e9) - a.start_now_ns) % span + origin

        def pose_at(tau):
            tau = np.clip(tau, t_ns[0], t_ns[-1])
            i = int(np.clip(np.searchsorted(t_ns, tau), 1, len(t_ns) - 1))
            w = (tau - t_ns[i - 1]) / max(1, t_ns[i] - t_ns[i - 1])
            pos = (1 - w) * P[i - 1, :3] + w * P[i, :3]
            from scipy.spatial.transform import Slerp

            r = Slerp([0, 1], R.concatenate([rot[i - 1], rot[i]]))(w)
            M = np.eye(4)
            M[:3, :3], M[:3, 3] = r.as_matrix(), pos
            return M

        errs = {k: {"pos_mm": [], "rot_deg": [], "width_mm": []} for k in (0, 4, 8, 15)}
        used = 0
        for c in calls:
            T = float(c["info_main_ts"][-1])
            tau0 = tau_of(T)
            if tau0 - origin < a.exclude_loop_edge_s * 1e9 or origin + span - tau0 < (0.8 + a.exclude_loop_edge_s) * 1e9:
                continue
            used += 1
            M0 = pose_at(tau0)
            Rp = rot6d_to_mat(c["reply_action0_eef_rot_axis_angle"])
            for k in errs:
                Mk = np.linalg.inv(M0) @ pose_at(tau0 + int(k * 0.05e9))
                dp = np.linalg.norm(c["reply_action0_eef_pos"][k] - Mk[:3, 3]) * 1000
                cosv = (np.trace(Rp[k].T @ Mk[:3, :3]) - 1) / 2
                errs[k]["pos_mm"].append(dp)
                errs[k]["rot_deg"].append(np.degrees(np.arccos(np.clip(cosv, -1, 1))))
                # width: absolute
                i = int(np.clip(np.searchsorted(t_ns, tau0 + int(k * 0.05e9)), 0, len(t_ns) - 1))
                errs[k]["width_mm"].append(abs(float(c["reply_action0_gripper_width"][k, 0]) - float(z["width"][i])) * 1000)
        res["teacher_forcing"] = {"n_used": used, **{f"k{k}": {m: float(np.median(v)) for m, v in e.items()} for k, e in errs.items()}}

    if a.zarr and a.replay_npz:
        import zarr
        from behavior_prompting.common.imagecodecs_numcodecs import register_codecs

        register_codecs(verbose=False)
        root = zarr.open(zarr.ZipStore(a.zarr, mode="r"), mode="r")
        zs, zl = int(z["zarr_start"]), z["zarr_local"]
        maes = []
        for c in calls:
            if "img224_robot0_main_camera" not in c:
                continue
            for j in range(2):
                tau = tau_of(float(c["info_main_ts"][j]))
                i = int(np.argmin(np.abs(t_ns - tau)))
                ref = root["data/camera_right_main_rgb"][zs + int(zl[i])]
                maes.append(float(np.abs(c["img224_robot0_main_camera"][j].astype(float) - ref.astype(float)).mean()))
                if os.environ.get("TAPDBG"): print("dbg", j, i, (tau - t_ns[i]) / 1e6, maes[-1])
        res["image_mae_224_main_vs_zarr"] = stats(maes)
        # ultrawide age, expressed against training: training pairs main frame i with the newest UW frame at or before
        # it (meta/upsample_index). Offset = (UW frame actually shown) - (UW frame training would use), in UW frames
        # (10 Hz -> 100 ms each); 0 = same as training, -1 = one frame older, +1 = newer.
        up = root["meta"]["upsample_index_camera_right_ultrawide_rgb"]
        uwa = root["data/camera_right_ultrawide_rgb"]
        offs = []
        for c in calls:
            if "img224_robot0_ultrawide_camera" not in c:
                continue
            tau = tau_of(float(c["info_main_ts"][-1]))
            i = int(np.argmin(np.abs(t_ns - tau)))
            u_train = int(up[zs + int(zl[i])])
            cand = list(range(max(0, u_train - 3), min(uwa.shape[0], u_train + 4)))
            m = [np.abs(c["img224_robot0_ultrawide_camera"][-1].astype(float) - uwa[j].astype(float)).mean() for j in cand]
            offs.append(cand[int(np.argmin(m))] - u_train)
        res["uw_frame_offset_vs_training"] = {int(k): int(v) for k, v in zip(*np.unique(offs, return_counts=True))}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
