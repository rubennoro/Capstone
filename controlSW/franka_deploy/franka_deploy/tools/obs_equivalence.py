"""Deployment-shaped observations vs the training dataset dict (D1c), env_bp side.

Three stages, two environments (real-env's PolicyAgent lives in the numpy-2 env, the model in env_bp; files are .npz):
  1. export  (env_bp):   sample N training-dataset items; for each, write the raw deployment inputs taken from the
                         replay buffer itself (absolute TCP poses as xyz_wxyz, widths, uint8 224 images of exactly the
                         frames the sampler used) + the dataset's own model dict, prompt and GT action (pickle).
  2. capture (env_realenv, franka_deploy.tools.agent_capture): run the unchanged real-env PolicyAgent.predict_actions
                         on those inputs with a fake RMQ client; save the exact `obs_to_send` and the absolute actions.
  3. compare (env_bp):   same prompt, same diffusion noise (seed): model output from the dataset dict vs from the
                         captured deployment obs (with and without *_wrt_start) and, optionally (--rig), with the
                         images replaced by RigCamera-path 960x720 frames rendered from the original iPhone frames.

    python -m franka_deploy.tools.obs_equivalence export  --ckpt C --work-dir W --dataset-path Z --out-dir D [--n-val 12 --n-train 12]
    python -m franka_deploy.tools.agent_capture --in-dir D                                    # env_realenv
    python -m franka_deploy.tools.obs_equivalence compare --ckpt C --work-dir W --in-dir D [--rig --raw-dir R] --out J
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import socket

import numpy as np

K_STEPS = (0, 4, 8, 15)


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _to_numpy_tree(t):
    import torch

    if isinstance(t, dict):
        return {k: _to_numpy_tree(v) for k, v in t.items()}
    if torch.is_tensor(t):
        return t.numpy()
    return t


def rotvec_to_wxyz(rv: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_rotvec(rv).as_quat()  # xyzw
    return np.concatenate([q[:, 3:], q[:, :3]], axis=1)


def sample_frame_indices(ds, idx: int) -> dict:
    """Re-derive the replay-buffer indices SequenceSampler.sample_sequence uses for dataset item idx."""
    smp, rb = ds.sampler, ds.replay_buffer
    cur_within, ep, task = (int(x) for x in smp.indices[idx])
    task_len = int(rb.task_lengths[task])
    end = int(rb.task_data_ends[task])
    start = end - task_len
    cur = start + cur_within
    out = {"episode_idx": ep, "task_idx": task, "start": start, "end": end, "cur": cur}
    for key in smp.lowdim_keys:
        h, ds_steps = smp.key_horizon[key], smp.key_down_sample_steps[key]
        ix = [cur - i * ds_steps for i in range(h)][::-1]
        ix = [x for x in ix if x >= start]
        ix = [ix[0]] * (h - len(ix)) + ix
        out[key] = ix
    for key in smp.rgb_keys:
        h, ds_steps = smp.key_horizon[key], smp.key_down_sample_steps[key]
        if rb.is_key_upsampled(key):
            uc, us = rb.map_upsample_index(key, cur), rb.map_upsample_index(key, start)
            nv = min(h, (uc - us) // ds_steps + 1)
            ix = list(range(uc - (nv - 1) * ds_steps, uc + 1, ds_steps))
        else:
            nv = min(h, (cur - start) // ds_steps + 1)
            ix = list(range(cur - (nv - 1) * ds_steps, cur + 1, ds_steps))
        ix = [ix[0]] * (h - len(ix)) + ix
        out[key] = [int(x) for x in ix]
    ends = np.asarray(rb.episode_ends[:])
    ep_start = int(ends[ep - 1]) if ep > 0 else 0
    out["main_local"] = [x - ep_start for x in out["camera_right_main_rgb"]]
    uw_ends = np.asarray(rb.meta["episode_ends_camera_right_ultrawide_rgb"][:])
    uw_start = int(uw_ends[ep - 1]) if ep > 0 else 0
    out["uw_local"] = [x - uw_start for x in out["camera_right_ultrawide_rgb"]]
    a_h, a_ds = smp.key_horizon["action"], smp.key_down_sample_steps["action"]
    fut = [min(cur + k * a_ds, end - 1) for k in range(a_h)]
    out["action"] = fut
    return out


def export(a):
    from franka_deploy.inference.server import build_node
    from franka_deploy.tools.offline_eval import build_eval_datasets

    node, _ = build_node(a.ckpt, a.work_dir, None, f"tcp://127.0.0.1:{_free_port()}", a.device)
    ds, val = build_eval_datasets(node.cfg, a.dataset_path)
    rb = ds.replay_buffer
    names = [str(x) for x in rb.meta["episode_names"][:]] if "episode_names" in rb.meta else None
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    jobs = [("val", val, i) for i in rng.choice(len(val), a.n_val, replace=False)] + \
           [("train", ds, i) for i in rng.choice(len(ds), a.n_train, replace=False)]
    summary = []
    for n, (split, d, i) in enumerate(jobs):
        item = d[int(i)]
        fi = sample_frame_indices(d, int(i))
        pre = "gripper_right"
        pos = rb[f"{pre}_eef_pos"][fi[f"{pre}_eef_pos"]].astype(np.float64)
        rv = rb[f"{pre}_eef_rot_axis_angle"][fi[f"{pre}_eef_rot_axis_angle"]].astype(np.float64)
        width = rb[f"{pre}_gripper_width"][fi[f"{pre}_gripper_width"]].astype(np.float64)
        main = rb["camera_right_main_rgb"][fi["camera_right_main_rgb"]]
        uw = rb["camera_right_ultrawide_rgb"][fi["camera_right_ultrawide_rgb"]]
        obs = _to_numpy_tree(item["obs"])
        # the uint8 frames must be exactly what the dataset turned into floats
        img_ok = bool(np.array_equal(np.moveaxis(main, -1, 1).astype(np.float32) / 255.0, obs["camera_right_main_rgb"])
                      and np.array_equal(np.moveaxis(uw, -1, 1).astype(np.float32) / 255.0, obs["camera_right_ultrawide_rgb"]))
        fpos = rb[f"{pre}_eef_pos"][fi["action"]].astype(np.float64)
        frv = rb[f"{pre}_eef_rot_axis_angle"][fi["action"]].astype(np.float64)
        fwid = rb[f"{pre}_gripper_width"][fi["action"]].astype(np.float64)
        ep_name = names[fi["episode_idx"]] if names else str(fi["episode_idx"])
        start_pose = np.concatenate([rb[f"{pre}_eef_pos"][fi["start"]], rotvec_to_wxyz(rb[f"{pre}_eef_rot_axis_angle"][fi["start"]][None])[0]])
        tag = f"{n:03d}_{split}"
        np.savez(os.path.join(a.out_dir, f"deploy_in_{tag}.npz"),
                 eef_xyz_wxyz=np.concatenate([pos, rotvec_to_wxyz(rv)], 1), gripper_width=width,
                 main=main, ultrawide=uw, start_xyz_wxyz=start_pose,
                 gt_action=item["action"].numpy(), gt_future_xyz_wxyz=np.concatenate([fpos, rotvec_to_wxyz(frv)], 1),
                 gt_future_width=fwid)
        prompt = obs.pop("prompt")
        with open(os.path.join(a.out_dir, f"ref_{tag}.pkl"), "wb") as f:
            pickle.dump({"obs": obs, "prompt": prompt, "action": item["action"].numpy(), "split": split, "ds_index": int(i),
                         "episode": ep_name, "frames": fi}, f)
        summary.append({"tag": tag, "split": split, "ds_index": int(i), "episode": ep_name, "cur": fi["cur"],
                        "main_idx": fi["camera_right_main_rgb"], "uw_idx": fi["camera_right_ultrawide_rgb"],
                        "uint8_images_match_dataset": img_ok})
        print(summary[-1], flush=True)
    with open(os.path.join(a.out_dir, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


# ------------------------------------------------------------------------------------------------ rig path
class _RawVideo:
    """Finds the raw iPhone video frame that a zarr 224 frame came from (min MAE in a window) and returns it."""

    def __init__(self, path):
        import av

        self.path = path
        self.small, self.full_cache = [], {}
        from behavior_prompting.common.cv_util import get_image_transform_with_border

        with av.open(path) as c:
            for fr in c.decode(video=0):
                rgb = fr.to_ndarray(format="rgb24")
                if not self.small:
                    self.tf = get_image_transform_with_border(in_res=(rgb.shape[1], rgb.shape[0]), out_res=(224, 224))
                self.small.append(self.tf(rgb))

    def match(self, img224: np.ndarray, guess: int, window: int):
        lo, hi = max(0, guess - window), min(len(self.small), guess + window + 1)
        maes = [np.abs(self.small[j].astype(np.int16) - img224.astype(np.int16)).mean() for j in range(lo, hi)]
        j = int(np.argmin(maes))
        return lo + j, float(maes[j])

    def frame_bgr(self, index: int):
        import av

        if index not in self.full_cache:
            with av.open(self.path) as c:
                for i, fr in enumerate(c.decode(video=0)):
                    if i == index:
                        self.full_cache[index] = fr.to_ndarray(format="bgr24")
                        break
        return self.full_cache[index]


def rig_images(raw_dir, episode, ref_obs, frames, cache, rig_cfg):
    """960x720 RGB main/ultrawide pairs from the RigCamera path (default rig, iPhone intrinsics of this demo)."""
    import cv2
    from sensor_rig.deploy.frame_source import RigFrameSource
    from sensor_rig.drivers.base import Frame
    from sensor_rig.rig_config import load_rig
    from test_frame_source import FakeSource

    d = os.path.join(raw_dir, episode + "_left")
    js = json.load(open(glob.glob(os.path.join(d, "*.json"))[0]))
    stem = os.path.join(d, os.path.basename(d))
    out, info = {"main": [], "ultrawide": []}, {"match_mae": {"main": [], "ultrawide": []}, "raw_idx": {"main": [], "ultrawide": []}}
    vids = {}
    for cam, suffix, key in (("main", "_rgb.mp4", "camera_right_main_rgb"),
                             ("ultrawide", "_ultrawidergb.mp4", "camera_right_ultrawide_rgb")):
        vk = stem + suffix
        if vk not in cache:
            cache[vk] = _RawVideo(vk)
        vids[cam] = cache[vk]
        for t in range(2):
            z224 = (np.moveaxis(ref_obs[key][t], 0, -1) * 255.0).round().astype(np.uint8)
            local = frames["main_local" if cam == "main" else "uw_local"][t]
            j, m = vids[cam].match(z224, guess=local, window=120 if cam == "main" else 30)
            info["match_mae"][cam].append(m)
            info["raw_idx"][cam].append(j)
    rig = load_rig(rig_cfg)
    rig = rig.with_virtual_reference("main", js["mainCameraIntrinsics"], (1920, 1440))
    rig = rig.with_virtual_reference("ultrawide", js["ultrawideCameraIntrinsics"], (640, 480))
    for t in range(2):
        m_bgr = vids["main"].frame_bgr(info["raw_idx"]["main"][t])
        u_bgr = vids["ultrawide"].frame_bgr(info["raw_idx"]["ultrawide"][t])
        phys = {"main": rig.inverse_remap("main").apply(m_bgr), "ultrawide": rig.inverse_remap("ultrawide").apply(u_bgr)}
        ms, us = FakeSource("main"), FakeSource("ultrawide")
        t0 = 7_000_000_000_000
        ms.push(Frame(phys["main"], t0, 0, t0))
        us.push(Frame(phys["ultrawide"], t0 + 5_000_000, 0, t0))
        ms.finished = us.finished = True
        src = RigFrameSource(rig, sources={"main": ms, "ultrawide": us}, sync="lockstep")
        src.start()
        pkt = src.next_packet()
        for cam in ("main", "ultrawide"):
            img = pkt.images[cam]
            assert img.shape == (720, 960, 3)
            out[cam].append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BaseCamera publishes RGB
    return np.stack(out["main"]), np.stack(out["ultrawide"]), info


def compare(a):
    import torch
    from franka_deploy.inference.server import build_node
    from franka_deploy.tools.offline_eval import per_step_errors

    node, _ = build_node(a.ckpt, a.work_dir, None, f"tcp://127.0.0.1:{_free_port()}", a.device)
    pol = node.policy
    dev = node.device
    refs = sorted(glob.glob(os.path.join(a.in_dir, "ref_*.pkl")))
    rows, cache = [], {}
    P = {"ref": [], "dep": [], "rig": [], "gt": []}

    def predict(model_obs, seed):
        t = {k: torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0).to(dev) for k, v in model_obs.items()}
        torch.manual_seed(seed)
        with torch.inference_mode():
            return pol.predict_action(t)["action_pred"][0].float().cpu().numpy()

    for rp in refs:
        tag = os.path.basename(rp)[4:-4]
        ref = pickle.load(open(rp, "rb"))
        cap = dict(np.load(os.path.join(a.in_dir, f"captured_{tag}.npz"), allow_pickle=False))
        cap_nowrt = dict(np.load(os.path.join(a.in_dir, f"captured_nowrt_{tag}.npz"), allow_pickle=False))
        pol.reset()
        prompt = {k: v for k, v in ref["prompt"].items()}
        from behavior_prompting.common.pytorch_util import dict_apply

        pt = dict_apply(prompt, lambda x: torch.from_numpy(np.array(x)).unsqueeze(0).to(dev))
        with torch.inference_mode():
            pol.prompt(pt)
        ref_obs = {k: v for k, v in ref["obs"].items()}
        dep_obs = node._process_obs({k: v for k, v in cap.items() if not k.startswith("abs_")})
        dep_nowrt = node._process_obs({k: v for k, v in cap_nowrt.items() if not k.startswith("abs_")})
        in_diff = {k: float(np.abs(dep_obs[k] - ref_obs[k]).max()) for k in dep_obs if k in ref_obs and "wrt_start" not in k}
        row = {"tag": tag, "split": ref["split"], "episode": ref["episode"], "input_max_abs_diff": in_diff,
               "keys_dep": sorted(dep_obs), "keys_nowrt": sorted(dep_nowrt)}
        da, dn = [], []
        for s in a.seeds:
            r_ = predict(ref_obs, s)
            d_ = predict(dep_obs, s)
            n_ = predict(dep_nowrt, s)
            da.append(float(np.abs(r_ - d_).max()))
            dn.append(float(np.abs(d_ - n_).max()))
            P["ref"].append(r_)
            P["dep"].append(d_)
            P["gt"].append(ref["action"])
        row["max_abs_action_diff_ref_vs_dep"] = max(da)
        row["max_abs_action_diff_dep_vs_nowrt"] = max(dn)
        row["abs_action_path_err"] = {k: float(v) for k, v in cap.items() if k.startswith("abs_err")}
        if a.rig:
            main, uw, info = rig_images(a.raw_dir, ref["episode"], ref_obs, ref["frames"], cache, a.rig_config)
            rig_in = {k: v for k, v in cap.items() if not k.startswith("abs_")}
            rig_in["robot0_main_camera"], rig_in["robot0_ultrawide_camera"] = main, uw
            rig_obs = node._process_obs(rig_in)
            imgs = {}
            for k in ("camera_right_main_rgb", "camera_right_ultrawide_rgb"):
                imgs[k] = float(np.abs(rig_obs[k] - ref_obs[k]).mean() * 255)
            row["rig_image_mae_224"] = imgs
            row["rig_match_mae_raw_vs_zarr"] = info["match_mae"]
            row["rig_raw_idx"] = info["raw_idx"]
            for s in a.seeds:
                P["rig"].append(predict(rig_obs, s))
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k not in ("keys_dep", "keys_nowrt")}), flush=True)

    gt = np.stack(P["gt"])
    res = {"n_samples": len(rows), "seeds": a.seeds, "rows": rows,
           "max_abs_action_diff_ref_vs_dep": max(r["max_abs_action_diff_ref_vs_dep"] for r in rows),
           "max_abs_action_diff_dep_vs_nowrt": max(r["max_abs_action_diff_dep_vs_nowrt"] for r in rows),
           "max_input_abs_diff": max(max(r["input_max_abs_diff"].values()) for r in rows),
           "mse_ref_vs_gt": float(((np.stack(P["ref"]) - gt) ** 2).mean()),
           "mse_dep_vs_gt": float(((np.stack(P["dep"]) - gt) ** 2).mean()),
           "per_step_ref": per_step_errors(np.stack(P["ref"]), gt), "per_step_dep": per_step_errors(np.stack(P["dep"]), gt)}
    if a.rig:
        rig = np.stack(P["rig"])
        res["mse_rig_vs_gt"] = float(((rig - gt) ** 2).mean())
        res["rig_vs_224_mse_ratio"] = res["mse_rig_vs_gt"] / res["mse_dep_vs_gt"]
        res["rig_image_mae_224_max"] = {k: max(r["rig_image_mae_224"][k] for r in rows) for k in rows[0]["rig_image_mae_224"]}
        res["rig_image_mae_224_mean"] = {k: float(np.mean([r["rig_image_mae_224"][k] for r in rows])) for k in rows[0]["rig_image_mae_224"]}
        res["rig_match_mae_max"] = {c: max(max(r["rig_match_mae_raw_vs_zarr"][c]) for r in rows) for c in ("main", "ultrawide")}
        res["per_step_rig"] = per_step_errors(rig, gt)
        res["rig_vs_224_action_max_abs"] = float(np.abs(rig - np.stack(P["dep"])).max())
        res["rig_vs_224_action_mean_abs"] = float(np.abs(rig - np.stack(P["dep"])).mean())
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--work-dir", required=True)
    e.add_argument("--dataset-path", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--n-val", type=int, default=12)
    e.add_argument("--n-train", type=int, default=12)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--work-dir", required=True)
    c.add_argument("--in-dir", required=True)
    c.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    c.add_argument("--rig", action="store_true")
    c.add_argument("--raw-dir", default=None)
    c.add_argument("--rig-config", default="rig_default.yaml")
    c.add_argument("--device", default="cuda")
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    for k in ("dataset_path", "out_dir", "in_dir", "out", "raw_dir"):
        if getattr(a, k, None):
            setattr(a, k, os.path.abspath(getattr(a, k)))
    export(a) if a.cmd == "export" else compare(a)


if __name__ == "__main__":
    main()
