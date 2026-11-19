"""EXPERIMENTAL TRIAL (may be removed if it does not work out): 3-D Gaussian splatting of a map's mapping clip.

What it does: the mapping clip's frames (all keyframes plus every ``--step``-th tracked frame) are undistorted from
the fisheye to pinhole views and, with their SLAM poses (``T_W_cam``, fixed, or refined a little with ``--pose-opt``),
used to fit a 3-D Gaussian splat scene with ``gsplat`` (L1 + SSIM, gsplat's default densification, a per-frame colour
gain/offset for the camera's auto exposure). The gripper mask (grown, undistorted) is left out of the loss, so the
fingers are not baked into the scene. The Gaussians start from :mod:`umiproc.viz.dense_3d`'s cloud
(``analysis/dense_3d.ply``, run that first) plus the map's sparse points. Every ``holdout``-th frame is held out and
scored (PSNR/SSIM over the unmasked pixels).

``python -m umiproc.viz.splat_3d FOLDER [--root DIR] [--iters N] [--images DIR]`` (not a ``umiproc`` command).
Needs a CUDA GPU, ``torch`` and ``gsplat``: they are NOT umiproc dependencies (a separate venv, see the experiment
notes in the docstring of :func:`main`); importing this module needs neither. Writes ``analysis/splat_3d.ply``
(standard 3DGS PLY: x y z, nx ny nz, f_dc_*, f_rest_*, opacity, scale_*, rot_* in W), ``analysis/splat_3d.html`` (a
viewer: @mkkellogg/gaussian-splats-3d + three.js from jsdelivr, pinned; the splats embedded as a base64 ``.splat``,
colours without the view-dependent part, gravity up and the table tag at the origin like dense_3d, with the camera
path) and ``analysis/splat_3d.json`` (parameters, timings, scores). ``--images DIR``: side-by-sides of held-out
frames (real | rendered) and a few novel views rendered by gsplat. ``--viewer-only``: rewrite just the HTML from an
existing ``splat_3d.ply``/``.json`` (no GPU needed).

The HTML opens straight from disk (file://) in a WebGL2 browser with internet (for the two CDN scripts).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Params", "load_inputs", "load_frames", "read_ply_points", "train", "evaluate", "write_3dgs_ply",
           "read_3dgs_ply", "splat_bytes", "rewrite_viewer", "build", "main"]

OUT_STEM = "splat_3d"
SH_C0 = 0.28209479177387814
GS3D_VERSION = "0.4.7"
THREE_VERSION = "0.160.0"


@dataclass
class Params:
    step: int = 3                    # every step-th tracked frame (plus all keyframes)
    width: int = 640
    height: int = 480
    fov_deg: float = 100.0           # horizontal field of view of the pinhole views
    blur_ratio: float = 0.4          # drop frames whose Laplacian variance < this x the median
    mask_grow_px: int = 20           # grow the gripper mask (full-size pixels) before it is left out of the loss
    holdout: int = 10                # every holdout-th frame is held out for scoring (0: none)
    init_points: int = 250_000       # random subset of dense_3d.ply
    iters: int = 10_000
    sh_degree: int = 3
    sh_every: int = 1000             # one more SH degree every this many iterations
    ssim_lambda: float = 0.2
    refine_start: int = 500
    refine_stop: int = 6_000
    refine_every: int = 100
    reset_every: int = 3_000
    grow_grad2d: float = 0.0002
    max_gaussians: int = 900_000     # densification stops when this many are reached (VRAM)
    appearance: bool = True          # per-frame colour gain + offset
    pose_opt: bool = False           # per-frame small pose correction
    pose_lr: float = 1e-5
    seed: int = 0

    @property
    def K(self) -> np.ndarray:
        f = (self.width / 2) / np.tan(np.radians(self.fov_deg) / 2)
        return np.array([[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]])


@dataclass
class Inputs:
    folder: str
    analysis: Path
    map_dir: Path
    traj: object
    i0: int
    video: Path
    cam: object
    export: object
    mask_full: np.ndarray | None     # (H, W) bool, True = gripper
    tag: dict | None


@dataclass
class Frames:
    fed: np.ndarray                  # (N,) fed frame index
    keyframe: np.ndarray             # (N,) bool
    T_W_cam: np.ndarray              # (N, 4, 4)
    rgb: np.ndarray                  # (N, H, W, 3) uint8, pinhole views
    valid: np.ndarray                # (H, W) bool: in the loss (not the gripper)
    test: np.ndarray                 # (N,) bool: held out
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------- inputs
# (small copies of dense_3d's loaders, so this trial does not depend on that one's internals)


def load_inputs(folder: str, root=None) -> Inputs:
    import cv2

    from umiproc.slam.trajectory_csv import read_fork_csv
    from umiproc.viz.slam_3d import read_map_export
    from umiproc.viz.slam_3d_images import read_camera
    from umiproc.viz.slam_run import count_rows, read_t0_ns, times_from_frames_csv
    from umiproc.viz.target import find_recording, resolve, ultrawide_video

    t = resolve(folder, "mapping", root=root)
    csv_path = t.run_dir / "trajectory.csv"
    video, frames_csv = ultrawide_video(find_recording(t))
    times, i0 = times_from_frames_csv(frames_csv, read_t0_ns(csv_path), count_rows(csv_path))
    mask = cv2.imread(str(t.map_dir / "slam_mask.png"), cv2.IMREAD_GRAYSCALE)
    tag_path = t.map_dir / "tx_slam_tag.json"
    return Inputs(folder=t.folder or folder, analysis=t.analysis, map_dir=t.map_dir,
                  traj=read_fork_csv(csv_path, times), i0=i0, video=video,
                  cam=read_camera(t.map_dir / "orbslam3_settings.yaml"),
                  export=read_map_export(t.run_dir / "map_points.csv"),
                  mask_full=(mask > 0) if mask is not None else None,
                  tag=json.loads(tag_path.read_text()) if tag_path.is_file() else None)


def fisheye_maps(cam, K: np.ndarray, size: tuple[int, int]):
    import cv2

    Kf = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    return cv2.fisheye.initUndistortRectifyMap(Kf, np.array(cam.k, np.float64), np.eye(3), K, size, cv2.CV_32FC1)


def load_frames(inp: Inputs, p: Params) -> Frames:
    """Keyframes (map-export pose) + every ``step``-th tracked frame (trajectory pose), undistorted; blurry ones
    dropped; every ``holdout``-th held out (keyframes are never held out)."""
    import cv2

    from umiproc.viz.slam_3d_images import read_keyframe_images

    traj, ex = inp.traj, inp.export
    sel: dict[int, tuple[np.ndarray, bool]] = {}
    for j, k in enumerate(ex.keyframe_frames):
        if 0 <= k < traj.n_frames:
            sel[int(k)] = (ex.T_W_kf[j], True)
    for k in np.flatnonzero(traj.tracked)[::p.step]:
        sel.setdefault(int(k), (traj.T_W_cam[k], False))
    feds = sorted(sel)
    imgs = read_keyframe_images(inp.video, {k: inp.i0 + k for k in feds}, 0)
    mx, my = fisheye_maps(inp.cam, p.K, (p.width, p.height))
    m = np.zeros((inp.cam.height, inp.cam.width), np.uint8) if inp.mask_full is None else \
        inp.mask_full.astype(np.uint8)
    if p.mask_grow_px > 0:
        g = p.mask_grow_px
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    valid = cv2.remap((m == 0).astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderValue=0) > 0
    rgb, keep, sharp = [], [], []
    for k in feds:
        if k not in imgs:
            continue
        v = cv2.remap(imgs[k], mx, my, cv2.INTER_LINEAR)
        sharp.append(float(cv2.Laplacian(cv2.cvtColor(v, cv2.COLOR_RGB2GRAY), cv2.CV_64F)[valid].var()))
        rgb.append(v)
        keep.append(k)
    sharp = np.array(sharp)
    ok = sharp >= p.blur_ratio * np.median(sharp)
    keep = [k for k, o in zip(keep, ok) if o]
    rgb = np.stack([r for r, o in zip(rgb, ok) if o])
    kf = np.array([sel[k][1] for k in keep])
    test = np.zeros(len(keep), bool)
    if p.holdout:
        cand = np.flatnonzero(~kf)
        test[cand[p.holdout // 2::p.holdout]] = True
    return Frames(fed=np.array(keep), keyframe=kf, T_W_cam=np.stack([sel[k][0] for k in keep]), rgb=rgb,
                  valid=valid, test=test,
                  info={"selected": len(feds), "after_blur_check": len(keep), "keyframes": int(kf.sum()),
                        "train": int((~test).sum()), "test": int(test.sum()),
                        "valid_fraction": round(float(valid.mean()), 3)})


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(X (N, 3) float64, rgb (N, 3) uint8)`` of a binary PLY with x y z (float) and red green blue (uchar)."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").splitlines()
    n = next(int(l.split()[2]) for l in head if l.startswith("element vertex"))
    tmap = {"float": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "float32": "<f4"}
    props = [(l.split()[2], tmap[l.split()[1]]) for l in head if l.startswith("property")]
    rec = np.frombuffer(raw, dtype=np.dtype(props), count=n, offset=end)
    X = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    C = np.stack([rec["red"], rec["green"], rec["blue"]], 1).astype(np.uint8)
    return X, C


def init_points(inp: Inputs, frames: Frames, p: Params) -> tuple[np.ndarray, np.ndarray, dict]:
    """A random ``init_points`` subset of dense_3d.ply plus the sparse map points (seen >= 3 times, within 3 m of the
    path; coloured grey: their colour is learnt)."""
    ply = inp.analysis / "dense_3d.ply"
    if not ply.is_file():
        raise FileNotFoundError(f"{ply} is missing: run `python -m umiproc.viz.dense_3d {inp.folder}` first")
    X, C = read_ply_points(ply)
    rng = np.random.default_rng(p.seed)
    n_dense = len(X)
    if len(X) > p.init_points:
        i = rng.choice(len(X), p.init_points, replace=False)
        X, C = X[i], C[i]
    from scipy.spatial import cKDTree

    cams = frames.T_W_cam[:, :3, 3]
    sp = inp.export.points[inp.export.observations >= 3]
    sp = sp[cKDTree(cams).query(sp)[0] < 3.0]
    # sparse points far from the dense cloud only (the background the stereo did not reach)
    sp = sp[cKDTree(X).query(sp)[0] > 0.02] if len(sp) else sp
    X = np.concatenate([X, sp])
    C = np.concatenate([C, np.full((len(sp), 3), 128, np.uint8)])
    return X, C, {"dense_ply_points": int(n_dense), "from_dense": int(len(X) - len(sp)), "from_sparse": int(len(sp))}


# ---------------------------------------------------------------------------------------------------- training


def _ssim_map(a, b, win: int = 11, sigma: float = 1.5):
    """SSIM map (1, 3, H, W) of two (1, 3, H, W) images in [0, 1] (Gaussian window, 'same' padding)."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(win, device=a.device) - win // 2) ** 2).float() / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).expand(3, 1, win, win).contiguous()

    def f(x):
        return F.conv2d(x, w, padding=win // 2, groups=3)

    mu_a, mu_b = f(a), f(b)
    s_aa, s_bb, s_ab = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))


def _so3_exp(w):
    """(N, 3) axis-angle -> (N, 3, 3) (torch)."""
    import torch

    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx = Kx - Kx.transpose(1, 2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + s * Kx + (1 - c) * (Kx @ Kx)


def make_splats(X: np.ndarray, C: np.ndarray, p: Params, scene_scale: float, device):
    """The Gaussian parameters (gsplat's simple_trainer layout) and one Adam per parameter."""
    import torch
    from scipy.spatial import cKDTree

    d, _ = cKDTree(X).query(X, 4, workers=-1)
    dist = np.sqrt(np.mean(d[:, 1:] ** 2, axis=1)).clip(1e-4, 0.05)
    n = len(X)
    rgb = C.astype(np.float32) / 255.0
    sh = np.zeros((n, (p.sh_degree + 1) ** 2, 3), np.float32)
    sh[:, 0] = (rgb - 0.5) / SH_C0
    t = lambda a: torch.nn.Parameter(torch.tensor(np.asarray(a, np.float32), device=device))  # noqa: E731
    params = torch.nn.ParameterDict({
        "means": t(X),
        "scales": t(np.log(np.repeat(dist[:, None], 3, 1))),
        "quats": t(np.tile([1.0, 0, 0, 0], (n, 1))),
        "opacities": t(np.full(n, math.log(0.1 / 0.9))),
        "sh0": t(sh[:, :1]),
        "shN": t(sh[:, 1:]),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15) for k, lr in lrs.items()}
    return params, opts


def render(params, viewmats, K, w: int, h: int, sh_degree: int):
    """``(rgb (C, H, W, 3), alpha (C, H, W, 1), info)`` with gsplat."""
    import torch
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], 1)
    return rasterization(means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                         opacities=torch.sigmoid(params["opacities"]), colors=colors, viewmats=viewmats,
                         Ks=K[None].expand(len(viewmats), 3, 3), width=w, height=h, sh_degree=sh_degree,
                         packed=False, rasterize_mode="classic")


def train(frames: Frames, X0: np.ndarray, C0: np.ndarray, p: Params, log=None) -> dict:
    """Fit the splats; returns ``{params, appearance, pose deltas, stats}`` (torch tensors on the GPU)."""
    import torch
    from gsplat.strategy import DefaultStrategy

    log = log or (lambda *a: print(*a, flush=True))
    if not torch.cuda.is_available():
        raise RuntimeError("splat_3d needs a CUDA GPU (torch.cuda.is_available() is False)")
    dev = torch.device("cuda")
    torch.manual_seed(p.seed)
    rng = np.random.default_rng(p.seed)
    cams = frames.T_W_cam[:, :3, 3]
    scene_scale = float(np.linalg.norm(cams - cams.mean(0), axis=1).max() * 1.1)
    params, opts = make_splats(X0, C0, p, scene_scale, dev)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / p.iters))
    strategy = DefaultStrategy(refine_start_iter=p.refine_start, refine_stop_iter=p.refine_stop,
                               refine_every=p.refine_every, reset_every=p.reset_every, grow_grad2d=p.grow_grad2d,
                               verbose=False)
    strategy.check_sanity(params, opts)
    state = strategy.initialize_state(scene_scale=scene_scale)

    train_idx = np.flatnonzero(~frames.test)
    N = len(frames.fed)
    imgs = torch.from_numpy(frames.rgb).to(dev)                              # uint8 (N, H, W, 3)
    valid = torch.from_numpy(frames.valid).to(dev).float()[None, None]      # (1, 1, H, W)
    n_valid = valid.sum() * 3
    K = torch.tensor(p.K, dtype=torch.float32, device=dev)
    T_W_cam = torch.tensor(frames.T_W_cam, dtype=torch.float32, device=dev)
    app = torch.nn.Parameter(torch.zeros(N, 6, device=dev))                  # gain (3), offset (3)
    app_opt = torch.optim.Adam([app], lr=1e-3)
    dpose = torch.nn.Parameter(torch.zeros(N, 6, device=dev))                # rotation (3), translation (3)
    pose_opt = torch.optim.Adam([dpose], lr=p.pose_lr, weight_decay=1e-6)

    def viewmat(i: int):
        T = T_W_cam[i:i + 1]
        if p.pose_opt:
            D = torch.eye(4, device=dev)[None].clone()
            D[:, :3, :3] = _so3_exp(dpose[i:i + 1, :3])
            D[:, :3, 3] = dpose[i:i + 1, 3:]
            T = T @ D
        return torch.linalg.inv(T)

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    hist = []
    for step in range(p.iters):
        i = int(rng.choice(train_idx))
        gt = imgs[i].permute(2, 0, 1)[None].float() / 255.0
        out, _alpha, info = render(params, viewmat(i), K, p.width, p.height, min(step // p.sh_every, p.sh_degree))
        strategy.step_pre_backward(params, opts, state, step, info)
        img = out[0].permute(2, 0, 1)[None]
        if p.appearance:
            img = img * (1 + app[i, :3, None, None]) + app[i, 3:, None, None]
        l1 = ((img - gt).abs() * valid).sum() / n_valid
        ssim = (_ssim_map(img, gt) * valid).sum() / n_valid
        loss = (1 - p.ssim_lambda) * l1 + p.ssim_lambda * (1 - ssim)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        if p.appearance:
            app_opt.step()
            app_opt.zero_grad(set_to_none=True)
        if p.pose_opt and step >= p.refine_start:
            pose_opt.step()
        pose_opt.zero_grad(set_to_none=True)
        sched.step()
        strategy.step_post_backward(params, opts, state, step, info, packed=False)
        n = len(params["means"])
        if n >= p.max_gaussians and strategy.refine_stop_iter > step:
            strategy.refine_stop_iter = step              # stop densifying (VRAM)
            log(f"  step {step}: {n} Gaussians, densification stopped")
        if step % 500 == 0 or step == p.iters - 1:
            el = time.perf_counter() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            hist.append({"step": step, "loss": round(float(loss), 4), "l1": round(float(l1), 4),
                         "n": n, "s": round(el, 1), "peak_gb": round(mem, 2)})
            log(f"  step {step:5d}  loss {float(loss):.4f}  l1 {float(l1):.4f}  gaussians {n:7d}  "
                f"{el:6.0f} s  peak {mem:.2f} GB")
    return {"params": params, "app": app.detach(), "dpose": dpose.detach(), "viewmat": viewmat, "K": K,
            "imgs": imgs, "valid": valid, "scene_scale": scene_scale,
            "stats": {"train_s": round(time.perf_counter() - t0, 1),
                      "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                      "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                      "gaussians": int(len(params["means"])), "scene_scale": round(scene_scale, 4),
                      "history": hist}}


# ---------------------------------------------------------------------------------------------------- evaluation


def _psnr(a, b, m) -> float:
    mse = float((((a - b) ** 2) * m).sum() / (m.sum() * 3))
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def evaluate(frames: Frames, res: dict, p: Params, images_dir: Path | None = None, n_images: int = 4) -> dict:
    """PSNR/SSIM over the unmasked pixels of the held-out frames: raw (no colour correction) and with a per-channel
    gain/offset fitted to that frame (the auto-exposure the training frames learnt); also the training frames' raw
    scores (on a subset). Side-by-sides of ``n_images`` held-out frames into ``images_dir``."""
    import cv2
    import torch

    params, imgs, valid, K = res["params"], res["imgs"], res["valid"], res["K"]
    m = valid[0, 0][..., None]                                               # (H, W, 1)
    out: dict = {}
    test_idx = np.flatnonzero(frames.test)
    train_idx = np.flatnonzero(~frames.test)[::5]
    shots = set(test_idx[np.linspace(0, len(test_idx) - 1, min(n_images, len(test_idx))).astype(int)].tolist()) \
        if len(test_idx) else set()
    with torch.no_grad():
        for name, idx in (("test", test_idx), ("train_subset", train_idx)):
            ps, ss, pc, sc = [], [], [], []
            for i in idx:
                vm = torch.linalg.inv(torch.tensor(frames.T_W_cam[i], dtype=torch.float32, device=K.device))[None]
                r = render(params, vm, K, p.width, p.height, p.sh_degree)[0][0].clamp(0, 1)
                gt = imgs[i].float() / 255.0
                ps.append(_psnr(r, gt, m))
                ss.append(float((_ssim_map(r.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
                                 * valid).sum() / (valid.sum() * 3)))
                # per-channel affine colour fit on the valid pixels
                sel = m[..., 0] > 0
                A = r[sel]
                B = gt[sel]
                rc = r.clone()
                for c in range(3):
                    x, y = A[:, c], B[:, c]
                    vx = x.var()
                    g = ((x - x.mean()) * (y - y.mean())).mean() / vx.clamp_min(1e-8)
                    rc[..., c] = (r[..., c] - x.mean()) * g + y.mean()
                rc = rc.clamp(0, 1)
                pc.append(_psnr(rc, gt, m))
                sc.append(float((_ssim_map(rc.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
                                 * valid).sum() / (valid.sum() * 3)))
                if images_dir is not None and name == "test" and i in shots:
                    a = (gt.cpu().numpy() * 255).astype(np.uint8)
                    b = (r.cpu().numpy() * 255).astype(np.uint8)
                    dim = (~frames.valid)[..., None]
                    a = np.where(dim, a // 3, a)
                    b2 = np.where(dim, b // 3, b)
                    pair = np.concatenate([a, np.full((p.height, 8, 3), 255, np.uint8), b2], 1)
                    cv2.putText(pair, f"real (fed {frames.fed[i]})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 0), 2)
                    cv2.putText(pair, f"rendered, PSNR {ps[-1]:.1f}", (p.width + 18, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 0), 2)
                    images_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(images_dir / f"heldout_fed{frames.fed[i]:05d}.png"), pair[..., ::-1])
            if len(idx):
                out[name] = {"frames": int(len(idx)), "psnr": round(float(np.mean(ps)), 2),
                             "ssim": round(float(np.mean(ss)), 4),
                             "psnr_colour_matched": round(float(np.mean(pc)), 2),
                             "ssim_colour_matched": round(float(np.mean(sc)), 4)}
    return out


def scene_transform(up_W, tag: dict | None) -> np.ndarray:
    """4x4 W -> scene: gravity up is +z, the tag's centre the origin (as dense_3d_viewer)."""
    from umiproc.viz.slam_3d import scene_rotation

    T_tag = np.asarray(tag["tx_slam_tag"], np.float64) if tag else None
    if up_W is None and T_tag is not None:
        up_W = tag.get("up_world") or (-T_tag[:3, 2]).tolist()
    T = np.eye(4)
    T[:3, :3] = scene_rotation(up_W)
    if T_tag is not None:
        T[:3, 3] = -T[:3, :3] @ T_tag[:3, 3]
    return T


def _look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """T_scene_cam (OpenCV camera: x right, y down, z forward) at ``eye`` looking at ``target``."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0, 1, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def novel_views(res: dict, T_scene_W: np.ndarray, frames: Frames, p: Params, images_dir: Path) -> list[str]:
    """A few views from outside the clip's path (gravity up, around the tag) and one along the path."""
    import cv2
    import torch

    params, K = res["params"], res["K"]
    cams_s = frames.T_W_cam[:, :3, 3] @ T_scene_W[:3, :3].T + T_scene_W[:3, 3]
    ctr = np.array([0.0, 0.0, 0.0])
    r = float(np.median(np.linalg.norm(cams_s - ctr, axis=1)))
    hz = float(np.median(cams_s[:, 2]))
    views = {"top": _look_at(np.array([0, -0.05, max(0.6, 1.3 * r)]), ctr),
             "front_low": _look_at(np.array([0, -1.2 * r, 0.6 * hz]), ctr),
             "orbit_45": _look_at(np.array([0.9 * r, -0.9 * r, hz]), ctr),
             "orbit_135": _look_at(np.array([0.9 * r, 0.9 * r, hz]), ctr),
             "behind_path": _look_at(cams_s.mean(0) * 1.4 + [0, 0, 0.1], ctr)}
    T_W_scene = np.linalg.inv(T_scene_W)
    f = (960 / 2) / np.tan(np.radians(70) / 2)
    Kn = torch.tensor([[f, 0, 479.5], [0, f, 359.5], [0, 0, 1]], dtype=torch.float32, device=K.device)
    paths = []
    images_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, T_s_c in views.items():
            vm = torch.tensor(np.linalg.inv(T_W_scene @ T_s_c), dtype=torch.float32, device=K.device)[None]
            img = render(params, vm, Kn, 960, 720, p.sh_degree)[0][0].clamp(0, 1).cpu().numpy()
            pth = images_dir / f"novel_{name}.png"
            cv2.imwrite(str(pth), (img[..., ::-1] * 255).astype(np.uint8))
            paths.append(str(pth))
    return paths


# ---------------------------------------------------------------------------------------------------- export


def splat_arrays(params) -> dict:
    """numpy copies: means (N, 3), scales (log), quats (wxyz, normalised), opacities (logit), sh0 (N, 1, 3),
    shN (N, K-1, 3)."""
    q = params["quats"].detach()
    q = q / q.norm(dim=-1, keepdim=True)
    return {"means": params["means"].detach().cpu().numpy(), "scales": params["scales"].detach().cpu().numpy(),
            "quats": q.cpu().numpy(), "opacities": params["opacities"].detach().cpu().numpy(),
            "sh0": params["sh0"].detach().cpu().numpy(), "shN": params["shN"].detach().cpu().numpy()}


def write_3dgs_ply(path: Path, a: dict) -> None:
    """The standard (INRIA) 3DGS PLY, in W."""
    n = len(a["means"])
    f_dc = a["sh0"].reshape(n, 3)
    f_rest = a["shN"].transpose(0, 2, 1).reshape(n, -1)                    # channel-major, as INRIA
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] +
             [f"f_rest_{i}" for i in range(f_rest.shape[1])] + ["opacity"] +
             [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([a["means"], np.zeros((n, 3)), f_dc, f_rest, a["opacities"][:, None], a["scales"],
                           a["quats"]], 1).astype("<f4")
    head = "ply\nformat binary_little_endian 1.0\ncomment umiproc splat_3d (experimental), world W of the SLAM map\n" \
           f"element vertex {n}\n" + "".join(f"property float {k}\n" for k in names) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(np.ascontiguousarray(data).tobytes())


def read_3dgs_ply(path: Path) -> dict:
    """The arrays of :func:`splat_arrays` back from a 3DGS PLY written by :func:`write_3dgs_ply`."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode("ascii").splitlines()
    n = next(int(l.split()[2]) for l in head if l.startswith("element vertex"))
    names = [l.split()[2] for l in head if l.startswith("property")]
    a = np.frombuffer(raw, "<f4", count=n * len(names), offset=end).reshape(n, len(names))
    col = {k: i for i, k in enumerate(names)}

    def take(*ks):
        return a[:, [col[k] for k in ks]].astype(np.float64)

    n_rest = sum(k.startswith("f_rest_") for k in names)
    return {"means": take("x", "y", "z"), "scales": take("scale_0", "scale_1", "scale_2"),
            "quats": take("rot_0", "rot_1", "rot_2", "rot_3"), "opacities": take("opacity")[:, 0],
            "sh0": take("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :],
            "shN": take(*[f"f_rest_{i}" for i in range(n_rest)]).reshape(n, 3, -1).transpose(0, 2, 1)}


def rewrite_viewer(folder: str, root=None, out_dir: Path | None = None) -> Path:
    """Only the HTML again, from an existing ``splat_3d.ply`` and ``splat_3d.json`` (no GPU, no torch)."""
    inp = load_inputs(folder, root)
    out_dir = Path(out_dir) if out_dir is not None else inp.analysis
    summary = json.loads((out_dir / f"{OUT_STEM}.json").read_text())
    p = Params(**{k: v for k, v in summary["params"].items() if k in Params.__dataclass_fields__})
    frames = load_frames(inp, p)
    T = scene_transform(inp.export.meta.get("up_W"), inp.tag)
    splat, _n = splat_bytes(read_3dgs_ply(out_dir / f"{OUT_STEM}.ply"), T)
    html = out_dir / f"{OUT_STEM}.html"
    write_page(html, viewer_payload(inp, frames, T, {k: summary[k] for k in
                                                     ("frames", "gaussians", "scores", "timings_s")}), splat)
    return html


def _quat_wxyz_from_R(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(R).as_quat()
    return np.concatenate([xyzw[..., 3:], xyzw[..., :3]], -1)


def splat_bytes(a: dict, T: np.ndarray, min_opacity: float = 1 / 255) -> tuple[bytes, int]:
    """The antimatter15 ``.splat`` format (32 bytes a splat: position, scale, RGBA, quaternion), turned by T (W ->
    scene), sorted by size x opacity (largest first); the colour is the SH's constant part only."""
    from scipy.spatial.transform import Rotation

    op = 1 / (1 + np.exp(-a["opacities"]))
    keep = op >= min_opacity
    X = a["means"][keep] @ T[:3, :3].T + T[:3, 3]
    s = np.exp(a["scales"][keep])
    q = a["quats"][keep]                                                     # wxyz
    Rq = Rotation.from_quat(np.concatenate([q[:, 1:], q[:, :1]], 1)).as_matrix()
    q2 = _quat_wxyz_from_R(T[:3, :3][None] @ Rq)
    rgb = np.clip(0.5 + SH_C0 * a["sh0"][keep].reshape(-1, 3), 0, 1)
    order = np.argsort(-(s.prod(1) * op[keep]))
    n = int(keep.sum())
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"] = X[order]
    rec["s"] = s[order]
    rec["c"][:, :3] = np.rint(rgb[order] * 255)
    rec["c"][:, 3] = np.rint(op[keep][order] * 255)
    rec["r"] = np.clip(np.rint(q2[order] * 128 + 128), 0, 255)
    return rec.tobytes(), n


def viewer_payload(inp: Inputs, frames: Frames, T: np.ndarray, summary: dict) -> dict:
    def apply(X):
        return np.asarray(X, np.float64).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]

    traj = inp.traj
    k = np.flatnonzero(traj.tracked)
    tag = None
    if inp.tag:
        Tt = np.asarray(inp.tag["tx_slam_tag"], np.float64)
        h = float(inp.tag.get("tag", {}).get("size_m", 0.16)) / 2
        tag = apply(np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]) @ Tt[:3, :3].T + Tt[:3, 3])
    first = T @ frames.T_W_cam[0]
    return {"title": f"{inp.folder}: Gaussian splats of the mapping clip (experimental)",
            "path": np.round(apply(traj.T_W_cam[k, :3, 3]), 4).tolist(),
            "path_t": np.round((traj.t_ns[k] - traj.t_ns[0]) / 1e9, 3).tolist(),
            "tag": np.round(tag, 4).tolist() if tag is not None else None,
            "first_cam": {"pos": np.round(first[:3, 3], 4).tolist(),
                          "look": np.round(first[:3, 3] + 0.5 * first[:3, 2], 4).tolist()},
            "summary": summary}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Map Gaussian splats</title>
<style>
:root { --bg: #f4f4f2; --panel: rgba(255,255,255,0.92); --text: #1d1d1f; --muted: #5c5c60; --line: #d6d6d2;
  --accent: #1f5fbf; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141416; --panel: rgba(28,28,32,0.9);
  --text: #ececee; --muted: #a0a0a8; --line: #3a3a40; --accent: #6ea8ff; } }
:root[data-theme="dark"] { --bg: #141416; --panel: rgba(28,28,32,0.9); --text: #ececee; --muted: #a0a0a8;
  --line: #3a3a40; --accent: #6ea8ff; }
html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; overflow: hidden; }
#view { position: fixed; inset: 0; background: #202024; }
#panel { position: fixed; top: 12px; left: 12px; width: min(300px, calc(100vw - 32px)); background: var(--panel);
  border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; box-sizing: border-box; z-index: 10;
  max-height: calc(100vh - 24px); overflow: auto; }
#panel h1 { font-size: 14px; margin: 0 0 4px; font-weight: 600; }
#panel .muted { color: var(--muted); }
#panel label { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
#panel .row { display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }
#panel button { font: inherit; padding: 3px 8px; border: 1px solid var(--line); border-radius: 5px;
  background: transparent; color: var(--text); cursor: pointer; }
#panel button:hover { border-color: var(--accent); }
#stats { font: 11px/1.45 ui-monospace, monospace; color: var(--muted); white-space: pre-wrap; margin-top: 6px; }
#err { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; padding: 16px;
  text-align: center; color: #eee; z-index: 5; }
details summary { cursor: pointer; color: var(--muted); }
</style>
<script type="importmap">{"imports": {"three": "__THREE__", "gs3d": "__GS3D__", "three/addons/": "__ADDONS__"}}</script>
</head><body>
<div id="view"></div>
<div id="panel">
 <h1 id="title"></h1>
 <div class="muted">Gravity up, the table tag at the origin; the grid has 10 cm squares. Drag to orbit, right-drag
  to pan, scroll to zoom. Colours without the view-dependent part. Experimental.</div>
 <label><input id="showPath" type="checkbox" checked> camera path (colour = time)</label>
 <label><input id="showTag" type="checkbox" checked> table tag</label>
 <label><input id="showGrid" type="checkbox"> 10 cm grid</label>
 <div class="row"><button data-v="persp">3/4 view</button><button data-v="top">top</button>
  <button data-v="front">front</button><button data-v="side">side</button><button data-v="cam">first camera</button></div>
 <details><summary>numbers</summary><div id="stats"></div></details>
</div>
<div id="err">The viewer needs WebGL2 and internet once (three.js and gaussian-splats-3d from cdn.jsdelivr.net).</div>
<script>window.D = __DATA__; window.SPLAT = "__SPLAT__";</script>
<script type="module">
import * as THREE from "three";
import * as GS from "gs3d";
const D = window.D, $ = id => document.getElementById(id);
window.LOADED = true;
$("title").textContent = D.title;
$("stats").textContent = JSON.stringify(D.summary, null, 1);
const extra = new THREE.Scene();
function ramp(t) {
  const s = [[0.27, 0.0, 0.33], [0.23, 0.32, 0.55], [0.13, 0.57, 0.55], [0.37, 0.79, 0.38], [0.99, 0.91, 0.14]];
  const x = Math.min(0.9999, Math.max(0, t)) * (s.length - 1), i = Math.floor(x), f = x - i;
  return s[i].map((v, k) => v + f * (s[i + 1][k] - v));
}
function lineObj(pts, colours, color) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pts.flat(), 3));
  if (colours) g.setAttribute("color", new THREE.Float32BufferAttribute(colours.flat(), 3));
  return new THREE.Line(g, new THREE.LineBasicMaterial(colours ? {vertexColors: true} : {color}));
}
const T1 = D.path_t[D.path_t.length - 1] || 1;
const path = lineObj(D.path, D.path_t.map(t => ramp(t / T1)));
extra.add(path);
const tagGroup = new THREE.Group();
if (D.tag) tagGroup.add(lineObj([...D.tag, D.tag[0]].map(p => [p[0], p[1], p[2] + 0.002]), null, 0xff2d55));
extra.add(tagGroup);
const grid = new THREE.GridHelper(2, 20, 0x888888, 0x888888);
grid.rotation.x = Math.PI / 2; grid.visible = false; extra.add(grid);
for (const [id, o] of [["showPath", path], ["showTag", tagGroup], ["showGrid", grid]])
  $(id).onchange = () => { o.visible = $(id).checked; };

const hash = new URLSearchParams(location.hash.slice(1));
const views = {persp: [[0.6, -0.75, 0.55], [0, 0.1, 0]], top: [[0, 0.0, 1.0], [0, 0.1, 0]],
  front: [[0, -0.9, 0.35], [0, 0.1, 0]], side: [[0.9, 0.1, 0.35], [0, 0.1, 0]],
  cam: [D.first_cam.pos, D.first_cam.look]};
const v0 = views[hash.get("view") || "persp"] || views.persp;
const viewer = new GS.Viewer({rootElement: $("view"), cameraUp: [0, 0, 1], initialCameraPosition: v0[0],
  initialCameraLookAt: v0[1], threeScene: extra, sharedMemoryForWorkers: false, gpuAcceleratedSort: false,
  sphericalHarmonicsDegree: 0, sceneRevealMode: GS.SceneRevealMode.Instant});
function setView(name) {
  const v = views[name]; viewer.camera.position.set(...v[0]); viewer.controls.target.set(...v[1]);
  viewer.camera.up.set(0, 0, 1); viewer.controls.update();
}
window.setView = setView;
document.querySelectorAll("button[data-v]").forEach(b => b.onclick = () => setView(b.dataset.v));
const bin = Uint8Array.from(atob(window.SPLAT), c => c.charCodeAt(0));
const url = URL.createObjectURL(new Blob([bin], {type: "application/octet-stream"}));
viewer.addSplatScene(url, {format: GS.SceneFormat.Splat, showLoadingUI: true, splatAlphaRemovalThreshold: 2})
  .then(() => { viewer.start(); window.READY = true; })
  .catch(e => { $("err").textContent = "Could not load the splats: " + e; $("err").style.display = "flex"; });
</script>
<script>setTimeout(() => { if (!window.LOADED) document.getElementById("err").style.display = "flex"; }, 20000);</script>
</body></html>
"""


def write_page(path: Path, payload: dict, splat: bytes) -> None:
    html = (PAGE.replace("__THREE__", f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js")
            .replace("__ADDONS__", f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/examples/jsm/")
            .replace("__GS3D__", f"https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@{GS3D_VERSION}"
                                 "/build/gaussian-splats-3d.module.js")
            .replace("__DATA__", json.dumps(payload, separators=(",", ":")))
            .replace("__SPLAT__", base64.b64encode(splat).decode("ascii")))
    Path(path).write_text(html)


# ---------------------------------------------------------------------------------------------------- driver


def build(folder: str, root=None, p: Params | None = None, out_dir: Path | None = None,
          images_dir: Path | None = None) -> dict:
    import torch

    p = p or Params()
    laps: dict[str, float] = {}
    t = time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        laps[name] = round(now - t, 1)
        t = now
        print(f"  {name}: {laps[name]:.1f} s", flush=True)

    inp = load_inputs(folder, root)
    frames = load_frames(inp, p)
    lap("load + undistort")
    X0, C0, init_info = init_points(inp, frames, p)
    lap("init points")
    res = train(frames, X0, C0, p)
    lap("train")
    scores = evaluate(frames, res, p, images_dir)
    T = scene_transform(inp.export.meta.get("up_W"), inp.tag)
    novel = novel_views(res, T, frames, p, images_dir) if images_dir is not None else []
    lap("evaluate")
    out_dir = Path(out_dir) if out_dir is not None else inp.analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    ply, html, js = (out_dir / f"{OUT_STEM}{ext}" for ext in (".ply", ".html", ".json"))
    a = splat_arrays(res["params"])
    write_3dgs_ply(ply, a)
    splat, n_splat = splat_bytes(a, T)
    stats = res["stats"]
    summary = {
        "format": "umiproc.splat_3d.v0 (experimental)", "folder": inp.folder, "map": inp.map_dir.name,
        "world": "the PLY is in W (the SLAM map's world); the viewer's .splat is turned gravity-up, tag at 0",
        "params": asdict(p), "K_pinhole": np.round(p.K, 3).tolist(), "frames": frames.info, "init": init_info,
        "gaussians": stats["gaussians"], "gaussians_in_viewer": n_splat, "scores": scores,
        "vram_peak_allocated_gb": stats["peak_vram_gb"], "vram_peak_reserved_gb": stats["peak_vram_reserved_gb"],
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "gsplat": __import__("gsplat").__version__, "scene_scale": stats["scene_scale"],
        "history": stats["history"],
        "appearance_gain_range": np.round([float(res["app"][:, :3].min()), float(res["app"][:, :3].max())],
                                          3).tolist(),
        "pose_delta_max": {"rot_deg": round(float(np.degrees(res["dpose"][:, :3].norm(dim=1).max().cpu())), 3),
                           "trans_mm": round(float(res["dpose"][:, 3:].norm(dim=1).max().cpu()) * 1e3, 2)}
        if p.pose_opt else None,
        "timings_s": laps, "novel_views": novel,
    }
    write_page(html, viewer_payload(inp, frames, T, {k: summary[k] for k in
                                                     ("frames", "gaussians", "scores", "timings_s")}), splat)
    lap("write")
    summary["timings_s"] = laps
    js.write_text(json.dumps(summary, indent=1) + "\n")
    summary["outputs"] = [str(ply), str(html), str(js)]
    return summary


def main(argv=None) -> int:
    """Experiment notes: run in a separate venv (never the umiproc one), e.g. ``python3.10 -m venv ~/envs/env_splat;
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121; pip install
    "gsplat==1.5.3+pt24cu121" -f https://docs.gsplat.studio/whl/gsplat/; pip install -e umi-processing``."""
    ap = argparse.ArgumentParser(prog="python -m umiproc.viz.splat_3d", description=__doc__.splitlines()[0])
    ap.add_argument("folder", metavar="FOLDER", help="a task folder's name (under <root>/tasks/)")
    ap.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    ap.add_argument("--out-dir", default=None, metavar="DIR", help="write here instead of <folder>/analysis/")
    ap.add_argument("--images", default=None, metavar="DIR", help="also write held-out side-by-sides + novel views")
    ap.add_argument("--iters", type=int, default=Params.iters)
    ap.add_argument("--step", type=int, default=Params.step, help="every N-th tracked frame (plus all keyframes)")
    ap.add_argument("--init-points", type=int, default=Params.init_points)
    ap.add_argument("--max-gaussians", type=int, default=Params.max_gaussians)
    ap.add_argument("--fov", type=float, default=Params.fov_deg)
    ap.add_argument("--size", default=f"{Params.width}x{Params.height}", help="pinhole view WxH")
    ap.add_argument("--sh-degree", type=int, default=Params.sh_degree, help="view-dependent colour degree (0-3)")
    ap.add_argument("--pose-opt", action="store_true", help="refine the training frames' poses a little")
    ap.add_argument("--no-appearance", action="store_true", help="no per-frame colour gain/offset")
    ap.add_argument("--viewer-only", action="store_true",
                    help="only rewrite splat_3d.html from an existing splat_3d.ply/.json (no training, no GPU)")
    args = ap.parse_args(argv)
    if args.viewer_only:
        print(f"wrote {rewrite_viewer(args.folder, args.root, args.out_dir)}")
        return 0
    w, h = (int(v) for v in args.size.lower().split("x"))
    p = Params(step=args.step, iters=args.iters, init_points=args.init_points, max_gaussians=args.max_gaussians,
               fov_deg=args.fov, width=w, height=h, pose_opt=args.pose_opt, sh_degree=args.sh_degree, appearance=not args.no_appearance,
               refine_stop=min(Params.refine_stop, int(0.6 * args.iters)))
    from umiproc.viz.slam_3d import Slam3dError
    from umiproc.viz.slam_run import PlotRunError

    try:
        s = build(args.folder, args.root, p, args.out_dir, Path(args.images) if args.images else None)
    except (PlotRunError, Slam3dError, OSError, ValueError, RuntimeError) as e:
        print(f"splat_3d: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: s[k] for k in ("frames", "gaussians", "scores", "vram_peak_allocated_gb")}, indent=1))
    for o in s["outputs"]:
        print(f"wrote {o} ({Path(o).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
