"""Camera study B, step "reference": the real training inputs of the layout's 15 (episode, row) references, exactly as
the policy saw them in training, and the same frames through the deploy path (Pi 640x480 stream -> the copy's
ImagePrep), compared with the package images.

Run from ``~/Capstone-main-sim`` (``unset PYTHONPATH``), two stages in two envs (zarr + JPEG-XL codecs live in the torch
env, PyAV in the other one), or both at once with ``all``::

    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.reference package   # stage 1 (CPU, no GPU)
    ~/envs/sim_policy_deploy/bin/python       -m sim_deploy.camera_study.reference raw       # stage 2 (PyAV decode)
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.reference all       # 1, then 2 as a subprocess

Inputs (read only): ``sim_evals/_camera_study/B/layout.json`` (package + references), the package zarr, its
``package.json`` (slot per episode), ``~/data/umi/tasks/<folder>/<slot>/{raw,processed}`` (the Pi recording and
umiproc's grid / piece / episode.zarr), ``~/data/umi/rigs/rig_pi_v3{,.lens}.yaml`` (training's rig) and the copy's
``deployment/contracts/rigs/rig_pi_v3_640.yaml`` (the deploy rig). Output: ``sim_evals/_camera_study/B/reference/``.

Stage 1 (``package``), per reference ``ep<ii>_<label>`` (current_data_idx = episode start + row T):

* replicates behavior_prompting's ``SequenceSampler.sample_sequence_without_prompting`` (the receding obs, used by both
  the plain and the prompting runs; task start == episode start in this package, checked) on the real
  ``ReplayBuffer``: main = ``camera_right_main_rgb`` rows T-3, T; ultrawide = the two newest 10 Hz frames through
  ``map_upsample_index``; low-dim rows T-3, T; every key padded with its first value;
* ``UmiTaskDataset.__getitem__``'s conversion with ``obs_pose_repr: relative``, with the dataset's own functions
  (``pose_to_mat``, ``convert_pose_mat_rep``, ``mat_to_pose10d``, float32 as in training): eef_pos / eef_rot6d relative
  to the pose at T, eef_rot6d_wrt_start relative to the episode's first pose WITHOUT the training-time noise (the
  dataset adds N(0, 0.05) to the start pose's 6 numbers, a "HACK"; that key is ``ignore_by_policy: true``), width
  absolute;
* cross-checks the same rows through the copy's deploy ``PoseConverter.to_observation`` and runs the copy's
  ``Observation(...).check_values()``; checks index 14 against P3's ``ep152_ref.npz``;
* writes ``obs.npz`` (exactly the Observation keys), ``main_0/1.png``, ``uw_0/1.png`` and ``meta.json``.

Stage 2 (``raw``): rebuilds umiproc's row -> frame map (``piece.json`` frame span on ``grid.npy``, the validate kept
span, mid-exposure camera times, nearest main frame, the 10 Hz ultrawide sub-stream; checked against the episode's
stored ``uw_upsample_local``), decodes the frames with PyAV (rgb24, as umiproc), and makes

* the native path (training's rig with the lens folded in, 1600x1200 / 1280x960): should reproduce the zarr images
  up to the JPEG-XL storage loss (the mapping check);
* the deploy path: INTER_AREA to 640x480 (what the Pi streams, PI_PAYLOAD.md; the copy's pi_recording.py replays
  recordings the same way), then the copy's ImagePrep with rig_pi_v3_640.yaml. Saved: ``raw640_*.png``,
  ``deploy_*.png``, ``deploy_obs.npz``; the differences vs the zarr images (mean abs diff per channel, PSNR, best
  shift) go into ``meta.json`` (also with a JPEG q90 round trip of the 640 frames, as the sim Pi sends them);
* ``raw_extra/``: 10 more 640x480 (main, ultrawide) pairs at the start (row 6) of episodes spread over the eraser
  positions (grasp TCP xy), and ``index.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAYOUT = REPO / "sim_evals/_camera_study/B/layout.json"
OUT = REPO / "sim_evals/_camera_study/B/reference"
P3_REF = REPO / "sim_evals/_start_pose_sweep/p3/ep152_ref.npz"
TASKS = Path("~/data/umi/tasks").expanduser()
TRAIN_RIG = Path("~/data/umi/rigs/rig_pi_v3.yaml").expanduser()
DEPLOY_RIG = REPO / "deployment/contracts/rigs/rig_pi_v3_640.yaml"
ENV_TORCH = Path("~/envs/sim_policy_deploy_torch/bin/python").expanduser()
ENV_AV = Path("~/envs/sim_policy_deploy/bin/python").expanduser()

MAIN_KEY, UW_KEY = "camera_right_main_rgb", "camera_right_ultrawide_rgb"
POS_KEY, ROT_KEY, W_KEY = "gripper_right_eef_pos", "gripper_right_eef_rot_axis_angle", "gripper_right_gripper_width"
HORIZON = 2                 # img_obs_horizon = low_dim_obs_horizon (task/umi.yaml)
DOWN_MAIN = 3               # obs_down_sample_steps (60 Hz rows -> 20 Hz)
DOWN_UW = 1                 # ultrawide_down_sample_steps (in 10 Hz frames)
STREAM_WH = (640, 480)      # PI_PAYLOAD.md 0.1: both cameras 640x480 at deploy
MAIN_MAX_DT_NS = 25_000_000  # umiproc validate: a main frame within 25 ms
UW_RATE_HZ = 10.0
N_EXTRA, EXTRA_ROW = 10, 6


def _ensure_deploy_path() -> None:
    p = str(REPO / "deployment" / "policy_deploy")
    if p not in sys.path:
        sys.path.insert(0, p)


def _layout() -> dict:
    return json.loads(LAYOUT.read_text())


def _refs(layout: dict) -> list[dict]:
    out = []
    for e in layout["selection"]:
        for row, label in zip(e["rows_in_episode"], e["labels"]):
            out.append({"index": int(e["index"]), "row": int(row), "label": label,
                        "folder": f"ep{int(e['index']):02d}_{label}"})
    return out


def _png(path: Path, rgb: np.ndarray) -> None:
    import cv2
    if not cv2.imwrite(str(path), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write {path}")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=1))


def _package_episodes(dataset: Path) -> list[dict]:
    return json.loads((dataset.parent / "package.json").read_text())["episodes"]


# ------------------------------------------------------------------------------------------- stage 1: the package

def _sample_obs(rb, idx: int, row: int) -> dict:
    """behavior_prompting SequenceSampler.sample_sequence_without_prompting, the receding obs keys only."""
    sl = rb.get_episode_slice(idx)
    start = int(sl.start)
    cur = start + int(row)
    if not start <= cur < int(sl.stop):
        raise IndexError(f"row {row} outside episode {idx}")
    res, used = {}, {}
    # main: not upsampled
    arr = rb[MAIN_KEY]
    nv = min(HORIZON, (cur - start) // DOWN_MAIN + 1)
    s0 = cur - (nv - 1) * DOWN_MAIN
    rows = list(range(s0, cur + 1, DOWN_MAIN))
    out = arr[s0: cur + 1: DOWN_MAIN]
    assert out.shape[0] == nv
    used["main_rows"] = [rows[0]] * (HORIZON - nv) + rows
    res["main"] = _pad(out)
    # ultrawide: upsampled key, horizon on its own 10 Hz index
    assert rb.is_key_upsampled(UW_KEY)
    arr = rb[UW_KEY]
    u_cur = int(rb.map_upsample_index(UW_KEY, cur))
    u_start = int(rb.map_upsample_index(UW_KEY, start))
    nv = min(HORIZON, (u_cur - u_start) // DOWN_UW + 1)
    s0 = u_cur - (nv - 1) * DOWN_UW
    uidx = list(range(s0, u_cur + 1, DOWN_UW))
    out = arr[s0: u_cur + 1: DOWN_UW]
    assert out.shape[0] == nv
    used["uw_global"] = [uidx[0]] * (HORIZON - nv) + uidx
    used["uw_episode_start"] = u_start
    res["uw"] = _pad(out)
    # low-dim
    li = [cur - k * DOWN_MAIN for k in range(HORIZON)][::-1]
    li = [x for x in li if x >= start]
    used["lowdim_rows"] = [li[0]] * (HORIZON - len(li)) + li
    for key in (POS_KEY, ROT_KEY, W_KEY):
        res[key] = _pad(rb[key][li])
    used.update(start=start, cur=cur, stop=int(sl.stop))
    return res, used


def _pad(out: np.ndarray) -> np.ndarray:
    if out.shape[0] < HORIZON:
        out = np.concatenate([np.repeat(out[:1], HORIZON - out.shape[0], axis=0), out], axis=0)
    return out


def _convert_lowdim(rb, s: dict, start: int) -> dict:
    """UmiTaskDataset.__getitem__ (receding part): obs_pose_repr relative, wrt_start without the training noise."""
    from behavior_prompting.common.pose_util import mat_to_pose10d, pose_to_mat
    from behavior_prompting.train_network.common.pose_repr_util import convert_pose_mat_rep
    pos, rot = s[POS_KEY].astype(np.float32), s[ROT_KEY].astype(np.float32)
    pose_mat = pose_to_mat(np.concatenate([pos, rot], axis=-1))
    start_pose = np.concatenate([rb[POS_KEY][start], rb[ROT_KEY][start]])
    rel_start = mat_to_pose10d(convert_pose_mat_rep(pose_mat, base_pose_mat=pose_to_mat(start_pose),
                                                    pose_rep="relative", backward=False))
    obs_pose = mat_to_pose10d(convert_pose_mat_rep(pose_mat, base_pose_mat=pose_mat[-1], pose_rep="relative",
                                                   backward=False))
    return {"eef_pos": obs_pose[:, :3].astype(np.float32), "eef_rot6d": obs_pose[:, 3:].astype(np.float32),
            "eef_rot6d_wrt_start": rel_start[:, 3:].astype(np.float32),
            "gripper_width_m": s[W_KEY].astype(np.float32).reshape(HORIZON, 1)}


def _deploy_lowdim(ep: dict, lowdim_rows_in_ep: list[int], images: dict):
    """The copy's manager PoseConverter on the same absolute poses (float64, as the deploy builds them)."""
    _ensure_deploy_path()
    from policy_deploy.manager.pose_converter import PoseConverter
    poses = ep["poses"][lowdim_rows_in_ep]
    widths = ep["widths"][lowdim_rows_in_ep]
    frames = SimpleNamespace(main=images["main"], ultrawide=images["ultrawide"])
    obs, _ = PoseConverter(exec_h=1, lookahead_ms=0).to_observation(poses, widths, ep["poses"][0], frames, 0, 0,
                                                                     "measured")
    return obs


def stage_package() -> int:
    import zarr
    from behavior_prompting.common.imagecodecs_numcodecs import register_codecs
    from behavior_prompting.common.replay_buffer import ReplayBuffer
    register_codecs(verbose=False)
    _ensure_deploy_path()
    from policy_deploy.contracts.c4_policy import Observation
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode

    layout = _layout()
    dataset = Path(layout["dataset"]).expanduser()
    pkg_eps = _package_episodes(dataset)
    store = zarr.ZipStore(str(dataset), mode="r")
    root = zarr.open(store, mode="r")
    rb = ReplayBuffer(root=root)
    names = [str(n) for n in rb.episode_names[:]]
    t_start = rb.task_data_ends[:] - rb.task_lengths[:]
    ep_start = np.r_[0, rb.episode_ends[:-1]]
    assert np.array_equal(t_start, ep_start), "task start != episode start: task sampling would differ"
    ref_p3 = np.load(P3_REF)
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for ref in _refs(layout):
        i, row = ref["index"], ref["row"]
        d = OUT / ref["folder"]
        d.mkdir(parents=True, exist_ok=True)
        s, used = _sample_obs(rb, i, row)
        start = used["start"]
        low = _convert_lowdim(rb, s, start)
        obs_d = dict(low, main=s["main"].astype(np.uint8), ultrawide=s["uw"].astype(np.uint8))
        obs = Observation(**obs_d)
        obs.check_values()
        np.savez(d / "obs.npz", **obs_d)
        for k in range(HORIZON):
            _png(d / f"main_{k}.png", obs_d["main"][k])
            _png(d / f"uw_{k}.png", obs_d["ultrawide"][k])
        # deploy-side low-dim on the same rows
        ep = load_episode(dataset, i)
        assert ep["rows"][0] == start and ep["name"] in ("", names[i])
        lrows = [r - start for r in used["lowdim_rows"]]
        dob = _deploy_lowdim(ep, lrows, {"main": obs_d["main"], "ultrawide": obs_d["ultrawide"]})
        dob.check_values()
        dep_diff = {k: float(np.max(np.abs(getattr(dob, k).astype(np.float64) - low[k].astype(np.float64))))
                    for k in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m")}
        # training-time wrt_start noise: how far the HACK moves it (one draw per sample; ignore_by_policy anyway)
        g = grasp_rows(ep["widths"])
        T = row
        pose_rows = {f"T-6": max(T - 6, 0), "T-3": max(T - 3, 0), "T": T}
        uw_local = [u - used["uw_episode_start"] for u in used["uw_global"]]
        down = rb.meta[f"downsample_index_{UW_KEY}"][:]
        p3 = None
        u0 = used["uw_episode_start"]
        if str(ref_p3["name"]) == names[i]:
            # P3's ep152_ref.npz: rows = the first 60 Hz row of each 10 Hz IMX219 frame (downsample_index), main at
            # those rows, uw = the episode's IMX219 frames in order, width_m at those rows
            assert np.array_equal(ref_p3["rows"], down[u0:u0 + len(ref_p3["rows"])]), "P3 rows are not downsample_index"
            k_main = int(np.flatnonzero(ref_p3["rows"] == start + T)[0]) if (start + T) in ref_p3["rows"] else None
            p3 = {"uw_maxabs": [int(np.abs(obs_d["ultrawide"][k].astype(int) - ref_p3["uw"][uw_local[k]].astype(int)).max())
                                for k in range(HORIZON)],
                  "main_T_maxabs": (int(np.abs(obs_d["main"][1].astype(int) - ref_p3["main"][k_main].astype(int)).max())
                                    if k_main is not None else None),
                  "width_T_absdiff": (float(abs(low["gripper_width_m"][1, 0] - ref_p3["width_m"][k_main]))
                                      if k_main is not None else None)}
        meta = {
            "reference": ref["folder"], "label": ref["label"],
            "package": layout["package"], "dataset": str(dataset), "package_index": i,
            "pi_name": names[i], "episode_name": names[i].split("_", 1)[1],
            "slot": pkg_eps[i]["slot"], "folder": pkg_eps[i]["folder"],
            "row_in_episode": T, "data_row": used["cur"], "episode_start_row": start,
            "n_rows": used["stop"] - start,
            "main_rows_in_episode": [r - start for r in used["main_rows"]],
            "lowdim_rows_in_episode": lrows,
            "uw_frames_local": uw_local, "uw_frames_global": used["uw_global"],
            "uw_frames_first_row_in_episode": [int(down[u]) - start for u in used["uw_global"]],
            "widths_m": low["gripper_width_m"][:, 0],
            "tcp_tag": {k: {"row": r, "T": ep["poses"][r], "width_m": ep["widths"][r]} for k, r in pose_rows.items()},
            "grasp_rows": {k: v for k, v in g.items()},
            "tcp_tag_grasp": {k: {"row": int(g[k]), "T": ep["poses"][int(g[k])], "width_m": ep["widths"][int(g[k])]}
                              for k in ("start", "half", "end")},
            "lowdim_convention": {
                "train": "UmiTaskDataset relative: inv(P_T) @ P_{T-3}, rot6d = first two ROWS; wrt_start inv(P_0) @ P "
                         "(training adds N(0,0.05) to P_0's pos+rotvec; ignore_by_policy: true); width absolute",
                "deploy_pose_converter_maxabs": dep_diff, "check_values": "ok (train and deploy)"},
            "p3_ep152_ref_check": p3,
        }
        _write_json(d / "meta.json", meta)
        report.append((ref["folder"], names[i], T, used["main_rows"], used["uw_global"], dep_diff, p3))
        print(f"{ref['folder']:<22} {names[i]} T {T:3d} row {used['cur']} main rows {meta['main_rows_in_episode']} "
              f"uw local {uw_local}  deploy-lowdim maxabs " + " ".join(f"{k} {v:.1e}" for k, v in dep_diff.items())
              + (f"  P3 {p3}" if p3 else ""))
    store.close()
    return 0


# ------------------------------------------------------------------------------------------- stage 2: the raw path

def _train_rig():
    """Training's rig (~/data/umi/rigs/rig_pi_v3.yaml + its lens file, folded as umiproc does), native sizes."""
    import yaml
    _ensure_deploy_path()
    from policy_deploy.frames._dataproc.config import load_rig
    from policy_deploy.frames.deploy_rig import fold_lens, lens_path
    rig = yaml.safe_load(TRAIN_RIG.read_text())
    lf = lens_path(TRAIN_RIG)
    if lf.is_file():
        rig = fold_lens(rig, yaml.safe_load(lf.read_text()), lf)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / TRAIN_RIG.name
        p.write_text(yaml.safe_dump(rig, sort_keys=False))
        return load_rig(p)


def _nearest(t_grid, t_cam):
    """umiproc dp/export/frames.py nearest_frame_index (ties -> earlier)."""
    hi = np.clip(np.searchsorted(t_cam, t_grid, side="left"), 0, len(t_cam) - 1)
    lo = np.clip(hi - 1, 0, len(t_cam) - 1)
    idx = np.where(np.abs(t_cam[lo] - t_grid) <= np.abs(t_cam[hi] - t_grid), lo, hi).astype(np.int64)
    return idx


def _uw_substream(t_grid, t_uw, rate_hz=UW_RATE_HZ):
    """umiproc dp/export/frames.py ultrawide_substream -> (sel, upsample_local)."""
    period = 1e9 / float(rate_hz)
    n = int(np.floor((t_grid[-1] - t_grid[0]) / period)) + 1
    sched = t_grid[0] + np.round(np.arange(n) * period).astype(np.int64)
    sel = np.unique(_nearest(sched, t_uw))
    up = np.clip(np.searchsorted(t_uw[sel], t_grid, side="right") - 1, 0, len(sel) - 1).astype(np.int64)
    return sel[: int(up[-1]) + 1].astype(np.int64), up


class _Episode:
    """One umiproc slot: the raw recording + the processed grid -> per package row, the raw frame indices."""

    def __init__(self, folder: str, slot: str, pi_name: str, n_rows: int, rig):
        import csv
        self.dir = TASKS / folder / slot
        raw = self.dir / "raw"
        self.meta = json.loads((raw / "meta.json").read_text())
        self.cam = {c["role"]: name for name, c in self.meta["cameras"].items()}
        self.sensor = {c["role"]: c["sensor"] for c in self.meta["cameras"].values()}
        self.res = {c["role"]: tuple(c["resolution"]) for c in self.meta["cameras"].values()}
        piece = json.loads((self.dir / "processed" / "piece.json").read_text())
        assert piece["pi_name"] == pi_name and piece["pieces"] == 1, (piece["pi_name"], pi_name)
        grid = np.load(self.dir / "processed" / "grid.npy").astype(np.int64)
        rows = grid[(grid >= piece["t_first_ns"]) & (grid <= piece["t_last_ns"])]
        tr = piece["trimmed"]
        i0, i1 = int(tr["start_rows"]), int(tr["of_rows"]) - int(tr["end_rows"])
        assert len(rows) == int(tr["of_rows"]) and i1 - i0 == n_rows, (len(rows), tr, n_rows)
        self.t_grid = rows[i0:i1]
        t = {}
        for role in ("main", "ultrawide"):
            with open(raw / f"{self.cam[role]}_frames.csv") as f:
                r = list(csv.DictReader(f))
            t[role] = np.asarray(rig.camera_time_ns(role, np.array([int(x["t_ns"]) for x in r], np.int64),
                                                    np.array([int(x["exposure_ns"]) for x in r], np.int64)), np.int64)
        self.t_cam = t
        self.main_idx = _nearest(self.t_grid, t["main"])
        self.main_dt_ms = (t["main"][self.main_idx] - self.t_grid) / 1e6
        self.uw_sel, self.uw_up = _uw_substream(self.t_grid, t["ultrawide"])
        from sim_deploy.scene.zarr_lowdim import ZarrLowDim
        stored = ZarrLowDim(self.dir / "processed" / "episode.zarr").array("uw_upsample_local")
        self.up_matches_stored = bool(np.array_equal(stored, self.uw_up))
        self.n_uw = len(self.uw_sel)

    def video(self, role: str) -> Path:
        return self.dir / "raw" / f"{self.cam[role]}.mp4"


def _decode(path: Path, indices) -> dict[int, np.ndarray]:
    """PyAV rgb24 frames ``indices`` (umiproc RawEpisode.read_video_frames)."""
    import av
    want = sorted(set(int(i) for i in indices))
    out = {}
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(st)):
            if i in want:
                out[i] = fr.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
    if len(out) != len(want):
        raise RuntimeError(f"{path}: missing frames {sorted(set(want) - set(out))}")
    return out


def _to640(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.resize(rgb, STREAM_WH, interpolation=cv2.INTER_AREA)


def _jpeg(rgb: np.ndarray, q: int = 90) -> np.ndarray:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _compare(a: np.ndarray, ref: np.ndarray, max_shift: int = 4) -> dict:
    """a vs ref (224x224x3 u8): mean abs diff per channel, PSNR, max abs; best integer shift of a (min MAE over the
    overlap, |dx|,|dy| <= max_shift) and the sub-pixel phase-correlation shift (a relative to ref)."""
    import cv2
    a64, r64 = a.astype(np.float64), ref.astype(np.float64)
    d = np.abs(a64 - r64)
    mse = float(np.mean((a64 - r64) ** 2))
    best = (0, 0, float(d.mean()))
    m = max_shift
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            sa = a64[m + dy: 224 - m + dy, m + dx: 224 - m + dx]
            sr = r64[m: 224 - m, m: 224 - m]
            v = float(np.abs(sa - sr).mean())
            if v < best[2] - 1e-12:
                best = (dx, dy, v)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gr = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY).astype(np.float64)
    win = cv2.createHanningWindow((224, 224), cv2.CV_64F)
    (sx, sy), resp = cv2.phaseCorrelate(gr, ga, win)
    (bx, by), _ = cv2.phaseCorrelate(gr, gr, win)      # this OpenCV reports (0.5, 0.5) for identical images: subtract
    sx, sy = sx - bx, sy - by
    return {"mad_rgb": [round(float(x), 3) for x in d.reshape(-1, 3).mean(0)], "mad": round(float(d.mean()), 3),
            "max_abs": int(d.max()), "psnr_db": round(float(10 * np.log10(255.0 ** 2 / mse)) if mse > 0 else 99.0, 2),
            "best_int_shift_dxdy": [best[0], best[1]], "best_int_shift_mad": round(best[2], 3),
            "phasecorr_shift_dxdy": [round(float(sx), 3), round(float(sy), 3)], "phasecorr_response": round(float(resp), 3)}


def stage_raw() -> int:
    import cv2
    cv2.setNumThreads(4)
    _ensure_deploy_path()
    from policy_deploy.frames._dataproc.imageprep import ImagePrep
    from policy_deploy.frames.rig_file import load_live_rig
    from sim_deploy.scene.placement import grasp_rows
    from sim_deploy.scene.zarr_lowdim import load_episode, n_episodes

    layout = _layout()
    dataset = Path(layout["dataset"]).expanduser()
    pkg_eps = _package_episodes(dataset)
    train_rig = _train_rig()
    deploy_rig = load_live_rig(DEPLOY_RIG)
    prep_train, prep_dep = ImagePrep(train_rig), ImagePrep(deploy_rig)
    assert tuple(deploy_rig.cameras["main"].resolution) == STREAM_WH == tuple(deploy_rig.cameras["ultrawide"].resolution)
    eps: dict[int, _Episode] = {}

    def episode(i: int) -> _Episode:
        if i not in eps:
            pe = pkg_eps[i]
            eps[i] = _Episode(pe["folder"], pe["slot"], pe["pi_name"], int(pe["n_steps"]), train_rig)
        return eps[i]

    index, summary = [], []
    for ref in _refs(layout):
        d = OUT / ref["folder"]
        meta = json.loads((d / "meta.json").read_text())
        obs = dict(np.load(d / "obs.npz"))
        i = ref["index"]
        e = episode(i)
        assert e.res["main"] == (1600, 1200) and e.res["ultrawide"] == (1280, 960)
        main_f = [int(e.main_idx[r]) for r in meta["main_rows_in_episode"]]
        uw_f = [int(e.uw_sel[u]) for u in meta["uw_frames_local"]]
        fm, fu = _decode(e.video("main"), main_f), _decode(e.video("ultrawide"), uw_f)
        raw = {"main": [fm[f] for f in main_f], "ultrawide": [fu[f] for f in uw_f]}
        for role in ("main", "ultrawide"):
            assert raw[role][0].shape[:2] == e.res[role][::-1], (role, raw[role][0].shape)
        prepf = {"main": "prep_main", "ultrawide": "prep_ultrawide"}
        native = {r: np.stack([getattr(prep_train, prepf[r])(x) for x in raw[r]]) for r in raw}
        r640 = {r: [_to640(x) for x in raw[r]] for r in raw}
        dep = {r: np.stack([getattr(prep_dep, prepf[r])(x) for x in r640[r]]) for r in raw}
        depj = {r: np.stack([getattr(prep_dep, prepf[r])(_jpeg(x)) for x in r640[r]]) for r in raw}
        short = {"main": "main", "ultrawide": "uw"}
        for r in raw:
            for k in range(HORIZON):
                _png(d / f"raw640_{short[r]}_{k}.png", r640[r][k])
                _png(d / f"deploy_{short[r]}_{k}.png", dep[r][k])
        np.savez(d / "deploy_obs.npz", **{k: obs[k] for k in ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start",
                                                               "gripper_width_m")},
                 main=dep["main"].astype(np.uint8), ultrawide=dep["ultrawide"].astype(np.uint8))
        cmp = {}
        for r in raw:
            for k in range(HORIZON):
                key = f"{short[r]}_{k}"
                cmp[key] = {"native_vs_zarr": _compare(native[r][k], obs[r][k]),
                            "deploy_vs_native": _compare(dep[r][k], native[r][k]),
                            "deploy_vs_zarr": _compare(dep[r][k], obs[r][k]),
                            "deploy_jpeg90_vs_zarr": _compare(depj[r][k], obs[r][k])}
        meta["raw"] = {
            "slot_dir": str(e.dir), "cam_files": {r: e.video(r).name for r in ("main", "ultrawide")},
            "sensors": e.sensor, "recorded_resolution": {r: list(e.res[r]) for r in e.res},
            "stream_resolution": list(STREAM_WH), "downscale": "cv2.INTER_AREA",
            "main_frame_idx": main_f, "main_dt_ms": [round(float(e.main_dt_ms[r]), 3) for r in meta["main_rows_in_episode"]],
            "uw_frame_idx": uw_f,
            "uw_frame_t_minus_row_T_ms": [round(float((e.t_cam["ultrawide"][f] - e.t_grid[ref["row"]]) / 1e6), 3) for f in uw_f],
            "uw_upsample_matches_episode_zarr": e.up_matches_stored, "n_uw_substream": e.n_uw,
            "mapping": "grid.npy rows in piece [t_first, t_last], validate kept span (piece.json trimmed), mid-exposure "
                       "camera times (t_ns - exposure/2), main = nearest frame, ultrawide = 10 Hz sub-stream (umiproc)",
            "image_compare": cmp,
        }
        _write_json(d / "meta.json", meta)
        files = sorted(p.name for p in d.iterdir())
        index.append({"folder": f"reference/{ref['folder']}", "files": files,
                      "episode": {"package_index": i, "pi_name": meta["pi_name"], "episode_name": meta["episode_name"],
                                  "slot": meta["slot"]},
                      "row": ref["row"], "label": ref["label"], "meta": meta})
        summary.append((ref["folder"], cmp, e.up_matches_stored, main_f, uw_f))
        print(f"{ref['folder']:<22} main frames {main_f} uw frames {uw_f} up==stored {e.up_matches_stored}")
        for key, c in cmp.items():
            n, dpl, dj = c["native_vs_zarr"], c["deploy_vs_zarr"], c["deploy_jpeg90_vs_zarr"]
            print(f"   {key:<7} native MAD {n['mad']:6.3f} PSNR {n['psnr_db']:5.1f} max {n['max_abs']:3d} | deploy MAD "
                  f"{dpl['mad_rgb']} PSNR {dpl['psnr_db']:5.1f} shift {dpl['best_int_shift_dxdy']} pc "
                  f"{dpl['phasecorr_shift_dxdy']} | +jpeg90 MAD {dj['mad']:6.3f} PSNR {dj['psnr_db']:5.1f}")

    # extra raw frame pairs: episode starts spread over the eraser positions (grasp TCP xy, tag frame)
    ref_idx = {r["index"] for r in _refs(layout)}
    n = n_episodes(dataset)
    gxy = {}
    for i in range(n):
        ep = load_episode(dataset, i)
        g = grasp_rows(ep["widths"])
        gxy[i] = ep["poses"][g["end"]][:2, 3]
    cand = [i for i in range(n) if i not in ref_idx]
    pts = np.array([gxy[i] for i in cand])
    med = np.median(np.array(list(gxy.values())), axis=0)
    chosen = [cand[int(np.argmax(np.linalg.norm(pts - med, axis=1)))]]
    while len(chosen) < min(N_EXTRA, len(cand)):
        dmin = np.min(np.stack([np.linalg.norm(pts - gxy[c], axis=1) for c in chosen]), axis=0)
        chosen.append(cand[int(np.argmax(dmin))])
    xd = OUT / "raw_extra"
    xd.mkdir(parents=True, exist_ok=True)
    extra = []
    for i in sorted(chosen):
        e = episode(i)
        mf = int(e.main_idx[EXTRA_ROW])
        uf = int(e.uw_sel[e.uw_up[EXTRA_ROW]])
        m = _to640(_decode(e.video("main"), [mf])[mf])
        u = _to640(_decode(e.video("ultrawide"), [uf])[uf])
        stem = f"ep{i:02d}_{pkg_eps[i]['pi_name'].split('_', 1)[1]}_row{EXTRA_ROW:03d}"
        _png(xd / f"{stem}_main.png", m)
        _png(xd / f"{stem}_uw.png", u)
        ep = load_episode(dataset, i)
        extra.append({"package_index": i, "pi_name": pkg_eps[i]["pi_name"], "slot": pkg_eps[i]["slot"],
                      "row_in_episode": EXTRA_ROW, "main_frame_idx": mf, "uw_frame_idx": uf,
                      "files": [f"{stem}_main.png", f"{stem}_uw.png"],
                      "tcp_tag_row": ep["poses"][EXTRA_ROW], "grasp_tcp_xy_tag": gxy[i],
                      "uw_upsample_matches_episode_zarr": e.up_matches_stored})
    _write_json(xd / "index.json", {"note": "640x480 frames (PyAV rgb24 decode, INTER_AREA from 1600x1200 / 1280x960) at "
                                            f"row {EXTRA_ROW} of each episode (main = nearest frame, uw = the newest "
                                            "10 Hz frame), episodes chosen by farthest-point sampling of the grasp "
                                            "TCP xy (eraser position) among the non-reference episodes",
                                    "frames": extra})
    _write_json(OUT / "index.json", {
        "package": layout["package"], "dataset": str(dataset), "tool": "sim_deploy/camera_study/reference.py",
        "obs_files": "obs.npz = training input (package images); deploy_obs.npz = same low-dim, deploy-path images",
        "references": index, "raw_extra": "reference/raw_extra/index.json"})
    print("extra episodes", sorted(chosen))
    print("wrote", OUT / "index.json")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "all"
    if cmd == "package":
        return stage_package()
    if cmd == "raw":
        return stage_raw()
    if cmd == "all":
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        for py, stage in ((ENV_TORCH, "package"), (ENV_AV, "raw")):
            rc = subprocess.call([str(py), "-m", "sim_deploy.camera_study.reference", stage], cwd=str(REPO), env=env)
            if rc:
                return rc
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
