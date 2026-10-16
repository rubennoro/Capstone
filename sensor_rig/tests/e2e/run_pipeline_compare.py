#!/usr/bin/env python3
"""D10c end-to-end check: real iPhone episodes vs simulated-rig exports through the REAL iPhUMI pipeline.

    $PY sensor_rig/tests/e2e/run_pipeline_compare.py --workdir <dir> [--demos 49242,18798,87784,73132] [--keep]

Steps (P1 §5.3). Every path is absolute and inside the work dir W; the repo's iPhUMI ``tmp_demonstrations`` /
``tmp_sessions`` and ``data/raw`` are never written (checked by a before/after snapshot, see ``repo_snapshot``):

1. ``W/orig_raw/<date>/<NAME>/``: real directories with per-file symlinks to the originals (group.py copies with
   ``shutil.copyfile``, which follows them; ``group.symlink`` stays false so the label script can never write through a
   link into ``data/raw``).
2. ``W/gripper_calibration.yaml``: copy of the repo yaml, ``default["JetsonRig1,1"]`` added (copy of iPhone17,1) if
   missing. Both runs use it.
3. ``sensor_rig sim --no-import`` -> ``pose-import`` (the simulated VIO trajectory, through the real import checks)
   -> ``export`` -> ``validate`` (CLI, as a user would run them).
4. For X in {A: originals, B: rig exports}: the commands of ``scripts/cluster/process_dataset.sbatch`` with
   ``cwd=W/pipe_X`` and absolute Hydra overrides for every cwd-relative default (``demonstrations_dir``,
   ``group.gripper_calibration``, ``detect.aruco_config``, ``sessions_dir``, ``session_dir``). Deviation from the
   sbatch: ``label.task_extraction.visualize=false`` (it only renders an overlay video, not used by the dataset).
5. Intermediate checks (valid.txt, gripper range, tag detection, aligned length) and the zarr comparison.
6. ``W/report.json`` + a printed table; exit 0 iff every check passes.

Tolerances (justification):
- TCP relative motion < 2e-3 m / 2e-3 rad: the exported poses equal the originals up to a world transform to ~1e-6
  (T2), but align.py interpolates rotation vectors component-wise (align.py:70-76). That interpolation is not
  equivariant under a change of world frame, so the 60 Hz resampled poses differ slightly between A and B.
- Gripper width max < 5e-3, median < 1e-3 (width units ~ metres of tag separation): both runs detect the same tags
  in images that differ only by resampling (iPhone -> physical lens -> virtual) and by the encoder; PnP noise on a
  16 mm tag at ~10 cm is ~1e-4 m, and the calibration's min/max are raw extrema, so a few-mm max error is the
  expected worst case at single frames; the median should be at the sub-mm level.
- Main images at 224 px: mean MAE < 4, p99 < 8 (T4 measured ~2.6 on the full pipeline path; the export's yuv420p
  round trip adds a ~-2 level bias, see report); UW mean MAE < 6 (smaller image, harder fisheye resampling).
  Only for demos whose iPhone K equals the virtual K (others are a different crop by design).
- Calibration min/max width within 2e-3; tag detection fraction within 0.05; aligned length within ±1.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
DP = REPO / "iPhUMI" / "python_package" / "iphumi" / "demonstration_processing"
SCRATCH = Path("/tmp/claude-1000/-home-ruben-Capstone/5ee0411f-3d2e-4964-b2ee-0d8c8e48aa78/scratchpad")
DEFAULT_SRC = REPO / "data" / "raw" / "2026-10-01"
CALIB_ID = "15647"
DEFAULT_DEMOS = "49242,18798,87784,73132"
SESSION = "firstreal"
TASK = "block_raise"
DATASET = "cmp"

TOL = {
    "calib_width": 2e-3, "tag_frac": 0.05, "aligned_len": 1,
    "tcp_rel_trans": 2e-3, "tcp_rel_rot": 2e-3, "tcp_world_resid": 2e-3,
    "width_max": 5e-3, "width_median": 1e-3,
    "main_mae_mean": 4.0, "main_mae_p99": 8.0, "uw_mae_mean": 6.0,
}


# ------------------------------------------------------------------------------------------------ helpers

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], cwd: Path, logfile: Path, env_extra: dict | None = None) -> None:
    env = dict(os.environ)
    env.update({"HYDRA_FULL_ERROR": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(env_extra or {})
    logfile.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logfile, "w") as f:
        f.write("$ " + " ".join(cmd) + f"\n# cwd={cwd}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
    log(f"  {logfile.name}: rc={p.returncode} ({time.time() - t0:.0f} s)")
    if p.returncode != 0:
        tail = logfile.read_text().splitlines()[-30:]
        raise RuntimeError(f"command failed (rc={p.returncode}), see {logfile}:\n" + "\n".join(tail))


def repo_snapshot() -> dict:
    """State of everything the pipeline could touch by accident: git status of iPhUMI/ and scripts/ (incl. ignored
    files, minus __pycache__), every file under DP/ (size, mtime), and the source data files (size, mtime)."""
    st = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", "iPhUMI", "scripts", "data"],
                        cwd=REPO, capture_output=True, text=True).stdout.splitlines()
    st = sorted(x for x in st if "__pycache__" not in x)
    files = {}
    for root in (DP, DEFAULT_SRC):
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                p = Path(dirpath) / fn
                s = p.lstat()
                files[str(p)] = (s.st_size, s.st_mtime_ns)
    return {"git": st, "files": files}


def snapshot_diff(a: dict, b: dict) -> list[str]:
    out = []
    if a["git"] != b["git"]:
        out.append(f"git status changed: {sorted(set(a['git']) ^ set(b['git']))}")
    fa, fb = a["files"], b["files"]
    for p in sorted(set(fa) | set(fb)):
        if fa.get(p) != fb.get(p):
            out.append(f"{p}: {fa.get(p)} -> {fb.get(p)}")
    return out


def select_sources(src: Path, demo_ids: list[str]) -> list[Path]:
    eps = []
    for d in sorted(src.iterdir()):
        parts = d.name.split("_")
        if len(parts) != 5 or parts[2] != SESSION:
            continue
        if (parts[3] == "grippercalibration" and parts[1] == CALIB_ID) or (
                parts[3] == "demonstration" and parts[1] in demo_ids):
            eps.append(d)
    return eps


def link_originals(eps: list[Path], dst_root: Path) -> None:
    for d in eps:
        out = dst_root / d.name[:10] / d.name
        out.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            if not (out / f.name).exists():
                (out / f.name).symlink_to(f.resolve())


def make_calibration_yaml(dst: Path) -> None:
    with open(DP / "calibration" / "gripper_calibration.yaml") as f:
        cal = yaml.safe_load(f)
    for gid, devices in cal.items():
        if "JetsonRig1,1" not in devices and "iPhone17,1" in devices:
            devices["JetsonRig1,1"] = json.loads(json.dumps(devices["iPhone17,1"]))
    with open(dst, "w") as f:
        yaml.safe_dump(cal, f)


def pipeline(X: str, raw_dir: Path, W: Path, py: str, workers: int) -> Path:
    pipe = W / f"pipe_{X}"
    pipe.mkdir(parents=True, exist_ok=True)
    demos = pipe / "tmp_demonstrations"
    sessions = pipe / "tmp_sessions"
    zarr_path = sessions / DATASET / f"replay_buffer_{DATASET}.zarr.zip"
    if zarr_path.is_file():
        log(f"pipeline {X}: {zarr_path} exists, skipping")
        return zarr_path
    common = [f"demonstrations_dir={demos}", f"group.gripper_calibration={W / 'gripper_calibration.yaml'}",
              f"detect.aruco_config={DP / 'calibration' / 'aruco_config.yaml'}", f"filters.session_name=[{SESSION}]"]
    L = W / "logs"
    log(f"pipeline {X}: group")
    run([py, str(DP / "process_demos_iphone.py"), *common, f"group.iphone_dir={raw_dir}", "stages=[group]"],
        pipe, L / f"{X}_1_group.log")
    log(f"pipeline {X}: label whole demo as task")
    run([py, str(REPO / "scripts" / "label_whole_demo_as_task.py"), "--demonstrations-dir", str(demos),
         "--session-name", SESSION, "--task-name", TASK], pipe, L / f"{X}_2_label_whole.log")
    log(f"pipeline {X}: detect, calibrate, align, validate, label")
    run([py, str(DP / "process_demos_iphone.py"), *common, f"num_workers={workers}",
         "skip_stages=[group,visualize]", "label.task_extraction.visualize=false"], pipe, L / f"{X}_3_process.log")
    log(f"pipeline {X}: create session")
    run([py, str(DP / "create_session_iphone.py"), f"demonstrations_dir={demos}", f"sessions_dir={sessions}",
         f"input_session_filters=[{SESSION}]", f"output_session_name={DATASET}", "sides=single"],
        pipe, L / f"{X}_4_session.log")
    log(f"pipeline {X}: build dataset")
    run([py, str(DP / "build_umi_dataset_iphone.py"), f"session_dir={sessions / DATASET}",
         f"replay_buffer.num_workers={workers}"], pipe, L / f"{X}_5_build.log")
    if not zarr_path.is_file():
        raise RuntimeError(f"pipeline {X} produced no {zarr_path}")
    return zarr_path


# ------------------------------------------------------------------------------------------------ comparison

class Report:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, check: str, value, tol=None, ok: bool | None = None, note: str = "") -> bool:
        if ok is None:
            ok = bool(value <= tol) if tol is not None else bool(value)
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.rows.append({"check": check, "value": value, "tol": tol, "ok": bool(ok), "note": note})
        return ok

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)

    def table(self) -> str:
        lines = [f"{'check':58s} {'value':>14s} {'tolerance':>11s}  result"]
        for r in self.rows:
            v = r["value"]
            vs = f"{v:.3g}" if isinstance(v, float) else str(v)
            ts = "" if r["tol"] is None else (f"{r['tol']:.3g}" if isinstance(r["tol"], float) else str(r["tol"]))
            lines.append(f"{r['check'][:58]:58s} {vs:>14s} {ts:>11s}  {'PASS' if r['ok'] else 'FAIL'}"
                         + (f"  ({r['note']})" if r["note"] else ""))
        return "\n".join(lines)


def _demo_dirs(pipe: Path) -> dict[str, Path]:
    return {d.name: d for d in sorted((pipe / "tmp_demonstrations").glob("*/*")) if d.is_dir()}


def _tag_fraction(path: Path, tag: int) -> float:
    with open(path) as f:
        det = json.load(f)
    if not det:
        return 0.0
    return sum(1 for d in det if str(tag) in d["tag_dict"] or tag in d["tag_dict"]) / len(det)


def intermediate_checks(W: Path, rep: Report) -> None:
    A, B = _demo_dirs(W / "pipe_A"), _demo_dirs(W / "pipe_B")
    rep.add("tmp_demonstrations: same demo folders", sorted(A) == sorted(B), note=f"{len(A)} folders")
    for name in sorted(set(A) & set(B)):
        a, b = A[name], B[name]
        short = name.split("_")[1]
        if "grippercalibration" in name:
            ga = json.load(open(a / "left_gripper_range.json"))
            gb = json.load(open(b / "left_gripper_range.json"))
            rep.add(f"calib {short}: gripper_id equal", ga["gripper_id"] == gb["gripper_id"])
            for k in ("min_width", "max_width"):
                rep.add(f"calib {short}: |{k} B-A|", abs(gb[k] - ga[k]), TOL["calib_width"],
                        note=f"A={ga[k]:.5f} B={gb[k]:.5f}")
        else:
            rep.add(f"demo {short}: valid.txt in B", (b / "valid.txt").is_file() or not (a / "valid.txt").is_file(),
                    note=f"A valid={(a / 'valid.txt').is_file()}")
            la = sum(1 for _ in open(a / "left_aligned.csv"))
            lb = sum(1 for _ in open(b / "left_aligned.csv"))
            rep.add(f"demo {short}: aligned.csv length diff", abs(la - lb), TOL["aligned_len"], note=f"A={la} B={lb}")
        for tag in (0, 1):
            fa = _tag_fraction(a / "left_tag_detection.json", tag)
            fb = _tag_fraction(b / "left_tag_detection.json", tag)
            rep.add(f"{short}: tag {tag} detection fraction diff", abs(fa - fb), TOL["tag_frac"],
                    note=f"A={fa:.3f} B={fb:.3f}")


def _open_zarr(path: Path):
    import zarr
    from iphumi.common.imagecodecs_numcodecs import register_codecs

    register_codecs()
    return zarr.open(zarr.ZipStore(str(path), mode="r"), mode="r")


def _virtual_k_demos(W: Path) -> dict[str, bool]:
    """5-digit episode id -> True if the original iPhone main AND UW K equal the rig's virtual K (zarr episode
    names drop the side suffix, so episodes are keyed by their id)."""
    from sensor_rig.rig_config import load_rig
    from sensor_rig.sim.simulator import SIM_RIG_PATH

    rig = load_rig(SIM_RIG_PATH)
    Km = rig.virtual_camera("main", "reference").K
    Ku = rig.virtual_camera("ultrawide", "reference").K
    out = {}
    for j in (W / "orig_raw").glob("*/*/*.json"):
        d = json.load(open(j))
        out[j.stem.split("_")[1]] = bool(np.allclose(d["mainCameraIntrinsics"], Km, atol=1e-3)
                           and np.allclose(d["ultrawideCameraIntrinsics"], Ku, atol=1e-3))
    return out


def zarr_checks(W: Path, za_path: Path, zb_path: Path, rep: Report, image_stride: int = 1) -> None:
    from sensor_rig.sim.compare import masked_mae, pose6_to_T, rel_pose_errors, world_transform_residual

    za, zb = _open_zarr(za_path), _open_zarr(zb_path)
    ka, kb = sorted(za["data"].keys()), sorted(zb["data"].keys())
    rep.add("zarr data/ key sets equal", ka == kb, note=",".join(ka))
    ma, mb = sorted(za["meta"].keys()), sorted(zb["meta"].keys())
    rep.add("zarr meta/ key sets equal", ma == mb)
    for k in sorted(set(ka) & set(kb)):
        a, b = za["data"][k], zb["data"][k]
        rep.add(f"data/{k}: dtype + per-item shape equal", a.dtype == b.dtype and a.shape[1:] == b.shape[1:],
                note=f"A{a.shape} {a.dtype} B{b.shape} {b.dtype}")
    for k in ("camera_right_main_rgb", "camera_right_ultrawide_rgb"):
        if k in kb:
            rep.add(f"{k} is (N,224,224,3) uint8 in B",
                    zb["data"][k].shape[1:] == (224, 224, 3) and zb["data"][k].dtype == np.uint8)
    rep.add("gripper_right_gripper_width present in B", "gripper_right_gripper_width" in kb)
    names_a = [str(x) for x in za["meta"]["episode_names"][:]]
    names_b = [str(x) for x in zb["meta"]["episode_names"][:]]
    rep.add("episode names equal", names_a == names_b, note=f"{len(names_a)} episodes")
    ta = [str(x) for x in za["meta"]["task_names"][:]] if "task_names" in ma else []
    tb = [str(x) for x in zb["meta"]["task_names"][:]] if "task_names" in mb else []
    rep.add("meta/task_names equal", ta == tb, note=",".join(sorted(set(ta))))

    ends_a, ends_b = za["meta"]["episode_ends"][:], zb["meta"]["episode_ends"][:]
    uwe_a = za["meta"]["episode_ends_camera_right_ultrawide_rgb"][:]
    uwe_b = zb["meta"]["episode_ends_camera_right_ultrawide_rgb"][:]
    up_a = za["meta"]["upsample_index_camera_right_ultrawide_rgb"][:]
    up_b = zb["meta"]["upsample_index_camera_right_ultrawide_rgb"][:]
    rep.add("episode_ends consistent with data length (B)", int(ends_b[-1]) == zb["data"]["gripper_right_eef_pos"].shape[0])
    comparable = _virtual_k_demos(W)
    pos_a, pos_b = za["data"]["gripper_right_eef_pos"][:], zb["data"]["gripper_right_eef_pos"][:]
    rot_a, rot_b = za["data"]["gripper_right_eef_rot_axis_angle"][:], zb["data"]["gripper_right_eef_rot_axis_angle"][:]
    w_a = za["data"]["gripper_right_gripper_width"][:, 0] if "gripper_right_gripper_width" in ka else None
    w_b = zb["data"]["gripper_right_gripper_width"][:, 0] if "gripper_right_gripper_width" in kb else None
    main_a, main_b = za["data"]["camera_right_main_rgb"], zb["data"]["camera_right_main_rgb"]
    uw_a, uw_b = za["data"]["camera_right_ultrawide_rgb"], zb["data"]["camera_right_ultrawide_rgb"]

    for e, name in enumerate(names_a):
        if name not in names_b:
            continue
        eb = names_b.index(name)
        short = name.split("_")[1]
        sa = slice(0 if e == 0 else int(ends_a[e - 1]), int(ends_a[e]))
        sb = slice(0 if eb == 0 else int(ends_b[eb - 1]), int(ends_b[eb]))
        na, nb = sa.stop - sa.start, sb.stop - sb.start
        rep.add(f"ep {short}: length diff", abs(na - nb), 1, note=f"A={na} B={nb}")
        ua0 = 0 if e == 0 else int(uwe_a[e - 1])
        ub0 = 0 if eb == 0 else int(uwe_b[eb - 1])
        nua, nub = int(uwe_a[e]) - ua0, int(uwe_b[eb]) - ub0
        rep.add(f"ep {short}: N_uw equal", nua == nub, note=f"A={nua} B={nub}")
        n = min(na, nb)
        ia = up_a[sa][:n] - ua0
        ib = up_b[sb][:n] - ub0
        rep.add(f"ep {short}: UW upsample index equal (common prefix)", bool(np.array_equal(ia, ib)),
                note=f"{int((ia != ib).sum())} differ")
        Ta = pose6_to_T(pos_a[sa][:n].astype(np.float64), rot_a[sa][:n].astype(np.float64))
        Tb = pose6_to_T(pos_b[sb][:n].astype(np.float64), rot_b[sb][:n].astype(np.float64))
        worst_r, worst_t = 0.0, 0.0
        for k in (1, 3, 60, "all"):
            r, t = rel_pose_errors(Tb, Ta, k)
            worst_r, worst_t = max(worst_r, r), max(worst_t, t)
        rep.add(f"ep {short}: TCP relative motion trans err [m]", worst_t, TOL["tcp_rel_trans"],
                note="k in 1,3,60,all-to-0")
        rep.add(f"ep {short}: TCP relative motion rot err [rad]", worst_r, TOL["tcp_rel_rot"])
        _, rr, rt = world_transform_residual(Tb, Ta)
        rep.add(f"ep {short}: B = D*A residual (trans m / rot rad)", max(rr, rt), TOL["tcp_world_resid"],
                note=f"trans {rt:.2e} rot {rr:.2e}")
        if w_a is not None and w_b is not None:
            dw = np.abs(w_b[sb][:n] - w_a[sa][:n])
            rep.add(f"ep {short}: gripper width max |B-A|", float(dw.max()), TOL["width_max"],
                    note=f"range A {w_a[sa].min():.4f}..{w_a[sa].max():.4f}")
            rep.add(f"ep {short}: gripper width median |B-A|", float(np.median(dw)), TOL["width_median"])
        if comparable.get(short):
            idx = np.arange(0, n, image_stride)
            ma_ = main_a[sa.start + idx[0]: sa.start + idx[-1] + 1][::image_stride]
            mb_ = main_b[sb.start + idx[0]: sb.start + idx[-1] + 1][::image_stride]
            mae = np.array([masked_mae(x, y) for x, y in zip(ma_, mb_)])
            rep.add(f"ep {short}: main 224 MAE mean", float(mae.mean()), TOL["main_mae_mean"],
                    note=f"{len(mae)} frames")
            rep.add(f"ep {short}: main 224 MAE p99", float(np.percentile(mae, 99)), TOL["main_mae_p99"])
            nu = min(nua, nub)
            xa = uw_a[ua0: ua0 + nu]
            xb = uw_b[ub0: ub0 + nu]
            umae = np.array([masked_mae(x, y) for x, y in zip(xa, xb)])
            rep.add(f"ep {short}: UW 224 MAE mean", float(umae.mean()), TOL["uw_mae_mean"], note=f"{nu} frames")
        else:
            rep.add(f"ep {short}: image comparison skipped", True,
                    note="iPhone K != virtual K (different crop by design)")


# ------------------------------------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workdir", default=None, help="work dir (default: a new temp dir in the scratchpad)")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="folder with the real iPhone episodes (read-only)")
    ap.add_argument("--demos", default=DEFAULT_DEMOS, help="comma-separated 5-digit ids of firstreal demos")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="keep the work dir on success")
    ap.add_argument("--python", default=sys.executable, help="python of the iphumi env")
    ap.add_argument("--image-stride", type=int, default=1, help="compare every n-th 60 Hz main image")
    args = ap.parse_args(argv)

    py = args.python
    if args.workdir:
        W = Path(args.workdir).resolve()
        W.mkdir(parents=True, exist_ok=True)
    else:
        base = SCRATCH / "e2e" if SCRATCH.is_dir() else Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        W = Path(tempfile.mkdtemp(prefix="d10c_", dir=base))
    src = Path(args.src).resolve()
    for forbidden in (DP.resolve(), (REPO / "data").resolve()):
        if W == forbidden or forbidden in W.parents:
            raise SystemExit(f"refusing: work dir {W} is inside {forbidden}")
    log(f"work dir {W}")
    snap0 = repo_snapshot()
    rep = Report()
    t_start = time.time()

    eps = select_sources(src, [x.strip() for x in args.demos.split(",") if x.strip()])
    n_cal = sum("grippercalibration" in e.name for e in eps)
    if n_cal != 1 or len(eps) < 2:
        raise SystemExit(f"need 1 calibration + >=1 demo, found {[e.name for e in eps]}")
    log(f"episodes: {[e.name.split('_')[1] for e in eps]}")
    link_originals(eps, W / "orig_raw")
    make_calibration_yaml(W / "gripper_calibration.yaml")

    # ---- sim -> pose-import -> export -> validate (CLI)
    L = W / "logs"
    rig_raw, rig_exp = W / "rig_raw", W / "rig_export"
    sr = [py, "-m", "sensor_rig"]
    if not rig_raw.exists():
        log("sim")
        run(sr + ["sim", "--iphone", *[str(W / "orig_raw" / e.name[:10] / e.name) for e in eps],
                  "--out", str(rig_raw), "--no-import", "--workers", str(args.workers)], W, L / "sim.log")
        from sensor_rig.raw_episode import find_raw_episodes

        for ep in find_raw_episodes(rig_raw):
            traj = ep.path / "sim" / "vio_trajectory.tum"
            if traj.is_file():
                run(sr + ["pose-import", "--raw", str(ep.path), "--traj", str(traj), "--format", "tum",
                          "--frame", "imu"], W, L / f"pose_import_{ep.name.split('_')[1]}.log")
    if not rig_exp.exists():
        log("export")
        run(sr + ["export", "--raw", str(rig_raw), "--out", str(rig_exp), "--workers", str(args.workers)],
            W, L / "export.log")
    log("validate")
    run(sr + ["validate", str(rig_exp), "--gripper-calibration", str(W / "gripper_calibration.yaml"), "--decode"],
        W, L / "validate.log")
    rep.add("sensor_rig validate (rig exports, --decode)", True, note=(L / "validate.log").read_text()
            .strip().splitlines()[-1])
    imports = {}
    for p in rig_raw.glob("*/*/trajectory/trajectory.json"):
        info = json.load(open(p))
        imports[p.parents[1].name] = info
        short = p.parents[1].name.split("_")[1]
        rep.add(f"pose-import {short}: gravity angle [deg]", info["gravity_check"]["angle_to_up_deg"], 3.0)
        rep.add(f"pose-import {short}: |traj delay| [ms]", abs(info["time_offset_check"]["traj_delay_ms"]), 5.0)

    # ---- pipelines
    za = pipeline("A", W / "orig_raw", W, py, args.workers)
    zb = pipeline("B", rig_exp, W, py, args.workers)

    # ---- comparisons
    intermediate_checks(W, rep)
    zarr_checks(W, za, zb, rep, args.image_stride)

    snap1 = repo_snapshot()
    diff = snapshot_diff(snap0, snap1)
    rep.add("repo untouched (iPhUMI/, scripts/, data/raw)", not diff, note="; ".join(diff[:5]))

    report = {"workdir": str(W), "ok": rep.ok, "tolerances": TOL, "rows": rep.rows,
              "episodes": [e.name for e in eps], "runtime_s": round(time.time() - t_start, 1),
              "zarr": {"A": str(za), "B": str(zb)}}
    (W / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(rep.table())
    print(f"\n{'ALL PASS' if rep.ok else 'FAILURES'}; report: {W / 'report.json'}")
    if rep.ok and not args.keep and not args.workdir:
        shutil.rmtree(W, ignore_errors=True)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
