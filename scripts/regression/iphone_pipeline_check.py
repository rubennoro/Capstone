#!/usr/bin/env python3
"""iPhone processing regression check: real iPhUMI pipeline on 5 fixed firstreal episodes vs a golden summary.

    python scripts/regression/iphone_pipeline_check.py                 # check (exit 1 on mismatch)
    python scripts/regression/iphone_pipeline_check.py --make-golden   # (re)write the golden file
    options: --python PY (iphumi env; default: this interpreter)  --code-root DIR (repo whose iPhUMI/ and scripts/
             are run; default: this repo)  --src DIR  --workdir DIR  --keep  --workers N  --strict
             --lowdim-atol X  --golden FILE  --with-label-video

What runs (the commands of scripts/cluster/process_dataset.sbatch, steps 1-4, session ``firstreal``, task
``block_raise``, sides=single), each with ``cwd=<workdir>/pipe`` and absolute Hydra overrides for every cwd-relative
default (demonstrations_dir, group.gripper_calibration, detect.aruco_config, sessions_dir, session_dir), so the repo's
iPhUMI ``tmp_demonstrations/`` / ``tmp_sessions/`` and ``data/raw`` are never written (verified with a before/after
snapshot). Inputs: ``<workdir>/raw/<date>/<episode>/`` are real directories of per-file symlinks to the originals
(group.py copies with shutil.copyfile, which follows them). The gripper calibration yaml is the code root's own file
(so an edit to it is exercised). ``PYTHONPATH=<code-root>/iPhUMI/python_package`` is prepended for every step and
``iphumi.__file__`` is checked to be inside the code root.

Deviation from the sbatch: ``label.task_extraction.visualize=false`` by default (it only renders a subtask overlay
video that the dataset never reads; ~doubles the runtime). ``--with-label-video`` restores the exact sbatch command.

What is compared (golden: scripts/regression/golden/iphone_firstreal_5ep.json):
- inputs: sha256 of every source file (a changed input is reported as such, not as a pipeline regression);
- intermediates: per demo folder valid/invalid, gripper range json, aligned csv rows + sha256;
- zarr structure: groups, keys, shape, dtype, chunks, compressor config, attrs;
- low-dim arrays (eef pos / rot, gripper width) and all meta arrays: sha256 of the decoded values, exact. On a
  mismatch the full golden values are used to print max |diff| and the first differing row. ``--lowdim-atol`` can
  accept a tiny float difference (e.g. a different OpenCV/numpy build on the cluster); default 0 = exact.
- images (JPEG-XL lossy, numthreads=1): sha256 of the decoded pixels, exact by default. Measured: three runs on one
  machine produced byte-identical zips (encoder is single-threaded and deterministic), so exact is the expected
  result. A different libjxl/imagecodecs build (e.g. the cluster env) may legitimately encode/decode slightly
  differently, so a decoded-hash mismatch falls back to per-frame mean RGB vs golden: max |diff| <= 0.5 grey levels
  and per-episode mean |diff| <= 0.1 -> PASS with a WARNING (``--strict`` turns that into a failure).
- compressed chunk bytes per array: informational only (never fails).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GOLDEN = HERE / "golden" / "iphone_firstreal_5ep.json"
DP_REL = Path("iPhUMI/python_package/iphumi/demonstration_processing")
SESSION, TASK, DATASET = "firstreal", "block_raise", "regress"
CALIB_ID = "15647"
DEMO_IDS = ["49242", "18798", "87784", "73132"]
DEFAULT_SRC = REPO / "data" / "raw" / "2026-10-01"
IMAGE_KEYS = ("camera_right_main_rgb", "camera_right_ultrawide_rgb")
IMG_FRAME_TOL, IMG_EPISODE_TOL = 0.5, 0.1


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================== summarizer (runs under --python)

def summarize_zarr(zarr_path: str, out_json: str) -> None:
    """Internal mode, executed with the pipeline's python and PYTHONPATH (needs zarr, numpy, iphumi codecs)."""
    import zipfile

    import numpy as np
    import zarr
    from iphumi.common.imagecodecs_numcodecs import register_codecs

    register_codecs()
    store = zarr.ZipStore(zarr_path, mode="r")
    root = zarr.open(store, mode="r")
    zf = zipfile.ZipFile(zarr_path)
    names = sorted(zf.namelist())
    out = {"root_attrs": dict(root.attrs), "groups": sorted(root.group_keys()), "arrays": {}}

    ends = root["meta"]["episode_ends"][:].tolist()
    uw_ends = root["meta"]["episode_ends_camera_right_ultrawide_rgb"][:].tolist() \
        if "episode_ends_camera_right_ultrawide_rgb" in root["meta"] else None

    for g in out["groups"]:
        for k in sorted(root[g].array_keys()):
            a = root[g][k]
            path = f"{g}/{k}"
            comp = a.compressor.get_config() if a.compressor is not None else None
            info = {"shape": list(a.shape), "dtype": str(a.dtype), "chunks": list(a.chunks), "compressor": comp,
                    "filters": [f.get_config() for f in (a.filters or [])], "attrs": dict(a.attrs)}
            raw = hashlib.sha256()
            for n in names:
                if n.startswith(path + "/") and not n.rsplit("/", 1)[-1].startswith("."):
                    raw.update(n.encode()); raw.update(zf.read(n))
            info["raw_chunks_sha256"] = raw.hexdigest()
            h = hashlib.sha256()
            if a.dtype == object:
                vals = [str(x) for x in a[:]]
                h.update(json.dumps(vals).encode())
                info["values"] = vals
            elif k in IMAGE_KEYS:
                means = []
                for s in range(0, a.shape[0], 128):
                    blk = np.ascontiguousarray(a[s:s + 128])
                    h.update(blk.tobytes())
                    means.extend(np.round(blk.reshape(blk.shape[0], -1, blk.shape[-1]).mean(1), 4).tolist())
                info["frame_mean_rgb"] = means
                e = uw_ends if "ultrawide" in k and uw_ends is not None else ends
                info["episode_ends_used"] = e
            else:
                v = np.ascontiguousarray(a[:])
                h.update(v.tobytes())
                info["values"] = v.tolist()  # float32 -> python float is exact; round-trips through JSON
            info["decoded_sha256"] = h.hexdigest()
            out["arrays"][path] = info
    with open(out_json, "w") as f:
        json.dump(out, f)


# ============================================================================== driver (stdlib only)

def run(cmd: list[str], cwd: Path, logfile: Path, env: dict) -> float:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logfile, "w") as f:
        f.write("$ " + " ".join(cmd) + f"\n# cwd={cwd}\n# PYTHONPATH={env.get('PYTHONPATH', '')}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0
    log(f"  {logfile.name}: rc={p.returncode} ({dt:.0f} s)")
    if p.returncode != 0:
        tail = logfile.read_text().splitlines()[-30:]
        raise RuntimeError(f"command failed (rc={p.returncode}), see {logfile}:\n" + "\n".join(tail))
    return dt


def snapshot(paths: list[Path], code_root: Path) -> dict:
    files = {}
    for root in paths:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for fn in filenames:
                p = Path(dirpath) / fn
                s = p.lstat()
                files[str(p)] = (s.st_size, s.st_mtime_ns)
    return files


def select_sources(src: Path) -> list[Path]:
    eps = []
    for d in sorted(src.iterdir()):
        parts = d.name.split("_")
        if len(parts) != 5 or parts[2] != SESSION:
            continue
        if (parts[3] == "grippercalibration" and parts[1] == CALIB_ID) or (
                parts[3] == "demonstration" and parts[1] in DEMO_IDS):
            eps.append(d)
    if len(eps) != 1 + len(DEMO_IDS):
        raise SystemExit(f"expected calibration {CALIB_ID} + demos {DEMO_IDS} in {src}, found {[e.name for e in eps]}")
    return eps


def git_info(root: Path) -> dict:
    def g(*a):
        p = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    dirty = g("status", "--porcelain", "--", "iPhUMI", "scripts")
    return {"commit": g("rev-parse", "HEAD"), "describe": g("log", "-1", "--format=%h %s"),
            "dirty_iphumi_or_scripts": bool(dirty)}


def pipeline(W: Path, raw: Path, code_root: Path, py: str, workers: int, label_video: bool, env: dict) -> dict:
    DP = code_root / DP_REL
    pipe = W / "pipe"
    pipe.mkdir(parents=True, exist_ok=True)
    demos, sessions = pipe / "tmp_demonstrations", pipe / "tmp_sessions"
    common = [f"demonstrations_dir={demos}", f"group.gripper_calibration={DP / 'calibration' / 'gripper_calibration.yaml'}",
              f"detect.aruco_config={DP / 'calibration' / 'aruco_config.yaml'}", f"filters.session_name=[{SESSION}]"]
    L, t = W / "logs", {}
    log("1. group")
    t["1_group"] = run([py, str(DP / "process_demos_iphone.py"), *common, f"group.iphone_dir={raw}", "stages=[group]"],
                       pipe, L / "1_group.log", env)
    log("2. label whole demo as task")
    t["2_label_whole"] = run([py, str(code_root / "scripts" / "label_whole_demo_as_task.py"), "--demonstrations-dir",
                              str(demos), "--session-name", SESSION, "--task-name", TASK], pipe, L / "2_label.log", env)
    log("3. detect, calibrate, align, validate, label")
    t["3_process"] = run([py, str(DP / "process_demos_iphone.py"), *common, f"num_workers={workers}",
                          "skip_stages=[group,visualize]"] + ([] if label_video else
                                                             ["label.task_extraction.visualize=false"]),
                         pipe, L / "3_process.log", env)
    log("4. create session + build replay buffer")
    t["4a_session"] = run([py, str(DP / "create_session_iphone.py"), f"demonstrations_dir={demos}",
                           f"sessions_dir={sessions}", f"input_session_filters=[{SESSION}]",
                           f"output_session_name={DATASET}", "sides=single"], pipe, L / "4a_session.log", env)
    t["4b_build"] = run([py, str(DP / "build_umi_dataset_iphone.py"), f"session_dir={sessions / DATASET}",
                         f"replay_buffer.num_workers={workers}"], pipe, L / "4b_build.log", env)
    z = sessions / DATASET / f"replay_buffer_{DATASET}.zarr.zip"
    if not z.is_file():
        raise RuntimeError(f"no zarr produced at {z}")
    return {"zarr": z, "timings_s": {k: round(v, 1) for k, v in t.items()}}


def intermediates(pipe: Path) -> dict:
    out = {}
    for d in sorted((pipe / "tmp_demonstrations").glob("*/*")):
        if not d.is_dir():
            continue
        e = {"valid": (d / "valid.txt").is_file(), "invalid": (d / "invalid.txt").is_file()}
        for side in ("left", "right"):
            rng = d / f"{side}_gripper_range.json"
            if rng.is_file():
                e[f"{side}_gripper_range"] = json.loads(rng.read_text())
            al = d / f"{side}_aligned.csv"
            if al.is_file():
                e[f"{side}_aligned_rows"] = sum(1 for _ in open(al))
                e[f"{side}_aligned_sha256"] = sha256_file(al)
        out[d.name] = e
    return out


# ---------------------------------------------------------------------------------------------- comparison

class Diff:
    def __init__(self):
        self.fail: list[str] = []
        self.warn: list[str] = []
        self.ok: list[str] = []
        self.info: list[str] = []


def _flat(v):
    if isinstance(v, list):
        for x in v:
            yield from _flat(x)
    else:
        yield v


def compare(golden: dict, cur: dict, strict: bool, atol: float) -> Diff:
    D = Diff()
    # inputs
    gi, ci = golden["inputs"], cur["inputs"]
    if gi != ci:
        changed = sorted(k for k in set(gi) | set(ci) if gi.get(k) != ci.get(k))
        D.fail.append(f"INPUT DATA differs from golden (not a pipeline regression): {changed[:5]}")
    else:
        D.ok.append(f"inputs: {len(gi)} source files identical (sha256)")
    # intermediates
    gm, cm = golden["intermediates"], cur["intermediates"]
    if sorted(gm) != sorted(cm):
        D.fail.append(f"tmp_demonstrations folders differ: golden {sorted(gm)} vs {sorted(cm)}")
    for name in sorted(set(gm) & set(cm)):
        short = name.split("_")[1]
        for k in sorted(set(gm[name]) | set(cm[name])):
            if gm[name].get(k) != cm[name].get(k):
                D.fail.append(f"intermediate {short}/{k}: golden {gm[name].get(k)} vs now {cm[name].get(k)}")
    if not any(m.startswith("intermediate") or m.startswith("tmp_demo") for m in D.fail):
        D.ok.append(f"intermediates: {len(cm)} demo folders (valid flags, gripper range, aligned csv) identical")
    # zarr structure
    gz, cz = golden["zarr"], cur["zarr"]
    for key in ("groups", "root_attrs"):
        if gz[key] != cz[key]:
            D.fail.append(f"zarr {key}: golden {gz[key]} vs now {cz[key]}")
    ga, ca = gz["arrays"], cz["arrays"]
    if sorted(ga) != sorted(ca):
        D.fail.append(f"zarr array keys differ: missing {sorted(set(ga) - set(ca))}, new {sorted(set(ca) - set(ga))}")
    for path in sorted(set(ga) & set(ca)):
        g, c = ga[path], ca[path]
        struct = [f for f in ("shape", "dtype", "chunks", "compressor", "filters", "attrs") if g[f] != c[f]]
        if struct:
            D.fail.append(f"{path}: {', '.join(f'{f} {g[f]} -> {c[f]}' for f in struct)}")
            continue
        desc = f"{path} {tuple(c['shape'])} {c['dtype']}"
        if g["raw_chunks_sha256"] != c["raw_chunks_sha256"]:
            D.info.append(f"{path}: compressed chunk bytes differ (informational)")
        if g["decoded_sha256"] == c["decoded_sha256"]:
            D.ok.append(f"{desc}: decoded values identical (sha256 {c['decoded_sha256'][:12]})")
            continue
        if "frame_mean_rgb" in g:  # images
            fg, fc = g["frame_mean_rgb"], c["frame_mean_rgb"]
            dif = [max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(fg, fc)]
            mx = max(dif) if dif else 0.0
            ends, s, worst_ep = g["episode_ends_used"], 0, 0.0
            for e in ends:
                seg = dif[s:e]
                if seg:
                    worst_ep = max(worst_ep, sum(seg) / len(seg))
                s = e
            msg = (f"{desc}: decoded pixels differ; per-frame mean RGB max |diff| {mx:.3f} (tol {IMG_FRAME_TOL}), "
                   f"worst episode mean |diff| {worst_ep:.3f} (tol {IMG_EPISODE_TOL})")
            if mx <= IMG_FRAME_TOL and worst_ep <= IMG_EPISODE_TOL and not strict:
                D.warn.append(msg + " -> within tolerance (different JPEG-XL build?)")
            else:
                D.fail.append(msg)
            continue
        gv, cv = g.get("values"), c.get("values")
        if gv is None or cv is None:
            D.fail.append(f"{desc}: decoded sha256 differs")
            continue
        if c["dtype"] == "object" or "int" in c["dtype"] or "bool" in c["dtype"]:
            first = next((i for i, (x, y) in enumerate(zip(gv, cv)) if x != y), None)
            D.fail.append(f"{desc}: values differ, first at index {first}: golden {gv[first] if first is not None else None}"
                          f" vs now {cv[first] if first is not None else None}")
            continue
        fg, fc = list(_flat(gv)), list(_flat(cv))
        diffs = [abs(a - b) for a, b in zip(fg, fc)]
        mx = max(diffs)
        width = max(1, len(fg) // max(1, c["shape"][0]))
        first = next(i for i, d in enumerate(diffs) if d > 0) // width
        n = sum(1 for d in diffs if d > 0)
        msg = f"{desc}: {n}/{len(diffs)} values differ, max |diff| {mx:.3g}, first differing row {first}"
        if mx <= atol:
            D.warn.append(msg + f" -> within --lowdim-atol {atol}")
        else:
            D.fail.append(msg)
    return D


# ---------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-golden", action="store_true", help="write the golden file instead of checking")
    ap.add_argument("--golden", default=str(GOLDEN))
    ap.add_argument("--python", default=sys.executable, help="python of the iphumi env (default: this interpreter)")
    ap.add_argument("--code-root", default=str(REPO), help="repo whose iPhUMI/ + scripts/ are run (default: this repo)")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="folder with the raw firstreal episodes (read-only)")
    ap.add_argument("--workdir", default=None, help="work dir (default: new temp dir, removed on success)")
    ap.add_argument("--keep", action="store_true", help="keep the temp work dir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="images must be bit-identical after decoding")
    ap.add_argument("--lowdim-atol", type=float, default=0.0, help="accept low-dim float diffs up to this")
    ap.add_argument("--with-label-video", action="store_true", help="render the label overlay video like the sbatch")
    ap.add_argument("--summarize", nargs=2, metavar=("ZARR", "OUT_JSON"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.summarize:
        summarize_zarr(*a.summarize)
        return 0

    t_all = time.time()
    code_root = Path(a.code_root).resolve()
    DP = code_root / DP_REL
    src = Path(a.src).resolve()
    if a.workdir:
        W = Path(a.workdir).resolve()
        if W.exists() and any(W.iterdir()):
            raise SystemExit(f"work dir {W} is not empty")
        W.mkdir(parents=True, exist_ok=True)
    else:
        W = Path(tempfile.mkdtemp(prefix="iphone_regress_"))
    for forbidden in {DP.resolve(), (REPO / DP_REL).resolve(), src, (REPO / "data").resolve(),
                      (code_root / "data").resolve()}:
        if W == forbidden or forbidden in W.parents:
            raise SystemExit(f"refusing: work dir {W} is inside {forbidden}")

    env = dict(os.environ)
    pkg = str(code_root / "iPhUMI" / "python_package")
    env.update({"PYTHONPATH": pkg + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "HYDRA_FULL_ERROR": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    # iphumi is a namespace package (iphumi.__file__ is None): check __path__[0] and real module files instead.
    probe_src = ("import sys, json, numpy, cv2, zarr, iphumi, iphumi.common.imagecodecs_numcodecs as a, "
                 "iphumi.common.replay_buffer as b, iphumi.demonstration_processing.utils.generic_util as c; "
                 "print(json.dumps({'path': list(iphumi.__path__), 'files': [a.__file__, b.__file__, c.__file__], "
                 "'versions': ' '.join([sys.version.split()[0], numpy.__version__, cv2.__version__, "
                 "zarr.__version__])}))")
    probe = subprocess.run([a.python, "-c", probe_src], cwd=str(W), env=env, capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(f"{a.python} cannot import iphumi/numpy/cv2/zarr:\n{probe.stderr}")
    pr = json.loads(probe.stdout.strip().splitlines()[-1])
    iphumi_file, versions = pr["path"][0], pr["versions"]
    for f in [iphumi_file, *pr["files"]]:
        if not Path(f).resolve().is_relative_to(code_root):
            raise SystemExit(f"iphumi module imported from {f}, not from code root {code_root} (iphumi.__path__ = "
                             f"{pr['path']})")
    log(f"code root {code_root}  ({git_info(code_root)['describe']})")
    log(f"iphumi.__path__[0] = {iphumi_file}; modules {pr['files']}")
    log(f"python/numpy/cv2/zarr = {versions}; work dir {W}")

    watched = [DP / "tmp_demonstrations", DP / "tmp_sessions", src, REPO / DP_REL / "tmp_demonstrations",
               REPO / DP_REL / "tmp_sessions"]
    snap0 = snapshot(watched, code_root)

    eps = select_sources(src)
    inputs = {}
    raw = W / "raw"
    for d in eps:
        out = raw / d.name[:10] / d.name
        out.mkdir(parents=True, exist_ok=True)
        for f in sorted(d.iterdir()):
            (out / f.name).symlink_to(f.resolve())
            inputs[f"{d.name}/{f.name}"] = sha256_file(f)

    res = pipeline(W, raw, code_root, a.python, a.workers, a.with_label_video, env)
    summ = W / "zarr_summary.json"
    log("summarize zarr")
    t0 = time.time()
    subprocess.run([a.python, str(Path(__file__).resolve()), "--summarize", str(res["zarr"]), str(summ)],
                   cwd=str(W), env=env, check=True)
    res["timings_s"]["5_summarize"] = round(time.time() - t0, 1)
    cur = {"inputs": inputs, "intermediates": intermediates(W / "pipe"), "zarr": json.loads(summ.read_text())}

    snap1 = snapshot(watched, code_root)
    touched = sorted(p for p in set(snap0) | set(snap1) if snap0.get(p) != snap1.get(p))

    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code_root": str(code_root), "git": git_info(code_root),
            "iphumi_file": iphumi_file, "python_numpy_cv2_zarr": versions, "src": str(src),
            "episodes": [e.name for e in eps], "workers": a.workers, "label_video": a.with_label_video,
            "timings_s": res["timings_s"]}
    rc = 0
    if touched:
        print(f"FAIL: repo/data files changed during the run: {touched[:5]}")
        rc = 1
    if a.make_golden:
        g = Path(a.golden)
        g.parent.mkdir(parents=True, exist_ok=True)
        g.write_text(json.dumps({"meta": meta, **cur}, indent=None, separators=(",", ":")) + "\n")
        n = cur["zarr"]["arrays"]
        log(f"golden written: {g} ({g.stat().st_size / 1e3:.0f} kB); "
            f"{len(n)} arrays, {n['meta/episode_ends']['values'][-1]} frames, {len(cur['intermediates'])} demo dirs")
    else:
        golden = json.loads(Path(a.golden).read_text())
        D = compare(golden, cur, a.strict, a.lowdim_atol)
        gm = golden["meta"]
        print(f"\niPhone pipeline regression check: golden from {gm['git']['describe']} ({gm['created']}, "
              f"code root {gm['code_root']}, dirty={gm['git']['dirty_iphumi_or_scripts']})")
        print(f"                                  now from    {meta['git']['describe']} "
              f"(code root {code_root}, dirty={meta['git']['dirty_iphumi_or_scripts']})")
        for tag, rows in (("OK  ", D.ok), ("INFO", D.info), ("WARN", D.warn), ("FAIL", D.fail)):
            for r in rows:
                print(f"  {tag} {r}")
        verdict = "FAIL" if D.fail else ("PASS (with warnings)" if D.warn else "PASS (exact)")
        print(f"RESULT: {verdict}: {len(D.ok)} identical, {len(D.warn)} within tolerance, {len(D.fail)} mismatches")
        (W / "check_report.json").write_text(json.dumps({"meta": meta, "ok": D.ok, "info": D.info, "warn": D.warn,
                                                         "fail": D.fail}, indent=1))
        rc = rc or (1 if D.fail else 0)
    log(f"total {time.time() - t_all:.0f} s; step timings {res['timings_s']}")
    if rc == 0 and not a.keep and not a.workdir:
        shutil.rmtree(W, ignore_errors=True)
    else:
        log(f"work dir kept: {W}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
