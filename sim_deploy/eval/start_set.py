"""Start sets (W3, user decisions 2026-11-22): one fixed list of eraser placements, the same for every checkpoint.

    python -m sim_deploy.eval starts build [--plan YAML | --checkpoints ID ...] [--look robot] [--name w3]
    python -m sim_deploy.eval starts show [FILE]
    python -m sim_deploy.eval starts check [FILE] [--look robot]      the file against the current scene

File ``sim_deploy/eval/start_sets/<name>.yaml`` (human readable, committed; ``w3.yaml`` for the W3 batch). Rollout k of
an eval with ``--starts set[:FILE]`` uses ``starts[k]``. W3 has 18:

1. **Rollouts 0-9 = the shared demos 0-9.** All 9 training datasets begin with the same 10 demos (the whole of
   pick1-ep10-v1, PICK_TEST1 2026-11-18 episodes 138-147); ``build`` checks that every dataset's first 10 demos have
   the same names, row counts, poses and widths, and records it. Demo k's placement = ``Scene.replay_placement`` at the
   start pose (scene/placement.py): ``O_T_grasp = O_T_TCP_start . inv(P_s) . P_g`` with P_s the demo's first recorded
   TCP pose and P_g the TCP pose when its fingers start to close; the eraser centre half its resting height behind the
   grasp fingertip point along the approach axis, its width along the grasp's closing axis projected on the table,
   laid flat at that x, y. The vertical part of the transfer is dropped (``height_error_m``). Every checkpoint holds
   exactly one of these demos out for validation (its own training split, below): that rollout is recorded
   ``held_out: true`` for that checkpoint, so the report shows the 9 trained demos and the 1 held-out demo apart.
2. **Rollouts 10-17 = the real-arm trials of 2026-11-20** (the 8 fitted in W2,
   ``sim_evals/_pipeline/W2/sets/robot_trials_fit.json``: the eraser's centre and yaw fitted to the IMX219 / IMX519
   images at q_start). The real object was the 51 mm eraser; the sim lays its own 30 mm eraser (sim_deploy/eraser.py;
   the real arm will use a <= 38 mm object, user decision) flat on the table at the fitted centre x, y and yaw. The
   values are copied into the file (with the fit file's sha256): the file does not change when W2 is re-run.
3. **Training split, per checkpoint** (``split_probe.py`` in ``~/envs/sim_policy_deploy_torch``, CPU): the checkpoint's
   own ``_extra_training_split_info`` (authoritative) and behavior_prompting's ``get_val_mask`` recomputed on the local
   dataset with the run's ``sample_type`` / ``val_ratio`` / ``seed``. ``held_out`` = the demos 0-9 (matched by
   episode name) in either val set; a disagreement between the two is written as a problem.
4. **Checks** per start (reported, not substituted): the whole eraser on the table (``on_table``), its centre and
   footprint inside the sim PC's workspace box (``in_box``), all 8 corners inside the IMX219's pinhole image at the
   start pose (``in_view_imx219``, no occlusion test). Information: IMX519 view, where it lies relative to the start
   (ahead / left / yaw_rel, the frame of scene/randomization_start.yaml).

The driver recomputes every start with its own model and start pose and refuses the file (outside a dry run) if one
differs (> 0.5 mm / 0.1 deg) or fails a check: rebuild it then.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from sim_deploy import REPO, ROOT

DIR = ROOT / "eval" / "start_sets"
DEFAULT_FILE = DIR / "w3.yaml"
SCHEMA = "sim_deploy.start_set/v1"
N_DEMOS = 10
TORCH_PY = Path("~/envs/sim_policy_deploy_torch/bin/python").expanduser()
PROBE = ROOT / "eval" / "split_probe.py"
LOCAL_DATASETS = Path("~/data/umi/datasets").expanduser()   # cluster ~/CapstoneProject/datasets/<label>/<name>/...
DEMO_DATASET = LOCAL_DATASETS / "PICK_TEST1" / "pick1-ep10-v1" / "pick1-ep10-v1.zarr.zip"   # = demos 0-9 of all 9
REAL_FIT = REPO / "sim_evals" / "_pipeline" / "W2" / "sets" / "robot_trials_fit.json"
REAL_OBJECT_MM = [127.0, 51.0, 12.0]                       # the eraser of the 2026-11-20 trials (sim_deploy/eraser.py)
DEFAULT_PLAN = ROOT / "eval" / "batches" / "w3.yaml"
TOL_M, TOL_DEG = 5e-4, 0.1
CHECKS = ("on_table", "in_box", "in_view_imx219")            # required; the rest of `checks` is information


def resolve_file(spec: str | None) -> Path:
    """None / "" -> the default file; a name (w3) -> start_sets/<name>.yaml; else the path."""
    if not spec:
        return DEFAULT_FILE
    p = Path(spec).expanduser()
    if p.suffix in (".yaml", ".yml") or p.exists():
        return p.resolve()
    return DIR / f"{spec}.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _rel(p) -> str:
    p = Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


# ------------------------------------------------------------------------------------------- runs and splits


def run_info(run_folder: str | Path) -> dict:
    """The run's dataset and split parameters (run_config/config.yaml task.dataset) and its checkpoint."""
    run = Path(run_folder).expanduser().resolve()
    cfg = yaml.safe_load((run / "run_config" / "config.yaml").read_text())
    ds = (cfg.get("task") or {}).get("dataset") or {}
    if not ds.get("dataset_path") or "sample_type" not in ds:
        raise ValueError(f"{run}: run_config/config.yaml has no task.dataset.dataset_path / sample_type")
    if int(ds.get("max_segments", -1) or -1) > 0:
        raise ValueError(f"{run}: task.dataset.max_segments {ds['max_segments']} > 0 is not handled")
    ckpts = sorted((run / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"{run}: no checkpoints/*.ckpt")
    task = run.parent.parent.name if run.parent.name == "policy" else run.parent.name
    cluster = str(ds["dataset_path"])
    return {"id": f"{task}/{run.name}", "run_folder": str(run), "checkpoint": str(ckpts[-1]),
            "dataset_cluster_path": cluster, "dataset": Path(cluster).name.split(".zarr")[0],
            "sample_type": str(ds["sample_type"]), "val_ratio": float(ds["val_ratio"]), "seed": int(ds["seed"])}


def local_dataset(cluster_path: str) -> Path:
    """The cluster's .../datasets/<label>/<name>/<name>.zarr.zip -> ~/data/umi/datasets/<label>/<name>/<name>.zarr.zip."""
    parts = Path(cluster_path).parts
    if "datasets" not in parts:
        raise ValueError(f"{cluster_path}: no 'datasets' folder in the path")
    return LOCAL_DATASETS.joinpath(*parts[len(parts) - parts[::-1].index("datasets"):])


def dataset_meta(path: Path) -> dict:
    from sim_deploy.scene.zarr_lowdim import ZarrLowDim

    r = ZarrLowDim(path)
    try:
        def arr(k):
            try:
                return r.array(k).tolist()
            except KeyError:
                return None
        ec = arr("meta/is_task_error_correction")
        return {"episode_names": r.strings("meta/episode_names"), "task_names": r.strings("meta/task_names"),
                "episode_ends": arr("meta/episode_ends"), "task_data_ends": arr("meta/task_data_ends"),
                "task_lengths": arr("meta/task_lengths"),
                "is_task_error_correction": None if ec is None else [bool(x) for x in ec]}
    finally:
        r.close()


def _val_demos(split: dict, names: list[str]) -> list[int] | None:
    """Episode indices with any segment out of training ({'<name>:<task>': is_training}); None if the keys do not
    cover exactly the dataset's episodes (another dataset content than the run's)."""
    by_name: dict[str, list[bool]] = {}
    for k, v in split.items():
        name = k.rsplit(":", 1)[0] if ":" in k else k
        by_name.setdefault(name, []).append(bool(v))
    if set(by_name) != set(names):
        return None
    return [i for i, n in enumerate(names) if not all(by_name[n])]


def training_split(info: dict, path: Path, meta: dict, demo_names: list[str]) -> dict:
    """The checkpoint's split (from the checkpoint and recomputed) and which of the shared demos it held out."""
    req = {"checkpoint": info["checkpoint"], "sample_type": info["sample_type"], "val_ratio": info["val_ratio"],
           "seed": info["seed"], "meta": meta}
    if not TORCH_PY.exists():
        raise FileNotFoundError(f"{TORCH_PY}: the torch env reads the checkpoint's split (sim_deploy/setup_envs.sh)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["CUDA_VISIBLE_DEVICES"] = ""                  # CPU only (torch.load map_location cpu, mmap)
    r = subprocess.run([str(TORCH_PY), str(PROBE)], input=json.dumps(req), capture_output=True, text=True, env=env,
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"split_probe on {info['id']} failed (exit {r.returncode}): {r.stderr.strip()[-800:]}")
    rep = json.loads(r.stdout)
    names = meta["episode_names"]
    ck = rep["checkpoint_split"]
    val_ck = None if ck is None else _val_demos(ck, names)
    val_re = _val_demos(rep["recomputed_split"], names)
    agree = ck is not None and ck == rep["recomputed_split"]
    problems = []
    if ck is None:
        problems.append("the checkpoint has no _extra_training_split_info: the recomputed split is used")
    elif val_ck is None:
        problems.append("the checkpoint's split keys are not this dataset's episodes (another dataset content?)")
    if not agree and ck is not None:
        problems.append("the checkpoint's split differs from get_val_mask recomputed on the local dataset")
    val_names = {names[i] for i in set(val_ck or []) | set(val_re or [])}
    held = [k for k, n in enumerate(demo_names) if n in val_names]
    if len(held) != 1:
        problems.append(f"{len(held)} of the shared demos 0-{len(demo_names) - 1} are val demos (expected 1)")
    return {"id": info["id"], "run_folder": info["run_folder"], "checkpoint": Path(info["checkpoint"]).name,
            "dataset": info["dataset"], "sample_type": info["sample_type"], "val_ratio": info["val_ratio"],
            "seed": info["seed"],
            "split_source": f"checkpoint state_dicts.{rep.get('checkpoint_split_from')}._extra_training_split_info"
            if ck is not None else "recomputed (get_val_mask)",
            "val_demos_checkpoint": None if val_ck is None else [{"index": i, "name": names[i]} for i in val_ck],
            "val_demos_recomputed": [{"index": i, "name": names[i]} for i in (val_re or [])],
            "agree": bool(agree), "held_out": held, "problems": problems}


def shared_demos_check(base: Path, datasets: dict[str, Path]) -> list[dict]:
    """Per dataset: its first N_DEMOS demos have the base dataset's names, row counts, poses and widths."""
    from sim_deploy.scene.zarr_lowdim import load_episode

    bnames = dataset_meta(base)["episode_names"][:N_DEMOS]
    beps = [load_episode(base, i) for i in range(N_DEMOS)]
    out = []
    for name, p in sorted(datasets.items()):
        names = dataset_meta(p)["episode_names"]
        same_names = names[:N_DEMOS] == bnames
        same_data = same_names and all(
            e["rows"] == b["rows"] and np.allclose(e["poses"], b["poses"]) and np.allclose(e["widths"], b["widths"])
            for e, b in ((load_episode(p, i), beps[i]) for i in range(N_DEMOS)))
        out.append({"dataset": name, "path": str(p), "episodes": len(names), "same_names": bool(same_names),
                    "same_data": bool(same_data)})
    return out


# ------------------------------------------------------------------------------------------------- placement


def load_scene(look: str = "robot", model_json=None):
    from sim_deploy import looks
    from sim_deploy.scene.placement import Scene

    return Scene.load(model_json or looks.build_dir(look) / "model.json", home="start")


def _checks(scene, pose) -> dict:
    corner = scene._corner_checks(pose)
    return {"on_table": corner["on_table"], "in_box": corner["in_workspace"],
            "in_view_imx219": corner["in_view_imx219"], "in_view_imx519": corner["in_view_imx519"]}


def _start_frame(scene, pose, yaw_deg: float) -> dict:
    o, f, lft, hdg = scene.start_frame()
    d = np.asarray(pose[:2]) - o
    return {"ahead_m": round(float(d @ f), 4), "left_m": round(float(d @ lft), 4),
            "yaw_rel_deg": round(float((yaw_deg - hdg + 90.0) % 180.0 - 90.0), 2)}


def _common(scene, pose, yaw_deg: float) -> dict:
    checks = _checks(scene, pose)
    return {"eraser": {"x_m": round(float(pose[0]), 4), "y_m": round(float(pose[1]), 4), "yaw_deg": round(yaw_deg, 2)},
            "pose": [round(float(v), 6) for v in pose], "checks": {k: bool(v) for k, v in checks.items()},
            "usable": all(checks[k] for k in CHECKS), "failed": [k for k in CHECKS if not checks[k]],
            "start_frame": _start_frame(scene, pose, yaw_deg)}


def demo_placement(scene, dataset: Path, index: int) -> dict:
    rp = scene.replay_placement(f"replay:{dataset}#{index}")
    out = {"source": "demo", "id": f"demo {index}", "demo": {"index": index, "name": rp["episode_name"]}}
    out.update(_common(scene, rp["pose"], rp["yaw_deg"]))
    out.update({"height_error_m": round(rp["height_error_m"], 4), "approach_tilt_deg": round(rp["approach_tilt_deg"], 1),
                "grasp_tip_above_table_m": round(rp["grasp_tip_above_table_m"], 4),
                "demo_start": {"tilt_deg": round(rp["recorded_start_tilt_deg"], 1),
                               "tip_height_m": round(rp["recorded_start_tip_height_m"], 4)}})
    return out


def _yaw(deg: float) -> float:
    return (float(deg) + 90.0) % 180.0 - 90.0                 # the box is symmetric: [-90, 90)


def real_trial_placement(scene, trial: dict) -> dict:
    """The sim eraser flat on the table at the trial's fitted centre x, y and yaw."""
    yaw = _yaw(trial["yaw_deg"])
    pose = scene.pose(trial["x_m"], trial["y_m"], yaw)
    out = {"source": "real_trial", "id": trial["id"], "real_trial": trial}
    out.update(_common(scene, pose, yaw))
    return out


def real_trials(fit_path: Path) -> list[dict]:
    """The fitted trials of robot_trials_fit.json, sorted by id, as copied into the start-set file."""
    d = json.loads(Path(fit_path).read_text())
    out = []
    for tid in sorted(d["trials"]):
        t = d["trials"][tid]
        q = t["eraser_pose_O"][3:7]
        out.append({"id": tid, "task": t["task"], "run": t["run"], "trial": t["trial"], "folder": t["folder"],
                    "x_m": round(t["x_mm"] / 1000, 5), "y_m": round(t["y_mm"] / 1000, 5),
                    "yaw_deg": round(float(t["yaw_deg"]), 2),
                    "fit_yaw_from_quat_deg": round(float(np.degrees(2 * np.arctan2(q[3], q[0]))), 2),
                    "fit_box_LWT_mm": t["box_LWT_mm"], "fit_rms_px": t["resid"]["rms_px"],
                    "fit_centre_max_px": t["resid"]["centre_max_px"]})
    return out


# ------------------------------------------------------------------------------------------------------ build


def build(infos: list[dict], name: str = "w3", look: str = "robot", fit_path: Path = REAL_FIT, log=print) -> dict:
    from sim_deploy import looks
    from sim_deploy.model.build import git_commit

    base = DEMO_DATASET
    if not base.exists():
        raise FileNotFoundError(f"{base}: the shared demos' dataset is missing")
    bmeta = dataset_meta(base)
    demo_names = bmeta["episode_names"][:N_DEMOS]
    if len(demo_names) != N_DEMOS:
        raise ValueError(f"{base}: {len(demo_names)} demos, expected {N_DEMOS}")
    paths: dict[str, Path] = {}
    for i in infos:
        p = local_dataset(i["dataset_cluster_path"])
        if not p.exists():
            raise FileNotFoundError(f"{p}: the local copy of {i['dataset']} ({i['id']}) is missing")
        paths[i["dataset"]] = p
    log(f"  shared demos: the first {N_DEMOS} of {len(paths)} datasets against {base.name} ...")
    shared = shared_demos_check(base, paths)
    bad = [s["dataset"] for s in shared if not s["same_data"]]
    if bad:
        raise ValueError(f"the first {N_DEMOS} demos of {bad} differ from {base.name}: no shared start set")
    metas = {ds: dataset_meta(p) for ds, p in paths.items()}
    splits = []
    for info in infos:
        log(f"  split of {info['id']} ({info['sample_type']}) ...")
        splits.append(training_split(info, paths[info["dataset"]], metas[info["dataset"]], demo_names))
    scene = load_scene(look)
    starts = [demo_placement(scene, base, k) for k in range(N_DEMOS)]
    trials = real_trials(fit_path)
    starts += [real_trial_placement(scene, t) for t in trials]
    for k, s in enumerate(starts):
        s["rollout"] = k
    st = scene.start
    return {
        "schema": SCHEMA, "name": name,
        "built": {"time": dt.datetime.now().isoformat(timespec="seconds"),
                  "command": "python -m sim_deploy.eval starts build", "git": git_commit(), "look": look,
                  "model_json": _rel(looks.build_dir(look) / "model.json"),
                  "model_sha256": scene.model["model_sha256"], "start_pose_file": _rel(st["file"]),
                  "q_start_rad": [float(x) for x in st["q"]],
                  "start_tcp_O_m": [round(float(v), 4) for v in np.asarray(st["O_T_tcp"])[:3, 3]],
                  "start_tilt_deg": round(float(st["tilt_deg"]), 2),
                  "start_heading_deg": round(float(st["heading_deg"]), 2),
                  "start_tip_height_m": round(float(st["height_above_table_m"]), 4),
                  "eraser_size_m": [float(v) for v in scene.eraser_size]},
        "rule": (f"rollout k -> starts[k], the same for every checkpoint: 0-{N_DEMOS - 1} = the shared demos 0-"
                 f"{N_DEMOS - 1} (the first {N_DEMOS} demos of every training dataset), the eraser at each demo's "
                 f"replay placement from the start pose, flat on the table; {N_DEMOS}-{len(starts) - 1} = the real-arm "
                 f"trials of 2026-11-20, the sim eraser flat at the fitted centre x, y and yaw "
                 f"(sim_deploy/eval/start_set.py)"),
        "demo_source": {"dataset": base.name.split(".zarr")[0], "path": str(base), "size_bytes": base.stat().st_size,
                        "sha256": _sha256(base), "names": demo_names, "shared_with": shared},
        "real_trial_source": {"file": _rel(fit_path), "sha256": _sha256(fit_path),
                              "schema": json.loads(Path(fit_path).read_text()).get("schema"),
                              "real_object_mm": REAL_OBJECT_MM,
                              "note": "the real object was the 127 x 51 x 12 mm eraser; the sim places its own "
                                      "eraser (sim_deploy/eraser.py, 30 mm wide) at the fitted centre and yaw, flat"},
        "checkpoints": splits,
        "starts": starts,
    }


class _NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data) -> bool:
        return True


def write(doc: dict, path: Path | None = None) -> Path:
    out = Path(path) if path else DIR / f"{doc['name']}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# Start set {doc['name']} (sim_deploy/eval/start_set.py; schema {SCHEMA}).\n"
            f"# Written by `python -m sim_deploy.eval starts build`; do not edit by hand (the driver recomputes and "
            f"compares every start).\n# Rollout k of an eval with `--starts set` uses starts[k]. Frame O = the robot "
            f"base, metres; yaw = the eraser's long axis from O x, deg.\n")
    out.write_text(head + yaml.dump(doc, Dumper=_NoAliases, sort_keys=False, default_flow_style=None, width=120))
    return out


def load(path: Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    doc["_path"] = str(Path(path).resolve())
    doc["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return doc


def recompute(scene, s: dict) -> dict:
    if s["source"] == "demo":
        return demo_placement(scene, Path(s["_dataset"]), int(s["demo"]["index"]))
    return real_trial_placement(scene, s["real_trial"])


def verify(doc: dict, scene) -> list[dict]:
    """Per start: recompute its placement with ``scene`` (the eval's model and start pose); problems or []."""
    path = Path(doc["demo_source"]["path"])
    q_bad = np.max(np.abs(np.asarray(scene.start["q"], float) - np.asarray(doc["built"]["q_start_rad"], float))) > 1e-6
    size_bad = not path.exists() or path.stat().st_size != doc["demo_source"]["size_bytes"]
    out = []
    for s in doc["starts"]:
        probs = []
        if q_bad:
            probs.append("the start pose q differs from the file's (stations/start_pose.yaml changed: rebuild)")
        if s["source"] == "demo" and size_bad:
            probs.append(f"{path} missing or its size differs from the file's")
            out.append({"rollout": s["rollout"], "problems": probs})
            continue
        p = recompute(scene, {**s, "_dataset": str(path)})
        dxy = float(np.hypot(p["pose"][0] - s["pose"][0], p["pose"][1] - s["pose"][1]))
        dz = abs(p["pose"][2] - s["pose"][2])
        dyaw = abs((p["eraser"]["yaw_deg"] - s["eraser"]["yaw_deg"] + 90.0) % 180.0 - 90.0)
        if dxy > TOL_M or dz > TOL_M or dyaw > TOL_DEG:
            probs.append(f"recomputed eraser moved {dxy * 1e3:.1f} mm / z {dz * 1e3:.1f} mm / yaw {dyaw:.2f} deg")
        if not p["usable"]:
            probs.append(f"fails {p['failed']} in this scene")
        out.append({"rollout": s["rollout"], "problems": probs, "recomputed": p})
    return out


def checkpoint_entry(doc: dict, run_folder) -> dict | None:
    return next((c for c in doc["checkpoints"] if Path(c["run_folder"]) == Path(run_folder)), None)


def protocol_rules(doc: dict) -> dict:
    """What eval.yaml's protocol hashes: the sources and each rollout's start and eraser pose (not the held-out demo,
    which belongs to the checkpoint: every checkpoint runs the same protocol)."""
    return {"start_set": doc["name"], "file_schema": SCHEMA, "demo_dataset_sha256": doc["demo_source"]["sha256"],
            "real_trial_fit_sha256": doc["real_trial_source"]["sha256"], "q_start_rad": doc["built"]["q_start_rad"],
            "starts": [{"rollout": s["rollout"], "source": s["source"], "id": s["id"],
                        "pose": [round(float(v), 4) for v in s["pose"]]} for s in doc["starts"]]}


def start_for(doc: dict, n: int) -> dict:
    if not 0 <= n < len(doc["starts"]):
        where = doc.get("_path") or doc.get("file") or "the start-set file"
        raise ValueError(f"rollout {n}: {Path(where).name} has starts 0..{len(doc['starts']) - 1}")
    return doc["starts"][n]


# ------------------------------------------------------------------------------------------------- printing


def start_line(s: dict, held: bool | None = None) -> str:
    e, sf = s["eraser"], s["start_frame"]
    what = (f"demo {s['demo']['index']} {s['demo']['name']}" if s["source"] == "demo"
            else f"trial {s['real_trial']['id']}")
    notes = ("FAILS " + ",".join(s["failed"])) if s["failed"] else "ok"
    info = [k for k, v in s["checks"].items() if not v and k not in CHECKS]
    if info:
        notes += f" (info: not {', '.join(info)})"
    if s["source"] == "demo":
        notes += f"; height error {s['height_error_m'] * 100:+.1f} cm"
    if held:
        notes += "; HELD OUT (val demo of this checkpoint)"
    return (f"{s['rollout']:>2} {s['source']:<10} {what:<38} {e['x_m']:7.3f} {e['y_m']:7.3f} {e['yaw_deg']:6.1f}  "
            f"{sf['ahead_m'] * 100:6.1f} {sf['left_m'] * 100:6.1f} {sf['yaw_rel_deg']:6.1f}  {notes}")


def show_lines(doc: dict) -> list[str]:
    d, b, r = doc["demo_source"], doc["built"], doc["real_trial_source"]
    L = [f"start set {doc['name']}: {len(doc['starts'])} starts; built {b['time']} (look {b['look']}, model "
         f"{b['model_sha256'][:12]}, start TCP {b['start_tcp_O_m']}, tilt {b['start_tilt_deg']} deg, "
         f"{b['start_tip_height_m'] * 100:.1f} cm above the table, eraser {b['eraser_size_m']})",
         f"  demos: {d['dataset']} (sha {d['sha256'][:12]}); first {N_DEMOS} identical in "
         f"{sum(s['same_data'] for s in d['shared_with'])}/{len(d['shared_with'])} datasets",
         f"  real trials: {r['file']} (sha {r['sha256'][:12]}); real object {r['real_object_mm']} mm"]
    for c in doc["checkpoints"]:
        vc = [v["index"] for v in c["val_demos_checkpoint"] or []]
        L.append(f"  {c['id']:<30} {c['dataset']:<22} {c['sample_type']:<7} val {vc} "
                 f"{'= recomputed' if c['agree'] else 'DIFFERS from recomputed'}  held-out rollout {c['held_out']}"
                 + (f"  problems: {c['problems']}" if c["problems"] else ""))
    L.append(f"  {'k':>2} {'source':<10} {'demo / trial':<38} {'x':>7} {'y':>7} {'yaw':>6}  {'ahead':>6} {'left':>6} "
             f"{'yawrel':>6}  checks / notes")
    L += ["  " + start_line(s) for s in doc["starts"]]
    return L


# ------------------------------------------------------------------------------------------------------- CLI


def _checkpoint_runs(a) -> list[dict]:
    from sim_deploy.eval import batch

    spec = list(a.checkpoints) if a.checkpoints else batch.load_plan(Path(a.plan or DEFAULT_PLAN))["checkpoints"]
    return batch.resolve_checkpoints(spec)


def main(a) -> int:
    if a.action == "build":
        infos = [run_info(c["run_folder"]) for c in _checkpoint_runs(a)]
        doc = build(infos, name=a.name, look=a.look, fit_path=Path(a.real_fit))
        out = write(doc, resolve_file(a.file) if a.file else None)
        for ln in show_lines(doc):
            print(ln)
        print(f"  -> {_rel(out)}")
        return 0
    f = resolve_file(a.file)
    if not f.exists():
        print(f"{f}: no start-set file (python -m sim_deploy.eval starts build)")
        return 1
    doc = load(f)
    if a.action == "show":
        for ln in show_lines(doc):
            print(ln)
        return 0
    res = verify(doc, load_scene(a.look))
    probs = [r for r in res if r["problems"]]
    print(f"{f.name}: {'OK' if not probs else 'FAIL'} ({len(res)} starts recomputed at look {a.look})")
    for r in probs:
        print(f"  rollout {r['rollout']}: {'; '.join(r['problems'])}")
    return 1 if probs else 0


def add_arguments(p) -> None:
    p.add_argument("action", choices=("build", "show", "check"))
    p.add_argument("file", nargs="?", default=None, help=f"the start-set file or name (default {_rel(DEFAULT_FILE)})")
    p.add_argument("--plan", default=None, help=f"build: the checkpoints of this batch plan (default {DEFAULT_PLAN.name})")
    p.add_argument("--checkpoints", nargs="+", default=None, help="build: TASK/run_NNN ids or run folders instead")
    p.add_argument("--name", default="w3", help="build: the set's name (file start_sets/<name>.yaml)")
    p.add_argument("--real-fit", default=str(REAL_FIT), help="build: the W2 fit of the real trials")
    p.add_argument("--look", default="robot", help="the look whose model places the eraser (default robot)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sim_deploy.eval.start_set")
    add_arguments(ap)
    sys.exit(main(ap.parse_args()))
