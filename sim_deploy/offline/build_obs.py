"""W1 obs cache: per (package episode, decision row T) the policy inputs two ways, merged and compared. Torch env, CPU
(``~/envs/sim_policy_deploy_torch/bin/python``); the deploy stage runs as a subprocess in the light env (PyAV lives
only there, torch only here).

    cd ~/Capstone-main-sim && unset PYTHONPATH
    ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.offline.build_obs --dataset pick1-v1 \
        --index 14 3 --rows 6 9 36 72 [--run RUN_FOLDER] [--force]

Writes ``sim_evals/_pipeline/W1/obs/<dataset>/ep<ii>_r<row>.npz`` + ``.json`` (the ``.deploy.*`` files of
deploy_path.py stay next to them). Each video is decoded once per episode (all rows in one walk), each dataset is
built once. npz keys: ``deploy_<field>`` (the Observation the policy process received over C4, widths in convention
A; ``deploy_gripper_width_m_B`` = convention B robot widths), ``deploy_main_t_ns`` / ``deploy_ultrawide_t_ns`` (C1
times), ``deploy_anchor_T``, ``deploy_lookup_poses``, ``deploy_start_pose``, ``deploy_traj_at_waypoint_times``;
``direct_<field>`` (UmiTaskDataset.__getitem__ at (episode, T), normalizer not applied), ``direct_wrt_start_clean``,
``action`` (16, 10, ground truth, relative to the pose at T), ``future_T`` (absolute rows T+3k), ``row_T``. The json
holds both paths' metadata, the input comparison (compare.compare_inputs) and the copy's version.

API for step 3: ``build_obs(dataset, indices, rows, run_folder=None, force=False, out_root=OBS_CACHE,
deploy_args=()) -> list[Path]`` and
``load_sample(dataset, index, row) -> (arrays: dict, meta: dict)``; ``sample_paths(dataset)``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sim_deploy.offline.common import (ENV_LIGHT, OBS_CACHE, REPO, clean_env, read_json, set_cpu_threads,
                                       write_json)
from sim_deploy.offline.compare import compare_inputs
from sim_deploy.offline.deploy_path import _station_copy, sample_stem
from sim_deploy.offline.episodes import episode, run_folders


def default_run(dataset: str) -> Path:
    """The run whose cfg builds the dataset: the first plain run trained on it, else the first prompting one."""
    runs = [r for r in run_folders() if r["dataset"] == dataset]
    if not runs:
        raise LookupError(f"no run folder trained on {dataset}")
    runs.sort(key=lambda r: (r["kind"] != "plain", r["run"]))
    return Path(runs[0]["run_folder"])


def used_fields(shape_meta: dict) -> tuple[str, ...]:
    """C4 fields the model reads (KeyMap.used_fields with the copy's policy key_map): ModelRunner.inputs."""
    from policy_deploy.policy.key_map import KeyMap
    from policy_deploy.policy.settings import Settings
    st = Settings.load(_station_copy(None), mock=True)
    return KeyMap.from_config(st.key_map).used_fields(shape_meta)


def run_deploy_stage(dataset: str, indices, rows, out_dir: Path, extra=()) -> None:
    cmd = [str(ENV_LIGHT), "-m", "sim_deploy.offline.deploy_path", "--dataset", dataset, "--index",
           *map(str, indices), "--rows", *map(str, rows), "--out", str(out_dir), *extra]
    print("[build_obs] deploy stage:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO), env=clean_env(), check=True)


_DATASETS: dict = {}


def direct_dataset(dataset: str, run_folder: Path):
    from sim_deploy.offline.direct_path import DirectDataset, run_dataset_cfg
    key = (dataset, str(run_folder))
    if key not in _DATASETS:
        t0 = time.monotonic()
        _DATASETS[key] = DirectDataset(dataset, run_dataset_cfg(run_folder))
        print(f"[build_obs] dataset {dataset} from {Path(run_folder).name} cfg in {time.monotonic() - t0:.1f} s",
              flush=True)
    return _DATASETS[key]


def build_obs(dataset: str, indices, rows, run_folder: Path | None = None, force: bool = False,
              out_root: Path = OBS_CACHE, deploy_args=()) -> list[Path]:
    """deploy_args: extra deploy_path.py flags (e.g. ("--target-lead-ms", "30")); give a matching out_root
    (e.g. W1/obs_lead30ms) so variants do not mix."""
    from sim_deploy.offline.direct_path import cfg_signature
    set_cpu_threads()
    out_dir = out_root / dataset
    run_folder = Path(run_folder) if run_folder else default_run(dataset)
    todo = [i for i in indices if force or any(not (out_dir / f"{sample_stem(i, r)}.npz").is_file()
                                               for r in rows if r < episode(dataset, i).n_steps)]
    need_deploy = [i for i in todo if force or any(not (out_dir / f"{sample_stem(i, r)}.deploy.npz").is_file()
                                                   for r in rows if r < episode(dataset, i).n_steps)]
    if need_deploy:
        run_deploy_stage(dataset, need_deploy, rows, out_dir, deploy_args)
    if not todo:
        return [out_dir / f"{sample_stem(i, r)}.npz" for i in indices for r in rows
                if r < episode(dataset, i).n_steps]
    dd = direct_dataset(dataset, run_folder)
    fields = used_fields(dd.cfg["shape_meta"])
    paths = []
    for i in indices:
        ref = episode(dataset, i)
        assert dd.names[i] == ref.pi_name, (dd.names[i], ref.pi_name)
        for r in rows:
            if r >= ref.n_steps:
                continue
            stem = sample_stem(i, r)
            dz = dict(np.load(out_dir / f"{stem}.deploy.npz"))
            dmeta = read_json(out_dir / f"{stem}.deploy.json")
            s = dd.sample(i, r)
            arrays = dict(dz)
            for k, v in s["obs"].items():
                arrays[f"direct_{k}"] = v
            arrays.update(direct_wrt_start_clean=s["wrt_start_clean"], action=s["action"], future_T=s["future_T"],
                          row_T=s["row_T"])
            meta = dict(dmeta)
            meta["direct"] = {k: v for k, v in s.items() if k not in ("obs", "action", "wrt_start_clean", "future_T",
                                                                      "row_T")}
            meta["direct_setup"] = {"dataset": dataset, "dataset_path": str(dd.path), "cfg_from_run": str(run_folder),
                                    "cfg_signature": cfg_signature(dd.cfg), "sample_type": dd.cfg.get("sample_type"),
                                    "split_in_cfg_run": s["split_in_dataset"]}
            meta["sample"] = {"dataset": dataset, "index": i, "row": r, "pi_name": ref.pi_name,
                              "folder": ref.folder, "slot": ref.slot}
            if dmeta["deploy"]["step"]["type"] == "StepDone":
                meta["inputs"] = compare_inputs(arrays, meta, fields)
            else:
                meta["inputs"] = {"deploy_skipped": dmeta["deploy"]["step"]}
            np.savez(out_dir / f"{stem}.npz", **arrays)
            write_json(out_dir / f"{stem}.json", meta)
            paths.append(out_dir / f"{stem}.npz")
    return paths


def load_sample(dataset: str, index: int, row: int, root: Path = OBS_CACHE) -> tuple[dict, dict]:
    stem = sample_stem(index, row)
    return dict(np.load(root / dataset / f"{stem}.npz")), read_json(root / dataset / f"{stem}.json")


def sample_paths(dataset: str, root: Path = OBS_CACHE) -> list[Path]:
    return sorted(p for p in (root / dataset).glob("ep*_r*.npz") if ".deploy" not in p.name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index", type=int, nargs="+", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--run", type=Path, default=None, help="the run whose cfg.task.dataset builds the dataset")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.monotonic()
    paths = build_obs(a.dataset, a.index, a.rows, a.run, a.force)
    for p in paths:
        m = read_json(p.with_suffix(".json"))
        s = m["inputs"].get("summary", m["inputs"])
        print(p.name, s)
    print(f"[build_obs] {len(paths)} samples in {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
