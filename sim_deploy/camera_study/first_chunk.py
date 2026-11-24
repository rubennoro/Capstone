"""W2 step 4: a checkpoint's FIRST chunk on a list of observations (real or sim image sets), N paired samples each,
through the copy's own policy code exactly as ``policy_deploy`` builds it.

Per checkpoint (run folder): a frozen station copy is made as the eval driver does (``sim_deploy/eval/driver.py
freeze_station``: ``stations/policy_sim.yaml`` with ``session_root`` in the scratch folder, then the copy's
``policy_deploy use RUN_FOLDER --station <copy>`` in the light env), then ``Settings.load(copy, mock=True)`` and
``build_policy`` (kind diffusion: a plain checkpoint -> DiffusionPolicy, a prompting one -> PromptableDiffusionPolicy).
Precision / diffusion steps / exec_h are the station's (the copy's defaults: fp32, the checkpoint's own steps, exec_h
8) unless ``--precision`` / ``--steps`` / ``--exec-h`` override them (recorded either way). The prompt of a prompting
run is the one ``policy_deploy use`` writes into the station's manager section (``task_name``, ``prompt_index``; the
default index 0), or ``--prompt warmup`` (the longest demo, ``warm_up_prompt()``, what P3's tool used), or
``--prompt-index I``.

Widths: the observations hold training's (handheld) widths. When the station has ``policy.gripper_map`` (the copy's
default), each obs width is first mapped to the robot opening with ``GripperMap.to_robot`` (what the robot would
read), so that ``infer``'s own ``to_model`` gives the model the training width back (checked: max |error| recorded);
the chunk widths then come back in robot metres. ``--gripper-map none`` drops the map (P3's setting: widths straight
through, the chunk widths clipped at the robot's 0.04 m).

Paired sampling: sample k of every observation uses seed ``seed0 + k``: ``reset()`` and (prompting) ``prompt(task,
index)`` once per observation (``--prompt-per sample``: before every sample, as P3's tool; checked bit-identical, the
prompt and predict_action are deterministic and predict_action keeps no state), then per sample ``torch.manual_seed``
/ ``np.random.seed`` / ``random.seed`` and ``infer(obs)``, so every image set gets the same noise draws. No batching
(one ``infer`` per sample, as the deploy).

Jobs file (JSON list): ``{"id": str, "obs": path to an obs.npz (low-dim + images), "main_from": optional obs.npz whose
main images replace them, "ultrawide_from": optional obs.npz for the IMX219 images, "demo": optional {"dataset",
"index", "row"} for the approach metric, or "demo_disp": a (16, 3) m displacement given inline}``. Output per job ``<out>/<id>.npz`` (eef_pos (N, 16, 3), eef_rot6d (N, 16,
6), gripper_width_m (N, 16, 1), seeds, infer_s, settings_key) + ``<id>.json`` (chunk_metrics.chunk_summary); a job
whose files exist with the same settings_key and inputs_sha (sha1 of its input files) is skipped (``--force`` redoes it). ``<out>/run_info.json``: settings,
station, prompt, load time, per-obs time, device, the copy's file states.

Run (torch env, from ``~/Capstone-main-sim``, ``unset PYTHONPATH``; on the GPU only under the shared lock)::

    flock ~/Capstone-main-sim/sim_evals/.gpu.lock \\
      ~/envs/sim_policy_deploy_torch/bin/python -m sim_deploy.camera_study.first_chunk \\
        --run ~/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --jobs JOBS.json --out OUT_DIR [--n 8]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
PD_DIR = REPO / "deployment" / "policy_deploy"
PD_ENV = Path("~/envs/sim_policy_deploy").expanduser()
POLICY_STATION = REPO / "sim_deploy" / "stations" / "policy_sim.yaml"
SCRATCH = REPO / "sim_evals" / "_pipeline" / "W2" / "attribution" / "_work"   # frozen stations (was a session scratch)
OBS_KEYS = ("eef_pos", "eef_rot6d", "eef_rot6d_wrt_start", "gripper_width_m", "main", "ultrawide")
COPY_FILES = ("policy/settings.py", "policy/policy_factory.py", "policy/diffusion_policy.py",
              "policy/promptable_diffusion_policy.py", "policy/model_runner.py", "policy/checkpoint_loader.py",
              "policy/prompt_source.py", "policy/gripper_map.py", "policy/base_policy.py", "contracts/c4_policy.py")


def _env() -> dict:
    import os
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def run_id(run: Path) -> str:
    """<task folder>__<run name>, e.g. PICK_TEST1__run_006_2026-11-18."""
    return f"{run.parent.parent.name}__{run.name}"


def freeze_station(run: Path, out: Path) -> Path:
    """policy_sim.yaml -> out/station.yaml (session_root out/session), then the copy's `policy_deploy use RUN`."""
    out.mkdir(parents=True, exist_ok=True)
    text = POLICY_STATION.read_text()
    text, n = re.subn(r"(?m)^session_root:.*$", f"session_root: {out / 'session'}", text)
    if n != 1:
        raise SystemExit(f"{POLICY_STATION}: no single session_root line")
    st = out / "station.yaml"
    st.write_text(f"# frozen by sim_deploy.camera_study.first_chunk {dt.datetime.now().isoformat(timespec='seconds')} "
                  f"from {POLICY_STATION.relative_to(REPO)}; then `policy_deploy use {run}`\n" + text)
    cmd = [str(PD_ENV / "bin" / "policy_deploy"), "use", str(run), "--station", str(st)]
    r = subprocess.run(cmd, cwd=PD_DIR, capture_output=True, text=True, env=_env())
    (out / "use.log").write_text(f"$ {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")
    if r.returncode != 0:
        raise SystemExit(f"policy_deploy use failed (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-800:]}")
    return st


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_state() -> dict:
    base = PD_DIR / "policy_deploy"
    out = {}
    for f in COPY_FILES:
        p = base / f
        st = p.stat()
        out[f] = {"sha256": _sha(p)[:16], "mtime": dt.datetime.fromtimestamp(st.st_mtime).isoformat()}
    return out


def inputs_sha(job: dict) -> str:
    """sha1 over the bytes of the job's input files (a re-rendered set is redone)."""
    h = hashlib.sha1()
    for k in ("obs", "main_from", "ultrawide_from"):
        if job.get(k):
            h.update(Path(job[k]).read_bytes())
    return h.hexdigest()[:16]


def load_obs(job: dict) -> dict:
    o = np.load(job["obs"])
    d = {k: np.array(o[k]) for k in OBS_KEYS}
    if job.get("main_from"):
        d["main"] = np.array(np.load(job["main_from"])["main"])
    if job.get("ultrawide_from"):
        d["ultrawide"] = np.array(np.load(job["ultrawide_from"])["ultrawide"])
    return d


class ChunkRunner:
    """The copy's policy for one checkpoint, and paired sampling of first chunks."""

    def __init__(self, station: Path, *, device: str | None, precision: str | None, steps: int | None,
                 gripper_map: str, prompt: str, prompt_index: int | None, exec_h: int | None) -> None:
        sys.path.insert(0, str(PD_DIR))
        import torch
        from policy_deploy.policy.policy_factory import build_policy
        from policy_deploy.policy.settings import Settings
        self.torch = torch
        doc = yaml.safe_load(station.read_text())
        st = Settings.load(station, mock=True)
        rep = {}
        if device:
            rep["device"] = device
        if precision:
            rep["precision"] = precision
        if steps:
            rep["diffusion_steps"] = steps
        if gripper_map == "none":
            rep["gripper_map"] = None
        if rep:
            st = dataclasses.replace(st, **rep)
        mgr = doc.get("manager") or {}
        self.exec_h = int(exec_h or mgr.get("exec_h") or st.exec_h or 8)
        if st.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        self.pol = build_policy(st)
        self.load_s = time.monotonic() - t0
        self.st = st
        self.gmap = st.gripper_map
        self.uses_prompt = bool(getattr(self.pol, "USES_PROMPT", False))
        self.prompt = None
        if self.uses_prompt:
            if prompt == "warmup":
                self.prompt = tuple(self.pol.warm_up_prompt())
            else:
                task = mgr.get("task_name")
                idx = prompt_index if prompt_index is not None else int(mgr.get("prompt_index", 0))
                self.prompt = (task, int(idx))
        info = self.pol.describe()
        self.info = {k: info.get(k) for k in ("checkpoint", "epoch", "diffusion_steps_checkpoint",
                                              "diffusion_steps_in_use", "num_train_timesteps", "device", "precision",
                                              "gpu", "torch", "load_s", "gripper_map", "prompt_split",
                                              "prompt_warm_up_demo") if k in info}
        self.settings = {"device": st.device, "precision": st.precision, "diffusion_steps": st.diffusion_steps,
                         "diffusion_steps_in_use": info.get("diffusion_steps_in_use"), "exec_h": self.exec_h,
                         "gripper_map": None if self.gmap is None else self.gmap.as_dict(),
                         "prompt": list(self.prompt) if self.prompt else None, "kind": st.active_kind,
                         "uses_prompt": self.uses_prompt}
        self.prompt_s: list[float] = []

    def reset_and_prompt(self) -> None:
        self.pol.reset()
        if self.uses_prompt:
            t0 = time.monotonic()
            self.pol.prompt(*self.prompt)
            self.prompt_s.append(time.monotonic() - t0)

    def settings_key(self, n: int, seed0: int) -> str:
        s = json.dumps({**self.settings, "n": n, "seed0": seed0}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def observation(self, d: dict):
        from policy_deploy.contracts.c4_policy import Observation
        d = dict(d)
        if self.gmap is not None:
            w_train = d["gripper_width_m"].astype(np.float32)
            w_robot = self.gmap.to_robot(w_train)
            back = self.gmap.to_model(w_robot)
            self.width_roundtrip_err = float(np.max(np.abs(back.astype(np.float64) - w_train)))
            d["gripper_width_m"] = w_robot.astype(np.float32)
        obs = Observation(**{k: d[k] for k in OBS_KEYS})
        obs.check_values()
        return obs

    def sample(self, obs, seed: int, *, reprompt: bool = True):
        """reprompt False: no reset/prompt before this sample (the caller did it for this obs; checked to give
        bit-identical chunks: scratch W2/attr/tools/prompt_equiv.py)."""
        torch = self.torch
        if reprompt:
            self.reset_and_prompt()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        t0 = time.monotonic()
        ch = self.pol.infer(obs)
        dt_s = time.monotonic() - t0
        return np.asarray(ch.eef_pos), np.asarray(ch.eef_rot6d), np.asarray(ch.gripper_width_m), dt_s


def sample_batch(cr: "ChunkRunner", obs, seed: int, n: int):
    """--batch: the n samples of one obs in ONE predict_action call (the obs repeated n times; the copy's ModelRunner
    steps otherwise: model_inputs, autocast, action_pred[:, :16], key_map.split_action, gripper map, make_chunk).
    One seed per obs: the (n, H, D) noise draw differs from n sequential draws, so the samples are not bit-identical
    to the unbatched path, only equally distributed (checked: scratch W2/attr/tools/batch_equiv.py); every image set
    still gets the same noise (paired)."""
    import torch
    from policy_deploy.contracts import ACTION_STEPS
    pol, rn = cr.pol, cr.pol._runner
    if cr.gmap is not None:
        obs = dataclasses.replace(obs, gripper_width_m=cr.gmap.to_model(obs.gripper_width_m))
    t = {k: torch.from_numpy(v).unsqueeze(0).expand(n, *v.shape).contiguous().to(rn.device)
         for k, v in rn.model_inputs(obs).items()}
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t0 = time.monotonic()
    with torch.inference_mode(), rn._autocast():
        res = rn.model.predict_action(t)
    act = res["action_pred"].detach().float().cpu().numpy()
    dt_s = time.monotonic() - t0
    pos, rot, wid = [], [], []
    for b in range(n):
        parts = rn.key_map.split_action(act[b, :ACTION_STEPS], rn._slices)
        p, r, w = (np.ascontiguousarray(parts[f], dtype=np.float32) for f in ("eef_pos", "eef_rot6d", "gripper_width_m"))
        if cr.gmap is not None:
            w = cr.gmap.to_robot(w)
        ch = pol.make_chunk(p, r, w)
        pos.append(np.asarray(ch.eef_pos)); rot.append(np.asarray(ch.eef_rot6d)); wid.append(np.asarray(ch.gripper_width_m))
    return np.stack(pos), np.stack(rot), np.stack(wid), dt_s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="the run folder (policy_deploy use RUN_FOLDER)")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--device", choices=("cuda", "cpu"))
    ap.add_argument("--precision", choices=("fp32", "bf16"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--exec-h", type=int)
    ap.add_argument("--gripper-map", choices=("station", "none"), default="station")
    ap.add_argument("--prompt", choices=("station", "warmup"), default="station")
    ap.add_argument("--prompt-index", type=int)
    ap.add_argument("--prompt-per", choices=("obs", "sample"), default="obs",
                    help="reset + prompt once per observation (default; same chunks) or before every sample (P3)")
    ap.add_argument("--station-dir", type=Path, help="default: scratch W2/attr/stations/<task>__<run>")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch", action="store_true", help="the N samples of an obs in one predict_action call")
    a = ap.parse_args(argv)
    run = a.run.expanduser().resolve()
    sys.path.insert(0, str(REPO))
    from sim_deploy.camera_study.chunk_metrics import chunk_summary, demo_displacement

    jobs = json.loads(a.jobs.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    sdir = a.station_dir or SCRATCH / "stations" / run_id(run)
    station = sdir / "station.yaml"
    if not station.exists():
        freeze_station(run, sdir)
    t_start = time.monotonic()
    cr = ChunkRunner(station, device=a.device, precision=a.precision, steps=a.steps, gripper_map=a.gripper_map,
                     prompt=a.prompt, prompt_index=a.prompt_index, exec_h=a.exec_h)
    if a.batch:
        cr.settings["batched"] = True          # only when set: unbatched keys stay as before
    key = cr.settings_key(a.n, a.seed0)
    todo = []
    for j in jobs:
        j["inputs_sha"] = inputs_sha(j)
        f = a.out / f"{j['id']}.npz"
        if f.exists() and not a.force:
            try:
                z = np.load(f)
                if str(z["settings_key"]) == key and (str(z["inputs_sha"]) == j["inputs_sha"]
                                                     if "inputs_sha" in z.files else True):
                    continue
            except Exception:
                pass
        todo.append(j)
    print(f"{run_id(run)}: kind {cr.settings['kind']} prompt {cr.prompt} settings {cr.settings} key {key}; "
          f"load {cr.load_s:.1f} s; {len(todo)} of {len(jobs)} jobs to do", flush=True)
    # warm-up (CUDA set-up; not recorded)
    if todo:
        cr.sample(cr.observation(load_obs(todo[0])), 0)
    per_obs, all_inf = [], []
    for i, j in enumerate(todo):
        t0 = time.monotonic()
        obs = cr.observation(load_obs(j))
        pos, rot, wid, inf = [], [], [], []
        if a.prompt_per == "obs" or a.batch:
            cr.reset_and_prompt()
        if a.batch:
            pos, rot, wid, s = sample_batch(cr, obs, a.seed0, a.n)
            inf.append(s / a.n)
        else:
            for k in range(a.n):
                p, r, w, s = cr.sample(obs, a.seed0 + k, reprompt=a.prompt_per == "sample")
                pos.append(p); rot.append(r); wid.append(w); inf.append(s)
            pos, rot, wid = np.stack(pos), np.stack(rot), np.stack(wid)
        all_inf += inf
        np.savez(a.out / f"{j['id']}.npz", eef_pos=pos, eef_rot6d=rot, gripper_width_m=wid,
                 seeds=np.arange(a.seed0, a.seed0 + a.n), infer_s=np.array(inf), settings_key=key,
                 inputs_sha=j["inputs_sha"])
        demo = None
        if j.get("demo"):
            dm = j["demo"]
            demo = demo_displacement(dm["dataset"], dm["index"], dm["row"])
        elif j.get("demo_disp") is not None:
            demo = np.asarray(j["demo_disp"], np.float64)
        summ = chunk_summary(pos, demo, cr.exec_h)
        summ.update(id=j["id"], job=j, settings_key=key, width_last_mean_m=float(wid[:, -1, 0].mean()),
                    width_roundtrip_err=getattr(cr, "width_roundtrip_err", None))
        (a.out / f"{j['id']}.json").write_text(json.dumps(summ, indent=1))
        per_obs.append(time.monotonic() - t0)
        msg = f"  [{i + 1}/{len(todo)}] {j['id']:<40} z {summ['z_cm']:+6.2f} (sd {summ['z_cm_std']:.2f})"
        if demo is not None:
            msg += f"  approach {summ['approach_cm']:+6.2f} (demo {summ['demo_along_cm']:.2f})"
        print(msg + f"  {per_obs[-1]:.2f} s", flush=True)
    mem = {}
    if cr.st.device == "cuda":
        mem = {"max_memory_allocated_mb": round(cr.torch.cuda.max_memory_allocated() / 2**20, 1),
               "max_memory_reserved_mb": round(cr.torch.cuda.max_memory_reserved() / 2**20, 1)}
    info_f = a.out / "run_info.json"
    old = json.loads(info_f.read_text()) if info_f.exists() else {}
    hist = old.get("invocations", [])
    hist.append({"time": dt.datetime.now().isoformat(timespec="seconds"), "jobs_file": str(a.jobs),
                 "n_jobs_done": len(todo), "load_s": round(cr.load_s, 2),
                 "per_obs_s_mean": round(float(np.mean(per_obs)), 3) if per_obs else None,
                 "per_sample_infer_s_mean": round(float(np.mean(all_inf)), 4) if all_inf else None, "prompt_s_mean": round(float(np.mean(cr.prompt_s)), 4)
                 if cr.prompt_s else None, "total_s": round(time.monotonic() - t_start, 1), "cuda_mem": mem})
    info = {"run": str(run), "run_id": run_id(run), "station": str(station), "settings": cr.settings,
            "settings_key": key, "n": a.n, "seed0": a.seed0, "model_info": cr.info, "copy_state": copy_state(),
            "invocations": hist}
    info_f.write_text(json.dumps(info, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
